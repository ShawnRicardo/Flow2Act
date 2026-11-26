# ruff: noqa: N801, N802, N999
"""EMA deployment and episode-scoped decision memory; no Teacher or motion cache."""

from collections import deque
import json
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client.base_policy import BasePolicy
import orbax.checkpoint as ocp

from openpi.flow2act.config_flow2act import Flow2ActConfig
from openpi.flow2act.data_flow2act import Flow2ActTokenizer
from openpi.flow2act.data_flow2act import norm_pad_flow2act
from openpi.flow2act.data_flow2act import observation_flow2act
from openpi.flow2act.model_flow2act import Flow2ActPi05
from openpi.shared import nnx_utils
from openpi.shared import normalize


class Flow2ActDecisionMemory:
    def __init__(self, length, cadence):
        self.length, self.cadence = length, cadence
        self.records, self.episodes = {}, {}

    def history(self, env, episode, step):
        if self.episodes.get(env) != episode:
            self.reset(env)
            self.episodes[env] = episode
        records = self.records.get(env, ())
        if records and step < records[-1][0]:
            raise ValueError("Decision time moved backwards within an episode")
        if records and step != records[-1][0] and step - records[-1][0] != self.cadence:
            raise ValueError("Client did not execute the trained decision cadence")
        by_step = dict(records)
        return [by_step.get(step - self.cadence * h) for h in range(1, self.length)]

    def commit(self, env, episode, step, observation):
        self.history(env, episode, step)
        records = self.records.setdefault(env, deque(maxlen=self.length))
        if records and records[-1][0] == step:
            return
        if records and step - records[-1][0] != self.cadence:
            raise ValueError("Client did not execute the trained decision cadence")
        records.append((step, jax.tree.map(lambda x: np.array(x, copy=True), observation)))

    def reset(self, env=None):
        if env is None:
            self.records.clear()
            self.episodes.clear()
        else:
            self.records.pop(env, None)
            self.episodes.pop(env, None)


def load_ema_flow2act(checkpoint, *, use_ema=True):
    path = Path(checkpoint).resolve()
    values = json.loads((path / "config_flow2act.json").read_text())
    values["student_taps"] = tuple(values["student_taps"])
    values.pop("resume_checkpoint", None)  # Ignore obsolete training metadata.
    cfg = Flow2ActConfig(**values)
    cfg.validate()
    shape_module = nnx.eval_shape(lambda: Flow2ActPi05(cfg, nnx.Rngs(0)))
    graphdef, shape = nnx.split(shape_module)
    which = "ema" if use_ema else "params"
    # Empty transforms selects exactly this subtree, leaving optimizer/live weights on disk.
    with ocp.PyTreeCheckpointer() as checkpointer:
        target = {which: shape}
        replicated = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        args = jax.tree.map(lambda _: ocp.ArrayRestoreArgs(restore_type=jax.Array, sharding=replicated), target)
        restored = checkpointer.restore(
            path / "state", args=ocp.args.PyTreeRestore(item=target, restore_args=args, transforms={})
        )
    module = nnx.merge(graphdef, restored[which])
    module.eval()
    stats = normalize.deserialize_json((path / "norm_stats.json").read_text())
    return module, cfg, stats


class Flow2ActPolicy(BasePolicy):
    def __init__(self, module, cfg, stats, *, num_steps=10):
        if num_steps < 1:
            raise ValueError("Flow integration steps must be positive")
        self.cfg, self.stats, self.num_steps = cfg, stats, num_steps
        self.tokenizer = Flow2ActTokenizer(cfg.tokenizer_path)
        self.sample = nnx_utils.module_jit(module.sample_actions)
        self.rng = jax.random.key(cfg.seed)
        self.memory = Flow2ActDecisionMemory(cfg.memory_length, cfg.execute_horizon)

    @property
    def metadata(self):
        return {
            "model": "pi05_JAX_flow2act",
            "stage": self.cfg.stage,
            "action_horizon": self.cfg.action_horizon,
            "execute_horizon": self.cfg.execute_horizon,
            "memory_length": self.cfg.memory_length,
            "requires_episode_id": True,
        }

    def reset(self):
        self.memory.reset()

    def _raw(self, obs):
        rgb = np.stack([np.asarray(obs[key]) for key in ("observation/image", "observation/wrist_image")])
        if rgb.dtype != np.uint8 or rgb.shape != (2, 256, 256, 3):
            raise ValueError("Policy expects full-FOV uint8 main/wrist RGB [256,256,3]")
        state = np.asarray(obs["observation/state"], np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Expected finite native LIBERO state[8]")
        return {"rgb": rgb, "state": norm_pad_flow2act(state, self.stats["state"])}

    def infer(self, obs):
        # Explicit identifiers make retries and episode boundaries unambiguous.
        env, episode, step = str(obs.get("env_id", "0")), str(obs["episode_id"]), int(obs["decision_step"])
        if step < 0 or int(obs.get("execute_horizon", self.cfg.execute_horizon)) != self.cfg.execute_horizon:
            raise ValueError("Invalid decision step or client execution horizon")
        history = self.memory.history(env, episode, step)
        raw = self._raw(obs)
        tokens, mask = self.tokenizer(str(obs["prompt"]))
        current = observation_flow2act(
            jnp.asarray(raw["rgb"])[None],
            jnp.asarray(raw["state"])[None],
            jnp.asarray(tokens)[None],
            jnp.asarray(mask)[None],
        )
        kwargs = {"num_steps": self.num_steps}
        if "noise" in obs:
            noise = np.asarray(obs["noise"], np.float32)
            if noise.shape != (self.cfg.action_horizon, 32) or not np.isfinite(noise).all():
                raise ValueError("Expected finite flow noise [action_horizon,32]")
            kwargs["noise"] = jnp.asarray(noise)[None]
        if self.cfg.motion and history:
            safe = [
                item if item is not None else {"rgb": np.zeros_like(raw["rgb"]), "state": np.zeros(32, np.float32)}
                for item in history
            ]
            valid = jnp.asarray([item is not None for item in history])
            kwargs["history"] = observation_flow2act(
                jnp.asarray(np.stack([item["rgb"] for item in safe])),
                jnp.asarray(np.stack([item["state"] for item in safe])),
                jnp.broadcast_to(tokens, (len(history), len(tokens))),
                jnp.broadcast_to(mask, (len(history), len(mask))),
                valid,
            )
            kwargs["history_valid"] = valid[None]
        next_rng, sample_rng = jax.random.split(self.rng)
        actions = np.asarray(self.sample(sample_rng, current, **kwargs))[0, :, :7]
        stat = self.stats["actions"]
        actions = (actions + 1) / 2 * (stat.q99 - stat.q01 + 1e-6) + stat.q01
        if not np.isfinite(actions).all():
            raise FloatingPointError("Non-finite policy action")
        self.memory.commit(env, episode, step, raw)
        self.rng = next_rng
        return {"actions": actions.astype(np.float32), "execute_horizon": self.cfg.execute_horizon}
