# ruff: noqa: N801, N802, N999
"""Derived pi05: unchanged base scopes, input-layer Future queries and detached memory."""

import flax.nnx as nnx
import flax.nnx.bridge as bridge
import jax
import jax.numpy as jnp

from openpi.flow2act.gemma_flow2act import Flow2ActModule
from openpi.flow2act.modules_flow2act import Flow2ActFuture
from openpi.flow2act.modules_flow2act import Flow2ActStaticAdapter
from openpi.flow2act.modules_flow2act import motion_loss_flow2act
from openpi.flow2act.modules_flow2act import positions_flow2act
from openpi.flow2act.modules_flow2act import spatial_flow2act
from openpi.flow2act.modules_flow2act import static_loss_flow2act
from openpi.models import gemma
from openpi.models import model
from openpi.models import pi0
from openpi.models import siglip


class Flow2ActPi05(pi0.Pi0):
    def __init__(self, cfg, rngs, *, base_config=None):
        base = base_config or cfg.model_config()
        if not base.pi05 or base.action_dim != 32 or base.discrete_state_input:
            raise ValueError("Flow2Act requires native pi05, action_dim=32, discrete_state_input=False")
        if cfg.stage == "baseline":
            super().__init__(base, rngs)
        else:
            model.BaseModel.__init__(self, base.action_dim, base.action_horizon, base.max_token_len)
            self.pi05 = True
            vlm = gemma.get_config(base.paligemma_variant)
            expert = gemma.get_config(base.action_expert_variant)
            if max(cfg.student_taps) >= vlm.depth:
                raise ValueError("Student tap exceeds VLM depth")
            llm = bridge.ToNNX(
                Flow2ActModule(configs=[vlm, expert], embed_dtype=base.dtype, adarms=True, taps=cfg.student_taps)
            )
            llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])
            img = bridge.ToNNX(
                siglip.Module(
                    num_classes=vlm.width, variant="So400m/14", pool_type="none", scan=True, dtype_mm=base.dtype
                )
            )
            img.lazy_init(next(iter(base.fake_obs().images.values())), train=False, rngs=rngs)
            self.PaliGemma = nnx.Dict(llm=llm, img=img)
            self.action_in_proj = nnx.Linear(32, expert.width, rngs=rngs)
            self.time_mlp_in = nnx.Linear(expert.width, expert.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(expert.width, expert.width, rngs=rngs)
            self.action_out_proj = nnx.Linear(expert.width, 32, rngs=rngs)
            self.deterministic = True
            # This RNG never advances the base initialization or training/data streams.
            extra_rngs = nnx.Rngs(jax.random.fold_in(jax.random.key(cfg.seed), 0x543457))
            self.Flow2Act = nnx.Dict(static=Flow2ActStaticAdapter(vlm.width, extra_rngs))
            if cfg.motion:
                self.Flow2Act.future = Flow2ActFuture(vlm.width, cfg, extra_rngs)
        self.cfg = cfg

    def extended_prefix(self, obs, history=None, history_valid=None):
        prefix, valid, ar = self.embed_prefix(obs)
        current_n = prefix.shape[1]
        visible = valid
        if not self.cfg.motion:
            return prefix, valid, ar, visible, current_n
        future = self.Flow2Act.future
        b, _, width = prefix.shape
        extra, masks = [], []
        if self.cfg.memory_length > 1:
            if history is None or history_valid is None:
                raise ValueError("Fixed-shape history and validity are required for memory_length > 1")
            tokens, hist_mask, hist_ar = self.embed_prefix(history)
            (encoded, _), _ = self.PaliGemma.llm(
                [tokens, None], positions=jnp.cumsum(hist_mask, -1) - 1, mask=pi0.make_attn_mask(hist_mask, hist_ar)
            )
            encoded = jax.lax.stop_gradient(encoded[:, :512]).reshape(b, self.cfg.memory_length - 1, 2, 256, width)
            encoded = future.memory_proj(future.memory_norm(encoded.astype(jnp.float32)))
            encoded += future.camera.value[None, None, :, None]
            encoded += spatial_flow2act(16, width).reshape(1, 1, 1, 256, width)
            ages = positions_flow2act(jnp.arange(1, self.cfg.memory_length) * self.cfg.execute_horizon, width)
            encoded += ages[None, :, None, None]
            past_state = jax.lax.stop_gradient(history.state).reshape(b, self.cfg.memory_length - 1, 32)
            for h in range(self.cfg.memory_length - 1):
                keep = history_valid[:, h]
                visual = jnp.where(keep[:, None, None], encoded[:, h].reshape(b, 512, width), 0)
                state = jnp.where(keep[:, None], future.state(past_state[:, h]) + ages[h], 0)
                extra.extend((visual, state[:, None]))
                masks.extend((jnp.broadcast_to(history_valid[:, h, None], (b, 512)), history_valid[:, h, None]))
        extra.append(future.state(obs.state)[:, None])
        masks.append(jnp.ones((b, 1), bool))
        memory_n = sum(x.shape[1] for x in extra)
        extra.append(jnp.broadcast_to(future.queries.value[None], (b, self.cfg.future_horizon, width)))
        masks.append(jnp.ones((b, self.cfg.future_horizon), bool))
        suffix_valid = jnp.concatenate(masks, 1)
        prefix = jnp.concatenate((prefix, *extra), 1)
        valid = jnp.concatenate((valid, suffix_valid), 1)
        ar = jnp.concatenate((ar, jnp.ones(suffix_valid.shape[1], bool)))
        # The action expert reads current prefix and query KV, never direct memory/state KV.
        visible = jnp.concatenate((visible, jnp.zeros((b, memory_n), bool), masks[-1]), 1)
        return prefix, valid, ar, visible, current_n

    def objective(self, rng, batch):
        obs, actions = batch["observation"], batch["actions"]
        if self.cfg.stage == "baseline" and "flow_noise" not in batch:
            action = jnp.mean(super().compute_loss(rng, obs, actions, train=False))
            return action, {
                "action": action,
                "static": jnp.float32(0),
                "motion2d": jnp.float32(0),
                "motion3d": jnp.float32(0),
            }
        # Keep the native random split, noise and beta-time objective exactly.
        _, noise_rng, time_rng = jax.random.split(rng, 3)
        noise = batch.get("flow_noise")
        time = batch.get("flow_time")
        if noise is None:
            noise = jax.random.normal(noise_rng, actions.shape)
            time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        noisy = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        prefix, valid, ar, visible, current_n = self.extended_prefix(
            obs, batch.get("history"), batch.get("history_valid")
        )
        suffix, smask, sar, cond = self.embed_suffix(obs, noisy, time)
        all_valid = jnp.concatenate((valid, smask), 1)
        mask = pi0.make_attn_mask(all_valid, jnp.concatenate((ar, sar)))
        mask = mask.at[:, prefix.shape[1] :, : prefix.shape[1]].set(
            mask[:, prefix.shape[1] :, : prefix.shape[1]] & visible[:, None]
        )
        kwargs = {} if self.cfg.stage == "baseline" else {"return_features": True}
        output = self.PaliGemma.llm(
            [prefix, suffix], positions=jnp.cumsum(all_valid, -1) - 1, mask=mask, adarms_cond=[None, cond], **kwargs
        )
        prefix_out, action_out = output[0]
        predicted = self.action_out_proj(action_out[:, -self.action_horizon :])
        action_loss = jnp.mean(jnp.square(predicted - (noise - actions)))
        if self.cfg.stage == "baseline":
            return action_loss, {
                "action": action_loss,
                "static": jnp.float32(0),
                "motion2d": jnp.float32(0),
                "motion3d": jnp.float32(0),
            }
        taps = output[2]
        b = actions.shape[0]
        adapted = self.Flow2Act.static(taps).reshape(4, b, 2, 16, 16, 3072)
        static_loss = static_loss_flow2act(adapted, batch["teacher"])
        loss2 = loss3 = jnp.float32(0)
        if self.cfg.motion:
            pred2, pred3 = self.Flow2Act.future.readout(prefix_out[:, :256], prefix_out[:, -self.cfg.future_horizon :])
            loss2 = motion_loss_flow2act(
                pred2,
                batch["motion2d"],
                batch["mask2d"],
                self.cfg.motion_beta2d,
                balance=True,
                threshold=self.cfg.moving_threshold,
                normalizer=batch.get("normalizer2d"),
            )
            if self.cfg.stage == "motion3d":
                loss3 = motion_loss_flow2act(
                    pred3,
                    batch["motion3d"],
                    batch["mask3d"],
                    self.cfg.motion_beta3d,
                    normalizer=batch.get("normalizer3d"),
                )
        total = (
            action_loss
            + self.cfg.static_weight * static_loss
            + self.cfg.motion2d_weight * loss2
            + self.cfg.motion3d_weight * loss3
        )
        return total, {"action": action_loss, "static": static_loss, "motion2d": loss2, "motion3d": loss3}

    def compute_loss(self, rng, observation, actions, *, train=False):
        if self.cfg.stage != "baseline":
            raise ValueError("Auxiliary training requires objective(rng, batch), including supervision")
        return super().compute_loss(rng, observation, actions, train=False)

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None, history=None, history_valid=None):
        if not self.cfg.motion:
            return super().sample_actions(rng, observation, num_steps=num_steps, noise=noise)
        observation = model.preprocess_observation(None, observation, train=False)
        prefix, valid, ar, visible, _ = self.extended_prefix(observation, history, history_valid)
        _, cache = self.PaliGemma.llm(
            [prefix, None], positions=jnp.cumsum(valid, -1) - 1, mask=pi0.make_attn_mask(valid, ar)
        )
        b = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (b, self.action_horizon, 32))
        dt = -1.0 / num_steps

        def step(carry):
            x, time = carry
            suffix, smask, sar, cond = self.embed_suffix(observation, x, jnp.broadcast_to(time, b))
            mask = jnp.concatenate(
                (
                    jnp.broadcast_to(visible[:, None], (b, suffix.shape[1], visible.shape[1])),
                    pi0.make_attn_mask(smask, sar),
                ),
                -1,
            )
            pos = jnp.sum(valid, -1)[:, None] + jnp.cumsum(smask, -1) - 1
            (_, out), _ = self.PaliGemma.llm(
                [None, suffix], positions=pos, mask=mask, kv_cache=cache, adarms_cond=[None, cond]
            )
            return x + dt * self.action_out_proj(out[:, -self.action_horizon :]), time + dt

        return jax.lax.while_loop(lambda c: c[1] >= -dt / 2, step, (noise, 1.0))[0]
