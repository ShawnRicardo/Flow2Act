# ruff: noqa: N801, N802, N999
"""FSDP training primitives and explicit atomic checkpoints, isolated from native train.py."""

import dataclasses
import json
import os
from pathlib import Path

import flax.nnx as nnx
import flax.struct as struct
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from openpi.flow2act.model_flow2act import Flow2ActPi05
from openpi.models.model import restore_params
from openpi.training import sharding
from openpi.training.optimizer import AdamW


@struct.dataclass
class Flow2ActState:
    step: object
    params: object
    ema: object
    opt_state: object
    rng: object
    sampler_position: object


def teacher_batch_flow2act(teacher_call, teacher, rgb, data_sharding):
    """One synchronized view pair/device/call, preserving sample order and layout."""
    chunk_size = len(data_sharding.device_set)
    if rgb.shape[0] == 0 or rgb.shape[0] % chunk_size:
        raise ValueError("Teacher batch must be a positive multiple of the device count")
    targets = []
    for start in range(0, rgb.shape[0], chunk_size):
        # Eager slices of a committed batch-sharded array can become replicated.
        # jit(in_shardings=...) requires the committed argument to match exactly.
        chunk = jax.device_put(rgb[start : start + chunk_size], data_sharding)
        targets.append(teacher_call(teacher, chunk))
    target_sharding = jax.sharding.NamedSharding(
        data_sharding.mesh, jax.sharding.PartitionSpec(None, sharding.DATA_AXIS)
    )
    # Partition concat inside its compiled plan, without a full replicated target.
    return jax.jit(jnp.concatenate, static_argnames=("axis",), out_shardings=target_sharding)(targets, axis=1)


def check_base_flow2act(loaded, expected):
    expected = {k: v for k, v in flatten_dict(expected).items() if k[0] != "Flow2Act"}
    loaded = flatten_dict(loaded)
    if set(loaded) != set(expected):
        raise ValueError(
            f"Base key mismatch: missing={set(expected) - set(loaded)}, extra={set(loaded) - set(expected)}"
        )
    for key, reference in expected.items():
        value = loaded[key]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"Base shape/dtype mismatch at {key}: {value.shape}/{value.dtype}, expected {reference}")
        flat = value.reshape(-1)
        for start in range(0, flat.size, 1_000_000):
            if not np.isfinite(flat[start : start + 1_000_000]).all():
                raise ValueError(f"Non-finite base weight: {key}")


def initialize_flow2act(cfg, mesh, *, base_config=None):
    tx = AdamW().create(cfg.schedule)
    init_key = jax.random.fold_in(jax.random.key(cfg.seed), 0x42415345)

    def initialize(rng, partial=None):
        module = Flow2ActPi05(cfg, nnx.Rngs(rng), base_config=base_config)
        graphdef, params = nnx.split(module)
        if partial is not None:
            params.replace_by_pure_dict(partial)
        return Flow2ActState(
            jnp.int32(0), params, params, tx.init(params), jax.random.key_data(jax.random.key(cfg.seed)), jnp.int32(0)
        ), graphdef

    # GraphDef is static; its base weights have exactly the native scanned paths.
    shape, graphdef = jax.eval_shape(initialize, init_key)
    state_sharding = sharding.fsdp_sharding(shape, mesh)
    loaded = restore_params(cfg.base_params, restore_type=np.ndarray)
    check_base_flow2act(loaded, shape.params.to_pure_dict())
    partial_sharding = sharding.fsdp_sharding(loaded, mesh)
    state, _ = jax.jit(
        initialize,
        in_shardings=(jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()), partial_sharding),
        out_shardings=(state_sharding, None),
    )(init_key, loaded)
    return state, state_sharding, graphdef, tx


def gradient_flow2act(graphdef, params, rng, batch):
    def objective(params):
        module = nnx.merge(graphdef, params)
        module.train()
        return module.objective(rng, batch)

    (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(params)
    return grads, {"loss": loss, **metrics}


def update_flow2act(cfg, tx, state, grads, sampler_position):
    updates, opt_state = tx.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    ema = jax.tree.map(lambda old, new: cfg.ema_decay * old + (1 - cfg.ema_decay) * new, state.ema, params)
    result = state.replace(
        step=state.step + 1,
        params=params,
        ema=ema,
        opt_state=opt_state,
        sampler_position=jnp.asarray(sampler_position, state.sampler_position.dtype),
    )
    return result, {"grad_norm": optax.global_norm(grads), "lr": cfg.schedule(state.step)}


def permissions_flow2act(root):
    """Only an exact artifact subtree, without traversing links or filesystems."""
    root = Path(root)
    device = root.stat().st_dev
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [
            name
            for name in dirs
            if not (Path(directory) / name).is_symlink() and (Path(directory) / name).stat().st_dev == device
        ]
        for path in [Path(directory), *(Path(directory) / name for name in files)]:
            if not path.is_symlink() and path.stat().st_dev == device:
                path.chmod(0o777)


def save_checkpoint_flow2act(output, state, cfg, norm_content):
    step = int(jax.device_get(state.step))
    parent = Path(output).resolve() / "checkpoints"
    parent.mkdir(parents=True, exist_ok=True)
    parent.chmod(0o777)
    final = parent / f"checkpoint-{step}"
    temporary = parent / f"checkpoint-{step}_flow2act.tmp"
    if final.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {final} / {temporary}")
    temporary.mkdir(mode=0o777)
    with ocp.Checkpointer(ocp.StandardCheckpointHandler()) as checkpointer:
        checkpointer.save(temporary / "state", args=ocp.args.StandardSave(state))
    (temporary / "config_flow2act.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2) + "\n")
    (temporary / "norm_stats.json").write_text(norm_content)
    temporary.rename(final)
    final.chmod(0o777)
    permissions_flow2act(final)
    return final
