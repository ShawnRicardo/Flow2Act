# ruff: noqa: N801, N802, N999
"""Explicit real B=1 full-size forward/backward smoke, without optimizer allocation.

This does not replace the 4-GPU GBS256 20-update gate. CPU is an explicit option.
"""

import argparse
import dataclasses
import json
import os
from pathlib import Path
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.flow2act.config_flow2act import Flow2ActConfig
from openpi.flow2act.data_flow2act import Flow2ActDataset
from openpi.flow2act.data_flow2act import Flow2ActWindows
from openpi.flow2act.data_flow2act import model_batch_flow2act
from openpi.flow2act.model_flow2act import Flow2ActPi05
from openpi.flow2act.stats_flow2act import prepare_stats_flow2act
from openpi.flow2act.teacher_flow2act import load_teacher_flow2act
from openpi.flow2act.teacher_flow2act import teacher_targets_flow2act
from openpi.flow2act.training_flow2act import check_base_flow2act
from openpi.flow2act.training_flow2act import gradient_flow2act
from openpi.flow2act.training_flow2act import permissions_flow2act
from openpi.models.model import restore_params


def main_flow2act():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("baseline", "static", "motion2d", "motion3d"), default="motion3d")
    parser.add_argument("--memory-length", type=int, default=Flow2ActConfig().memory_length)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    os.umask(0)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    output.chmod(0o777)
    cfg = dataclasses.replace(
        Flow2ActConfig(),
        stage=args.stage,
        workers=0,
        augment=False,
        memory_length=args.memory_length,
        num_gpus=1,
        fsdp_devices=1,
        global_batch_size=1,
    )
    started = time.monotonic()
    windows = Flow2ActWindows(cfg.data_root, cfg.action_horizon)
    prepare_stats_flow2act(windows, cfg.norm_stats, if_missing=True)
    dataset = Flow2ActDataset(cfg, windows=windows)
    raw = dataset.batch(np.array([30]), 0)
    dataset.close()
    batch = model_batch_flow2act(raw)
    if cfg.stage != "baseline":
        teacher = jax.device_put(load_teacher_flow2act(cfg.teacher_path))
        dtype = jnp.float16 if jax.devices()[0].platform == "gpu" else jnp.float32
        target = jax.jit(lambda w, rgb: teacher_targets_flow2act(w, rgb, compute_dtype=dtype))(
            teacher, raw["teacher_rgb"]
        )
        batch["teacher"] = jax.block_until_ready(target)
        del teacher

    def create(key, partial):
        module = Flow2ActPi05(cfg, nnx.Rngs(key))
        graph, params = nnx.split(module)
        params.replace_by_pure_dict(partial)
        return params, graph

    shape_module = nnx.eval_shape(lambda: Flow2ActPi05(cfg, nnx.Rngs(0)))
    shape = nnx.state(shape_module).to_pure_dict()
    loaded = restore_params(cfg.base_params, restore_type=np.ndarray)
    check_base_flow2act(loaded, shape)
    params, graph = jax.jit(create)(jax.random.key(0), loaded)
    del loaded
    print("Full-size base loaded strictly; starting real B=1 backward", flush=True)
    gradients, metrics = jax.jit(lambda p, b: gradient_flow2act(graph, p, jax.random.key(cfg.seed), b))(params, batch)
    jax.block_until_ready(metrics)
    finite = all(bool(jnp.isfinite(value).all()) for value in jax.tree.leaves(gradients))
    if not finite or not all(bool(jnp.isfinite(value)) for value in metrics.values()):
        raise FloatingPointError("Non-finite full-size B=1 forward/backward")
    report = {
        **{k: float(v) for k, v in metrics.items()},
        "finite_gradients": finite,
        "stage": cfg.stage,
        "memory_length": cfg.memory_length,
        "batch_size": 1,
        "backbone": "native_full_size",
        "platform": jax.devices()[0].platform,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "smoke_flow2act.json").write_text(json.dumps(report, indent=2) + "\n")
    permissions_flow2act(output)
    print(report)


if __name__ == "__main__":
    main_flow2act()
