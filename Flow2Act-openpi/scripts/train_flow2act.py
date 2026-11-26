# ruff: noqa: N801, N802, N999
"""Single-host, one-process multi-GPU JAX pi05 Flow2Act training."""

import argparse
import dataclasses
import functools
import json
import logging
import os
from pathlib import Path
import sys
import time

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from openpi.flow2act.config_flow2act import Flow2ActConfig


def arguments_flow2act():
    parser = argparse.ArgumentParser(description=__doc__)
    default = Flow2ActConfig()
    for field in dataclasses.fields(default):
        value = getattr(default, field.name)
        flag = "--" + field.name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=value)
        elif isinstance(value, tuple):
            parser.add_argument(flag, type=lambda text: tuple(map(int, text.split(","))), default=value)
        else:
            parser.add_argument(flag, type=type(value), default=value)
    args = parser.parse_args()
    return Flow2ActConfig(**vars(args))


def main_flow2act(cfg):
    cfg.validate()
    os.umask(0)
    for key in ("WORLD_SIZE", "NNODES", "SLURM_NNODES", "MLP_WORKER_NUM", "JAX_PROCESS_COUNT"):
        if int(os.environ.get(key, "1")) != 1:
            raise ValueError(f"Single-host JAX entry: refusing {key}={os.environ[key]}")
    for key in ("RANK", "NODE_RANK", "SLURM_NODEID", "JAX_PROCESS_INDEX"):
        if int(os.environ.get(key, "0")) != 0:
            raise ValueError(f"Single-host JAX entry: refusing {key}={os.environ[key]}")
    import jax
    import numpy as np

    from openpi.flow2act.data_flow2act import Flow2ActDataset
    from openpi.flow2act.data_flow2act import Flow2ActSampler
    from openpi.flow2act.data_flow2act import Flow2ActWindows
    from openpi.flow2act.data_flow2act import model_batch_flow2act
    from openpi.flow2act.stats_flow2act import prepare_stats_flow2act
    from openpi.flow2act.teacher_flow2act import load_teacher_flow2act
    from openpi.flow2act.teacher_flow2act import teacher_targets_flow2act
    from openpi.flow2act.training_flow2act import gradient_flow2act
    from openpi.flow2act.training_flow2act import initialize_flow2act
    from openpi.flow2act.training_flow2act import save_checkpoint_flow2act
    from openpi.flow2act.training_flow2act import teacher_batch_flow2act
    from openpi.flow2act.training_flow2act import update_flow2act
    from openpi.training import sharding

    if jax.process_count() != 1 or jax.local_device_count() != cfg.num_gpus:
        raise ValueError(
            f"Expected one process and {cfg.num_gpus} visible devices, got {jax.process_count()}/{jax.local_device_count()}"
        )
    if any(device.platform != "gpu" for device in jax.devices()):
        raise ValueError("Training entry requires CUDA GPUs; CPU unit tests use the isolated primitives")
    output = Path(cfg.output_dir or cfg.default_output()).resolve()
    cfg = dataclasses.replace(cfg, output_dir=str(output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Fresh run refuses nonempty OUTPUT_DIR: {output}")
    output.mkdir(parents=True, exist_ok=True)
    output.chmod(0o777)
    windows = Flow2ActWindows(cfg.data_root, cfg.action_horizon)
    norm_content = prepare_stats_flow2act(windows, cfg.norm_stats, if_missing=True)
    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    dataset = Flow2ActDataset(cfg, windows=windows)
    logging.info(
        "stage=%s devices=%d fsdp=%d effective GBS=%d GA=%d micro/device=%d windows=%s",
        cfg.stage,
        cfg.num_gpus,
        cfg.fsdp_devices,
        cfg.global_batch_size,
        cfg.ga,
        cfg.global_batch_size // (cfg.num_gpus * cfg.ga),
        dataset.windows.counts,
    )
    state, state_sharding, graphdef, tx = initialize_flow2act(cfg, mesh)
    (output / "config_flow2act.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2) + "\n")
    (output / "norm_stats.json").write_text(norm_content)
    (output / "config_flow2act.json").chmod(0o777)
    (output / "norm_stats.json").chmod(0o777)
    sampler = Flow2ActSampler(len(dataset.windows), cfg.seed)
    teacher = None
    teacher_call = None
    if cfg.stage != "baseline":
        teacher = jax.device_put(load_teacher_flow2act(cfg.teacher_path), replicated)
        teacher_call = jax.jit(
            teacher_targets_flow2act,
            in_shardings=(replicated, data_shard),
            out_shardings=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, sharding.DATA_AXIS)),
        )
    with sharding.set_mesh(mesh):
        grad_call = jax.jit(
            functools.partial(gradient_flow2act, graphdef), out_shardings=(state_sharding.params, replicated)
        )
        update_call = jax.jit(
            functools.partial(update_flow2act, cfg, tx),
            out_shardings=(state_sharding, replicated),
            donate_argnums=(0, 1),
        )
        add_call = jax.jit(lambda a, b: jax.tree.map(lambda x, y: x + y, a, b), donate_argnums=(0, 1))
        prepare = jax.jit(model_batch_flow2act)
        log_path = output / "metrics_flow2act.jsonl"
        try:
            initial_step = int(jax.device_get(state.step))
            last_step = (
                min(cfg.max_steps, initial_step + cfg.stop_after_updates) if cfg.stop_after_updates else cfg.max_steps
            )
            interactive = sys.stderr.isatty()
            progress_format = "{desc}: {percentage:6.2f}%|{bar:20}| {n_fmt}/{total_fmt} [elapsed={elapsed}{postfix}]"
            with (
                log_path.open("a") as log,
                logging_redirect_tqdm(),
                tqdm(
                    total=cfg.max_steps,
                    initial=initial_step,
                    desc="Optimizer updates",
                    unit="update",
                    bar_format=progress_format,
                    dynamic_ncols=True,
                    mininterval=1,
                    miniters=1,
                    disable=not interactive,
                    postfix="update=--, ETA=estimating",
                ) as progress,
            ):
                log_path.chmod(0o777)
                training_started = time.monotonic()
                steady_started = None
                logging.info(
                    "Starting optimizer updates at %d/%d; first update may compile JAX, ETA starts after the second update",
                    initial_step,
                    cfg.max_steps,
                )
                for step in range(initial_step, last_step):
                    started = time.monotonic()
                    serial = sampler.position
                    raw = dataset.batch(sampler.take(cfg.global_batch_size), serial)
                    rng = jax.random.fold_in(jax.random.wrap_key_data(state.rng), step)
                    _, noise_key, time_key = jax.random.split(rng, 3)
                    # Draw for the effective batch before slicing: GA never changes stochastic samples.
                    flow_noise = jax.random.normal(noise_key, raw["actions"].shape)
                    flow_time = jax.random.beta(time_key, 1.5, 1, (cfg.global_batch_size,)) * 0.999 + 0.001
                    normalizers = {
                        f"normalizer{d}": np.mean(np.any(raw[f"mask{d}"], axis=(1, 2, 3)))
                        for d in ("2d", "3d")
                        if f"mask{d}" in raw
                    }
                    gradients, metrics = None, None
                    micro = cfg.global_batch_size // cfg.ga
                    for accumulation in range(cfg.ga):
                        start, end = accumulation * micro, (accumulation + 1) * micro
                        piece = {key: jax.device_put(value[start:end], data_shard) for key, value in raw.items()}
                        if teacher is not None:
                            piece["teacher"] = teacher_batch_flow2act(
                                teacher_call, teacher, piece["teacher_rgb"], data_shard
                            )
                        batch = prepare(piece)
                        batch.update(
                            flow_noise=jax.device_put(flow_noise[start:end], data_shard),
                            flow_time=jax.device_put(flow_time[start:end], data_shard),
                            **normalizers,
                        )
                        grads, info = grad_call(state.params, rng, batch)
                        if cfg.ga != 1:
                            grads = jax.tree.map(lambda x: x / cfg.ga, grads)
                        gradients = grads if gradients is None else add_call(gradients, grads)
                        info = {key: value / cfg.ga for key, value in info.items()}
                        metrics = info if metrics is None else jax.tree.map(lambda x, y: x + y, metrics, info)
                    state, update_info = update_call(state, gradients, sampler.position)
                    jax.block_until_ready(state.step)
                    metrics = {key: float(value) for key, value in {**metrics, **update_info}.items()}
                    if not all(np.isfinite(value) for value in metrics.values()):
                        raise FloatingPointError(f"Non-finite training metric at update {step + 1}: {metrics}")
                    finished = time.monotonic()
                    update_seconds = finished - started
                    # The first JAX update may compile.
                    # Average subsequent wall time, including intervening checkpoint saves.
                    if steady_started is None:
                        steady_started = finished
                        eta_seconds = None
                    else:
                        seconds_per_update = (finished - steady_started) / (step - initial_step)
                        eta_seconds = (cfg.max_steps - step - 1) * seconds_per_update
                    if step + 1 == cfg.max_steps:
                        eta_seconds = 0.0
                    metrics.update(
                        update=step + 1,
                        samples_per_second=cfg.global_batch_size / update_seconds,
                        update_seconds=update_seconds,
                        elapsed_seconds=finished - training_started,
                        eta_seconds=eta_seconds,
                    )
                    eta_text = tqdm.format_interval(eta_seconds) if eta_seconds is not None else "estimating"
                    postfix = f"update={update_seconds:.2f}s, ETA={eta_text}, loss={metrics['loss']:.4f}"
                    progress.set_postfix_str(postfix, refresh=False)
                    progress.update(1)
                    log.write(json.dumps(metrics) + "\n")
                    log.flush()
                    if (step + 1) % cfg.log_interval == 0 or step == initial_step or step + 1 == last_step:
                        if not interactive:
                            logging.info(
                                "%s",
                                tqdm.format_meter(
                                    n=step + 1,
                                    total=cfg.max_steps,
                                    elapsed=metrics["elapsed_seconds"],
                                    prefix="Optimizer updates",
                                    postfix=postfix,
                                    bar_format=progress_format,
                                    ascii=True,
                                ),
                            )
                        logging.info("optimizer update %d/%d %s", step + 1, cfg.max_steps, metrics)
                    if (step + 1) % cfg.save_interval == 0 or step + 1 == cfg.max_steps:
                        saved = save_checkpoint_flow2act(output, state, cfg, norm_content)
                        logging.info("Saved %s", saved)
        finally:
            dataset.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main_flow2act(arguments_flow2act())
