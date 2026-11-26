"""Explicit CPU import of local JAX EMA checkpoints without changing tensor values."""

import dataclasses
import json
from pathlib import Path
import re

import flax.nnx as nnx
from flax.traverse_util import flatten_dict
import jax
import numpy as np
import orbax.checkpoint as ocp

from openpi.flow2act.config_flow2act import Flow2ActConfig
from openpi.flow2act.model_flow2act import Flow2ActPi05
from openpi.flow2act.training_flow2act import permissions_flow2act
from openpi.shared import normalize


def parameter_shape_flow2act(cfg):
    module = nnx.eval_shape(lambda: Flow2ActPi05(cfg, nnx.Rngs(0)))
    return nnx.state(module)


def import_config_flow2act(source, config_path, tokenizer_path):
    if config_path is None:
        candidates = sorted(source.glob("config*.json"))
        if len(candidates) != 1:
            raise ValueError("Provide --config when the source does not contain exactly one config*.json")
        config_path = candidates[0]
    values = json.loads(Path(config_path).read_text())
    controls = {"output_dir", "workers", "log_interval", "stop_after_updates"}
    required = {field.name for field in dataclasses.fields(Flow2ActConfig)} - controls
    if missing := required - values.keys():
        raise ValueError(f"Source checkpoint lacks scientific configuration fields: {sorted(missing)}")
    values["student_taps"] = tuple(values["student_taps"])
    values.pop("resume_checkpoint", None)  # Ignore obsolete training metadata.
    cfg = Flow2ActConfig(**values)
    cfg.validate()
    defaults = Flow2ActConfig()
    # Paths are deployment metadata; all scientific values, including memory, stay saved values.
    return dataclasses.replace(
        cfg,
        output_dir="",
        data_root=defaults.data_root,
        base_params=defaults.base_params,
        teacher_path=defaults.teacher_path,
        motion_cache=defaults.motion_cache,
        norm_stats=f"assets/libero4mix_H{cfg.action_horizon}_flow2act",
        tokenizer_path=str(tokenizer_path or defaults.tokenizer_path),
    )


def validate_import_stats_flow2act(content, cfg):
    raw = json.loads(content)
    stats = normalize.deserialize_json(content)
    for key, dimension in (("state", 8), ("actions", 7)):
        if key not in stats:
            raise ValueError(f"Missing native normalization: {key}")
        for name in ("mean", "std", "q01", "q99"):
            value = getattr(stats[key], name)
            if value is None or value.shape != (dimension,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid native normalization: {key}.{name}")
        if (stats[key].q99 < stats[key].q01).any():
            raise ValueError(f"Invalid normalization quantile range: {key}")
    if "sampling" in raw and raw["sampling"]["action_horizon"] != cfg.action_horizon:
        raise ValueError("Checkpoint normalization horizon differs from its saved action horizon")


def validate_parameters_flow2act(actual, expected):
    actual = flatten_dict(actual.to_pure_dict())
    expected = flatten_dict(expected.to_pure_dict())
    if actual.keys() != expected.keys():
        raise ValueError("Imported EMA parameter keys differ from the configured Flow2Act model")
    for key, value in actual.items():
        reference = expected[key]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"Imported EMA shape/dtype mismatch at {key}")
        flat = value.reshape(-1)
        for start in range(0, flat.size, 1_000_000):
            if not np.isfinite(flat[start : start + 1_000_000]).all():
                raise ValueError(f"Non-finite imported EMA parameter: {key}")


def import_checkpoint_flow2act(source, output, *, config_path=None, tokenizer_path=None):
    source, output = Path(source).resolve(), Path(output).absolute()
    if not source.is_dir():
        raise ValueError("--source must be a local JAX Orbax checkpoint directory, not a Torch .ckpt file")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite imported checkpoint: {output}")
    output = output.resolve()
    if source in output.parents:
        raise ValueError("Import output must be outside the source checkpoint")
    if any(device.platform != "cpu" for device in jax.devices()):
        raise ValueError("Checkpoint import requires JAX_PLATFORMS=cpu")
    commit = json.loads((source / "state/_CHECKPOINT_METADATA").read_text())
    if not commit.get("commit_timestamp_nsecs"):
        raise ValueError("Source checkpoint is not committed")
    cfg = import_config_flow2act(source, config_path, tokenizer_path)
    norm_content = (source / "norm_stats.json").read_text()
    validate_import_stats_flow2act(norm_content, cfg)
    expected = parameter_shape_flow2act(cfg)
    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(source / "state").tree
        if "ema" not in metadata or "step" not in metadata:
            raise ValueError("Source must contain EMA weights and the saved optimizer-update number")
        source_shape = nnx.State(expected.raw_mapping)
        source_root = "Flow2Act"
        if "Flow2Act" in expected:
            candidates = set(metadata["ema"]) - (set(expected) - {"Flow2Act"})
            if len(candidates) != 1:
                raise ValueError("Source auxiliary parameter root is ambiguous or missing")
            source_root = candidates.pop()
            if source_root != "Flow2Act":
                source_shape[source_root] = source_shape.pop("Flow2Act")
        if set(metadata["ema"]) != set(source_shape):
            raise ValueError("Source EMA top-level keys differ from the configured model")
        item = {"ema": source_shape, "step": jax.ShapeDtypeStruct((), metadata["step"].dtype)}
        restore_args = jax.tree.map(lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), item)
        restored = checkpointer.restore(
            source / "state", args=ocp.args.PyTreeRestore(item=item, restore_args=restore_args, transforms={})
        )
    params = restored["ema"]
    if source_root != "Flow2Act":
        params["Flow2Act"] = params.pop(source_root)
    validate_parameters_flow2act(params, expected)
    step = int(restored["step"])
    if step < 1 or output.name != f"checkpoint-{step}":
        raise ValueError(f"Output checkpoint name must match its saved update: checkpoint-{step}")
    if re.fullmatch(r"checkpoint-[1-9][0-9]*", source.name) and source.name != output.name:
        raise ValueError("Source checkpoint directory and saved optimizer-update number differ")
    temporary = output.with_name(output.name + "_flow2act.tmp")
    if temporary.exists():
        raise FileExistsError(f"Existing incomplete import is preserved: {temporary}")
    temporary.mkdir(parents=True, mode=0o777)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(temporary / "state", args=ocp.args.PyTreeSave({"ema": params, "step": restored["step"]}))
    (temporary / "config_flow2act.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2) + "\n")
    (temporary / "norm_stats.json").write_text(norm_content)
    temporary.rename(output)
    output.chmod(0o777)
    permissions_flow2act(output)
    return cfg, step
