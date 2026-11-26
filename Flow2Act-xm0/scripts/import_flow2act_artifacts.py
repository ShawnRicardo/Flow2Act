#!/usr/bin/env python3
"""Explicitly import previously produced artifacts into Flow2Act naming.

Checkpoint imports are portable inference exports, not optimizer resumptions.
Cache and DA3 imports create a new linked view without changing source files.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "xr0")]

CONTRACT_FIELDS = frozenset({
    "model_config", "method_config", "vlm_config", "experiment_config", "action_stats",
})
CACHE_STEMS = {
    "motion_2d": "displacement_normalized",
    "visconf": "native_visconf",
    "in_bounds": "in_bounds",
    "horizon_valid": "horizon_valid",
    "motion_3d": "delta_xyz_normalized",
    "valid_3d": "valid3d",
    "anchor_scale": "anchor_scale",
}


def _fresh_output(source: Path, output: Path) -> tuple[Path, Path]:
    source = source.expanduser().resolve(strict=True)
    output = output.expanduser()
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing to overwrite {output}")
    output = output.resolve(strict=False)
    if output == source or output.is_relative_to(source):
        raise ValueError("Output must be outside the source and must not change it")
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing to overwrite {output}")
    return source, output


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_policy_contract(payload: Mapping) -> dict:
    """Validate a unique complete policy contract before changing its names.

    Historical serialized envelope names are deliberately not imported as Python
    aliases. Discovery is restricted to complete top-level or client-state policy
    contracts; all scientific configuration and tensor restoration stay strict.
    """
    from flow2act.config import Flow2ActConfig
    from mibot.utils.checkpoint_contract_flow2act import SCHEMA_FLOW2ACT

    candidates = []
    for envelope in (payload, payload.get("client_state", {})):
        if not isinstance(envelope, Mapping):
            raise ValueError("Checkpoint client_state must be a mapping")
        candidates.extend(value for value in envelope.values()
                          if isinstance(value, Mapping) and CONTRACT_FIELDS <= value.keys()
                          and all(isinstance(value[field], Mapping) for field in CONTRACT_FIELDS))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one complete policy contract, found {len(candidates)}")
    contract = _json_copy(candidates[0])
    _require(contract.get("kind") == "trained_policy", "Only trained-policy checkpoints can be imported")
    _require(isinstance(contract.get("schema"), str) and bool(contract["schema"]),
             "Source policy contract requires a nonempty schema")
    method, model = contract["method_config"], contract["model_config"]
    try:
        configuration = Flow2ActConfig(**method)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unsupported source method configuration: {error}") from error
    _require(set(method) == set(configuration.to_dict()), "Source method configuration must be complete")
    fixed = {
        "static_alignment": "teacher_bilinear_antialias_to_native_student_grid",
        "static_loss": "cosine", "teacher_view_context": "joint_main_wrist",
        "query_location": "vlm_suffix_from_layer0",
        "action_context": "current_prefix_and_future_query_kv",
        "motion_target": "current_anchor_future_displacement",
    }
    for name, expected in fixed.items():
        _require(method[name] == expected, f"Unsupported source method field: {name}")
    _require(isinstance(model.get("type"), str) and bool(model["type"]),
             "Source model configuration requires a registry type")
    _require(model.get("state_shape") == [1, 32], "Source policy must use the XR0 32D state layout")
    _require(model.get("action_shape") == [method["action_horizon"], 32],
             "Source action shape does not match the method horizon and XR0 action layout")
    for field in ("stage", "future_horizon", "execute_horizon", "memory_length", "student_taps",
                  "geo_loss_weight", "motion_2d_loss_weight", "motion_3d_loss_weight"):
        _require(model.get(field) == method[field], f"Source model/method mismatch: {field}")
    _require(method["state_dim"] == 32, "Source method state width differs from XR0")
    _require(contract["vlm_config"].get("text_config", {}).get("hidden_size") == method["vlm_hidden_dim"],
             "Source VLM hidden width does not match the method")
    # These dimensions and semantics are fixed by the policy constructor. A
    # differently shaped spatial head must not be relabeled as this architecture.
    defaults = Flow2ActConfig().to_dict()
    for field in ("teacher_taps", "teacher_dim", "adapter_hidden_dim", "width", "readout_hidden_dim",
                  "motion_grid_size", "image_size", "motion_2d_beta", "motion_3d_beta",
                  "moving_threshold_px", "readout_horizon_chunk"):
        _require(_json_copy(method[field]) == _json_copy(defaults[field]),
                 f"Unsupported source architecture field: {field}")
    stats = contract["action_stats"]
    _require(stats.get("action_encoding") == "libero_delta_xyz_axis_angle_gripper_signed_v1"
             and stats.get("gripper_encoding") == "signed_1_minus_2g_v1",
             "Source action normalization requires signed LIBERO gripper statistics")
    for field in ("mean", "std"):
        rows = stats.get(field)
        _require(isinstance(rows, list) and len(rows) == method["action_horizon"]
                 and all(isinstance(row, list) and len(row) == 32 for row in rows),
                 f"Source {field} statistics must match the action window")
        _require(all(isinstance(number, (int, float)) and not isinstance(number, bool)
                     and math.isfinite(number) for row in rows for number in row),
                 f"Source {field} statistics must be finite numbers")
    _require(all(value > 0 for row in stats["std"] for value in row[:7])
             and all(value >= 0 for row in stats["std"] for value in row[7:])
             and all(value == 0 for row in stats["mean"] for value in row[7:]),
             "Source action statistics violate the LIBERO padding or positive scale contract")

    experiment = contract["experiment_config"]
    _require(isinstance(experiment.get("model"), dict)
             and isinstance(experiment["model"].get("params"), dict)
             and isinstance(experiment["model"]["params"].get("model"), dict)
             and isinstance(experiment.get("data"), dict),
             "Source experiment requires complete model and data registry configuration")
    experiment_model = experiment["model"]["params"]["model"]
    for field, value in experiment_model.items():
        if field != "type":
            _require(field in model and value == model[field],
                     f"Source experiment/model mismatch: {field}")
    model["type"] = "Flow2ActXR0"
    experiment["model"]["type"] = "BaseRunnerFlow2Act"
    experiment_model["type"] = "Flow2ActXR0"
    experiment["data"]["type"] = "BaseDataModuleFlow2Act"
    contract["schema"] = SCHEMA_FLOW2ACT
    return contract


def checkpoint_model_file(source: Path) -> Path:
    source = source.expanduser().resolve(strict=True)
    if source.is_file():
        return source
    candidates = [source / "checkpoint/mp_rank_00_model_states.pt", source / "mp_rank_00_model_states.pt"]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected one model-state file in {source}, found {len(matches)}")
    return matches[0]


def import_checkpoint(source: Path, output: Path) -> Path:
    """Write inference weights and renamed metadata; discard all optimizer state."""
    import torch
    from mibot.utils.checkpoint_contract_flow2act import CHECKPOINT_KEY_FLOW2ACT

    source, output = _fresh_output(Path(source), Path(output))
    payload = torch.load(checkpoint_model_file(source), map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Source checkpoint must be a mapping")
    contract = canonical_policy_contract(payload)
    state = payload.get("module", payload.get("state_dict"))
    _require(isinstance(state, Mapping) and bool(state)
             and all(isinstance(name, str) and isinstance(tensor, torch.Tensor) for name, tensor in state.items()),
             "Source checkpoint requires a nonempty tensor state dictionary")
    step = payload.get("global_step")
    _require(type(step) is int and step >= 0, "Source checkpoint requires a nonnegative global_step")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation refuses races as well as accidental reuse. Tensor names,
    # values and storage are passed through without dtype conversion or cloning.
    with output.open("xb") as stream:
        torch.save({"module": dict(state), "global_step": step, CHECKPOINT_KEY_FLOW2ACT: contract}, stream)
    return output


def _unique_file(directory: Path, pattern: str) -> Path:
    matches = sorted(path for path in directory.glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {pattern} in {directory}, found {len(matches)}")
    return matches[0]


def _link_file(source: Path, output: Path, mode: str):
    if mode == "symlink":
        output.symlink_to(source.resolve(strict=True))
    elif mode == "hardlink":
        os.link(source.resolve(strict=True), output)
    else:
        raise ValueError("link-mode must be symlink or hardlink")


def validate_cache_metadata(metadata: Mapping) -> int:
    _require(isinstance(metadata, Mapping), "Cache metadata must be an object")
    expected = {
        "coordinate": "displacement_xy_div_wh", "image_hw": [256, 256], "grid_hw": [32, 32],
        "source_camera": "observation.images.image", "canonical_context_frames": 128,
        "anchor_ownership": 78, "decoder_sequence_length": 16, "decoder_stride": 8,
        "decoder_iterations": 4, "decoder_temporal_attention": True, "decoder_overlap_warm_start": True,
        "geometry_batch": 1, "geometry_real_frames_only": True, "decoder_source_self_slot": True,
        "visibility": "native_sigmoid_visibility_and_confidence", "endpoint_bounds_before_float16": True,
        "optional_3d": "native_UV_plus_original_DA3_depth_fullK_C2W_in_current_camera_div_anchor_median_depth",
        "raw_xyz_flow_used": False,
    }
    for field, value in expected.items():
        _require(field in metadata and type(metadata[field]) is type(value) and metadata[field] == value,
                 f"Unsupported cache metadata: {field}")
    _require(type(metadata.get("horizon")) is int and 1 <= metadata["horizon"] <= 50,
             "Cache horizon must be an integer in [1, 50]")
    _require(type(metadata.get("has_motion_3d")) is bool, "Cache must declare whether 3D is available")
    _require(isinstance(metadata.get("schema"), str) and bool(metadata["schema"]),
             "Source cache metadata requires a nonempty schema")
    return metadata["horizon"]


def import_motion_cache(source: Path, output: Path, *, link_mode: str = "symlink") -> Path:
    """Validate array headers and link semantic sidecars into a fresh cache view."""
    import numpy as np
    from flow2act.cache import FILES_FLOW2ACT, SCHEMA_FLOW2ACT, canonical_suite_flow2act

    source, output = _fresh_output(Path(source), Path(output))
    if not source.is_dir():
        raise ValueError("Cache source must be a directory")
    if link_mode not in ("symlink", "hardlink"):
        raise ValueError("link-mode must be symlink or hardlink")
    header = _unique_file(source, "metadata*.json")
    metadata = json.loads(header.read_text())
    horizon = validate_cache_metadata(metadata)
    names = ["motion_2d", "visconf", "in_bounds", "horizon_valid"]
    if metadata["has_motion_3d"]:
        names += ["motion_3d", "valid_3d", "anchor_scale"]
    entries = []
    for episode in sorted(source.glob("libero_*/episode_*")):
        if not episode.is_dir():
            continue
        _require(canonical_suite_flow2act(episode.parent.name) == episode.parent.name,
                 "Cache suites must use canonical LIBERO names")
        _require(episode.name.removeprefix("episode_").isdigit(), f"Invalid episode directory {episode}")
        files = {name: _unique_file(episode, f"{CACHE_STEMS[name]}*.npy") for name in names}
        arrays = {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in files.items()}
        flow = arrays["motion_2d"]
        _require(flow.ndim == 5 and flow.shape[0] > 0, f"Invalid motion array in {episode}")
        count = flow.shape[0]
        shapes = {"motion_2d": (count, horizon, 2, 32, 32), "visconf": (count, horizon, 2, 32, 32),
                  "in_bounds": (count, horizon, 32, 32), "horizon_valid": (count, horizon),
                  "motion_3d": (count, horizon, 3, 32, 32), "valid_3d": (count, horizon, 32, 32),
                  "anchor_scale": (count,)}
        for name, array in arrays.items():
            dtype_valid = (array.dtype in (np.float16, np.float32) if name in
                           ("motion_2d", "visconf", "motion_3d", "anchor_scale") else array.dtype == np.bool_)
            _require(array.shape == shapes[name] and dtype_valid, f"Malformed {name} sidecar in {episode}")
        entries.append((episode.relative_to(source), files))
        del arrays, flow
    _require(bool(entries), "Source cache contains no episode sidecars")
    output.mkdir(parents=True, exist_ok=False)
    for relative, files in entries:
        destination = output / relative
        destination.mkdir(parents=True, exist_ok=False)
        for name, path in files.items():
            _link_file(path, destination / FILES_FLOW2ACT[name], link_mode)
    metadata = _json_copy(metadata)
    metadata["schema"] = SCHEMA_FLOW2ACT
    (output / "metadata_flow2act.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output


def import_da3_weights(source: Path, output: Path, *, link_mode: str = "symlink", specs=None) -> Path:
    """Link the extracted DA3 tensors after checking their safetensors headers."""
    from safetensors import safe_open
    if specs is None:
        from preprocess.extract_da3_weights_flow2act import ARTIFACT_SPECS
        specs = ARTIFACT_SPECS

    source, output = _fresh_output(Path(source), Path(output))
    if not source.is_dir():
        raise ValueError("DA3 source must be a directory containing extracted safetensors")
    if link_mode not in ("symlink", "hardlink"):
        raise ValueError("link-mode must be symlink or hardlink")
    _require(len(specs) == 2, "Both the DA3 backbone and depth head specifications are required")
    pairs = []
    for stem, spec in zip(("da3_anyview_giant", "da3_dualdpt"), specs):
        path = _unique_file(source, f"{stem}*.safetensors")
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            _require(len(keys) == spec.expected_key_count, f"Unexpected tensor count in {path}")
            shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in keys}
            _require(all(handle.get_slice(key).get_dtype() == "F32" for key in keys),
                     f"Expected extracted FP32 DA3 weights in {path}")
            if spec.expected_numel is not None:
                _require(sum(math.prod(shape) for shape in shapes.values()) == spec.expected_numel,
                         f"Unexpected tensor dimensions in {path}")
            if spec.expected_shapes is not None:
                _require(shapes == dict(spec.expected_shapes), f"Unexpected tensor shapes in {path}")
        pairs.append((path, spec.filename))
    output.mkdir(parents=True, exist_ok=False)
    for path, filename in pairs:
        _link_file(path, output / filename, link_mode)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("checkpoint", "Export a trained policy as a Flow2Act inference-only checkpoint"),
        ("motion-cache", "Create a validated Flow2Act-named view of an existing motion cache"),
        ("da3-weights", "Create a Flow2Act-named view of extracted DA3 backbone and depth head weights"),
    ):
        sub = subparsers.add_parser(name, help=description, description=description)
        sub.add_argument("--source", type=Path, required=True, help="Existing source artifact; never modified")
        sub.add_argument("--output", type=Path, required=True, help="Fresh output file or directory; never overwritten")
        if name != "checkpoint":
            sub.add_argument("--link-mode", choices=("symlink", "hardlink"), default="symlink",
                             help="Default symlink; hardlinks require the same filesystem")
    return parser.parse_args(argv)


def main(argv=None):
    os.umask(0)
    args = parse_args(argv)
    if args.command == "checkpoint":
        output = import_checkpoint(args.source, args.output)
        print(f"Inference-only checkpoint written to {output}; optimizer resume state was not exported")
    elif args.command == "motion-cache":
        print(import_motion_cache(args.source, args.output, link_mode=args.link_mode))
    else:
        print(import_da3_weights(args.source, args.output, link_mode=args.link_mode))


if __name__ == "__main__":
    main()
