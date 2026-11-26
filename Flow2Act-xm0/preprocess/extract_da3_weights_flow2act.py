#!/usr/bin/env python3
"""Extract the frozen DA3 backbone and DualDPT used by Flow2Act from Track4World."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file
import torch


@dataclass(frozen=True)
class ArtifactSpec:
    filename: str
    source_prefix: str
    expected_key_count: int
    expected_numel: int | None = None
    expected_shapes: Mapping[str, tuple[int, ...]] | None = None


ARTIFACT_SPECS = (
    ArtifactSpec(
        filename="da3_anyview_giant_track4world_flow2act.safetensors",
        source_prefix="backbone.model.backbone.",
        expected_key_count=675,
        expected_numel=1_136_489_216,
    ),
    ArtifactSpec(
        filename="da3_dualdpt_track4world_flow2act.safetensors",
        source_prefix="backbone.model.head.",
        expected_key_count=162,
        expected_numel=50_115_070,
    ),
)


def unwrap_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    value = payload
    for _ in range(2):
        if not isinstance(value, Mapping):
            break
        envelopes = [key for key in ("state_dict", "model") if key in value]
        if not envelopes:
            break
        if len(envelopes) != 1 or not isinstance(value[envelopes[0]], Mapping):
            raise TypeError(f"Ambiguous checkpoint envelope: {envelopes}")
        value = value[envelopes[0]]
    if not isinstance(value, Mapping):
        raise TypeError("Checkpoint does not contain a state dictionary")
    if any(not isinstance(key, str) or not isinstance(tensor, torch.Tensor) for key, tensor in value.items()):
        raise TypeError("Checkpoint state dictionary contains non-tensor entries")
    return value


def extract_artifact(
    state_dict: Mapping[str, torch.Tensor],
    spec: ArtifactSpec,
) -> dict[str, torch.Tensor]:
    selected: dict[str, torch.Tensor] = {}
    for source_key, source_tensor in state_dict.items():
        if not source_key.startswith(spec.source_prefix):
            continue
        target_key = source_key.removeprefix(spec.source_prefix)
        if not target_key or target_key in selected:
            raise ValueError(f"Invalid key after removing {spec.source_prefix!r}: {source_key!r}")
        tensor = source_tensor.detach().cpu()
        if tensor.dtype != torch.float32:
            raise ValueError(f"Expected FP32 source tensor for {spec.filename}:{target_key}, got {tensor.dtype}")
        selected[target_key] = tensor.contiguous().clone()
    if len(selected) != spec.expected_key_count:
        raise ValueError(f"{spec.source_prefix!r} selected {len(selected)} keys; expected {spec.expected_key_count}")
    if spec.expected_numel is not None:
        actual_numel = sum(tensor.numel() for tensor in selected.values())
        if actual_numel != spec.expected_numel:
            raise ValueError(f"{spec.filename} has {actual_numel} values; expected {spec.expected_numel}")
    if spec.expected_shapes is not None:
        if set(selected) != set(spec.expected_shapes):
            raise ValueError(f"Unexpected keys for {spec.filename}: {sorted(selected)}")
        for key, shape in spec.expected_shapes.items():
            if tuple(selected[key].shape) != shape:
                raise ValueError(f"Unexpected shape for {spec.filename}:{key}: {tuple(selected[key].shape)}")
    for key, tensor in selected.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite tensor in {spec.filename}: {key}")
    return selected


def validate_existing(path: Path, expected: Mapping[str, torch.Tensor]) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Existing output must be a regular file: {path}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(expected):
            raise RuntimeError(f"Existing output has incompatible keys: {path}")
        for key, expected_tensor in expected.items():
            actual = handle.get_tensor(key)
            if actual.dtype != expected_tensor.dtype or tuple(actual.shape) != tuple(expected_tensor.shape):
                raise RuntimeError(f"Existing output has incompatible tensor {key}: {path}")
            if not torch.equal(actual, expected_tensor):
                raise RuntimeError(f"Existing output has different tensor values for {key}: {path}")


def publish_artifact(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    if path.exists() or path.is_symlink():
        validate_existing(path, tensors)
        os.chmod(path, 0o777)
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(dict(tensors), temporary)
        os.chmod(temporary, 0o777)
        os.link(temporary, path, follow_symlinks=False)
        os.chmod(path, 0o777)
    finally:
        temporary.unlink(missing_ok=True)


def extract_all(
    checkpoint: Path,
    output_dir: Path,
    specs: Sequence[ArtifactSpec] = ARTIFACT_SPECS,
) -> list[Path]:
    source = checkpoint.expanduser().resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError(f"Missing Track4World checkpoint: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir.resolve(strict=True)
    if target == Path("/") or target == source.parent:
        raise ValueError(f"Unsafe derived-weight output directory: {target}")
    os.chmod(target, 0o777)

    payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    state_dict = unwrap_state_dict(payload)
    outputs: list[Path] = []
    for spec in specs:
        tensors = extract_artifact(state_dict, spec)
        output = target / spec.filename
        publish_artifact(output, tensors)
        outputs.append(output)
        del tensors
    return outputs


def main() -> None:
    os.umask(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for output in extract_all(args.checkpoint, args.output_dir):
        print(output)


if __name__ == "__main__":
    main()
