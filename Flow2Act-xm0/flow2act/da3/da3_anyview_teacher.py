# ruff: noqa: N999, N802, N812, FBT001, FBT002, FBT003, SLF001
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""Frozen DA3-Giant teacher with an optional differentiable frozen DualDPTFlow2Act."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from safetensors import safe_open
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .da3_dinov2 import DinoV2Flow2Act
from .da3_dualdpt import DualDPTFlow2Act

DA3_LAYER_IDS_FLOW2ACT = (19, 27, 33, 39)
DA3_PATCH_SIZE_FLOW2ACT = 14
DA3_FEATURE_DIM_FLOW2ACT = 3072
DA3_BACKBONE_FILENAME_FLOW2ACT = "da3_anyview_giant_track4world_flow2act.safetensors"
DA3_HEAD_FILENAME_FLOW2ACT = "da3_dualdpt_track4world_flow2act.safetensors"


def _uniform_grid_pair_flow2act(
    value: Sequence[int] | torch.Tensor,
    *,
    name: str,
    device: torch.device,
) -> tuple[int, int]:
    if torch.is_tensor(value):
        grid = value.to(device=device, dtype=torch.long)
        if grid.ndim == 3:
            torch._assert_async(
                torch.all(grid == grid[0, 0]),
                f"{name} requires one uniform grid per batch",
            )
            grid = grid[0, 0]
        if grid.shape != (2,):
            raise ValueError(f"{name} must be [2] or uniform [B,V,2]")
        torch._assert_async(torch.all(grid > 0), f"{name} must be positive")
        return int(grid[0].item()), int(grid[1].item())
    grid = tuple(value)
    if (
        len(grid) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in grid)
    ):
        raise ValueError(f"{name} must contain two positive integers")
    return int(grid[0]), int(grid[1])


def build_da3_anyview_giant_flow2act(
    layer_ids: Sequence[int] = DA3_LAYER_IDS_FLOW2ACT,
) -> DinoV2Flow2Act:
    selected = tuple(int(layer_id) for layer_id in layer_ids)
    if selected not in (DA3_LAYER_IDS_FLOW2ACT, (DA3_LAYER_IDS_FLOW2ACT[-1],)):
        raise ValueError("DA3 teacher layers must be (19,27,33,39) or the single post-block (39,)")
    return DinoV2Flow2Act(
        name="vitg",
        out_layers=list(selected),
        alt_start=13,
        qknorm_start=13,
        rope_start=13,
        cat_token=True,
    )


def build_da3_dualdpt_flow2act() -> DualDPTFlow2Act:
    return DualDPTFlow2Act(
        dim_in=DA3_FEATURE_DIM_FLOW2ACT,
        output_dim=2,
        features=256,
        out_channels=(256, 512, 1024, 1024),
    )


def strict_load_safetensors_flow2act(module: nn.Module, path: Path, *, assign_cpu: bool = False) -> None:
    """Load exact key/shape/dtype tensors; deliberately perform no hash check."""

    if type(assign_cpu) is not bool:
        raise TypeError("assign_cpu must be boolean")
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DA3 weight file does not exist: {path}")
    state = module.state_dict()
    with safe_open(path, framework="pt", device="cpu") as handle:
        artifact_keys = set(handle.keys())
        model_keys = set(state)
        if artifact_keys != model_keys:
            missing = sorted(model_keys - artifact_keys)
            unexpected = sorted(artifact_keys - model_keys)
            raise RuntimeError(f"DA3 key mismatch for {path}: missing={missing[:8]}, unexpected={unexpected[:8]}")
        assigned = {}
        with torch.no_grad():
            for key, target in state.items():
                if assign_cpu and (target.device.type not in ("cpu", "meta") or target.dtype != torch.float32):
                    raise ValueError("Shared DA3 weights require CPU/meta FP32 targets")
                source = handle.get_tensor(key)
                if source.shape != target.shape:
                    raise RuntimeError(f"DA3 shape mismatch for {key}: {tuple(source.shape)} != {tuple(target.shape)}")
                if source.dtype != target.dtype:
                    raise RuntimeError(f"DA3 dtype mismatch for {key}: {source.dtype} != {target.dtype}")
                if not torch.isfinite(source).all():
                    raise RuntimeError(f"DA3 tensor contains non-finite values: {key}")
                if assign_cpu:
                    assigned[key] = source
                else:
                    target.copy_(source.to(device=target.device), non_blocking=False)
            if assign_cpu:
                # Keep safetensors CPU storage instead of copying into private
                # parameters. The module owns the storage after this handle closes.
                module.load_state_dict(assigned, strict=True, assign=True)


class DA3AnyViewTeacherFlow2Act:
    """Rank-local frozen teacher kept outside the student and its checkpoint."""

    def __init__(
        self,
        *,
        backbone_weights: str | Path,
        head_weights: str | Path | None,
        device: str | torch.device,
        layer_ids: Sequence[int] = DA3_LAYER_IDS_FLOW2ACT,
    ) -> None:
        self.device = torch.device(device)
        self.layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        if self.layer_ids not in (DA3_LAYER_IDS_FLOW2ACT, (DA3_LAYER_IDS_FLOW2ACT[-1],)):
            raise ValueError("DA3 teacher layers must be (19,27,33,39) or (39,)")
        with torch.device("meta"):
            self.backbone = build_da3_anyview_giant_flow2act(self.layer_ids)
            self.head = build_da3_dualdpt_flow2act() if head_weights is not None else None
        self.backbone.to_empty(device=self.device)
        strict_load_safetensors_flow2act(self.backbone, Path(backbone_weights))
        self.backbone.eval().requires_grad_(False)
        if self.head is not None:
            self.head.to_empty(device=self.device)
            strict_load_safetensors_flow2act(self.head, Path(head_weights))
            self.head.eval().requires_grad_(False)
        self.image_mean = torch.tensor(
            (0.485, 0.456, 0.406),
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, 3, 1, 1)
        self.image_std = torch.tensor(
            (0.229, 0.224, 0.225),
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, 3, 1, 1)

    @staticmethod
    def _grid_from_images(images: torch.Tensor) -> tuple[int, int]:
        height, width = images.shape[-2:]
        if height % 14 or width % 14:
            raise ValueError(f"DA3 image H/W must be divisible by 14, got {(height, width)}")
        grid = height // 14, width // 14
        if grid not in ((18, 18), (18, 24)):
            raise ValueError(f"Unsupported DA3 grid {grid}")
        return grid

    @staticmethod
    def _validate_features(
        features: Sequence[torch.Tensor],
        grid_hw: tuple[int, int],
        *,
        layer_ids: Sequence[int] = DA3_LAYER_IDS_FLOW2ACT,
    ) -> tuple[int, int]:
        if len(features) != len(layer_ids):
            raise ValueError(f"DA3 must return {len(layer_ids)} configured tap tensors")
        first = features[0]
        if first.ndim != 4:
            raise ValueError("DA3 features must have shape [B,V,P,3072]")
        batch_size, num_views, _, _ = first.shape
        expected = (batch_size, num_views, grid_hw[0] * grid_hw[1], 3072)
        for layer_id, feature in zip(layer_ids, features, strict=True):
            if feature.shape != expected:
                raise ValueError(f"DA3 post-block {layer_id} must have shape {expected}, got {feature.shape}")
        return batch_size, num_views

    def decode_student(
        self,
        features: Sequence[torch.Tensor],
        student_grid_hw: Sequence[int] | torch.Tensor,
        *,
        teacher_grid_hw: Sequence[int] | torch.Tensor | None = None,
        output_hw: Sequence[int] | None = None,
        use_checkpoint: bool = True,
        batch_chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.head is None:
            raise RuntimeError("DualDPTFlow2Act is intentionally absent from this backbone-only teacher")
        if len(features) != 4:
            raise ValueError("DualDPTFlow2Act student decoding requires all four aligned feature layers")
        source = _uniform_grid_pair_flow2act(
            student_grid_hw,
            name="student_grid_hw",
            device=features[0].device,
        )
        batch_size, num_views = self._validate_features(features, source)

        target = source if teacher_grid_hw is None else _uniform_grid_pair_flow2act(
            teacher_grid_hw,
            name="teacher_grid_hw",
            device=features[0].device,
        )
        height, width = output_hw or (target[0] * 14, target[1] * 14)
        if (height // 14, width // 14) != target or height % 14 or width % 14:
            raise ValueError("DualDPTFlow2Act output_hw is incompatible with the patch grid")
        # DualDPT skips internal chunking when chunk_size >= V, even for B > 1.
        # Split B here to bound head activations within each checkpoint.
        chunk_size = 1 if batch_chunk_size is None else batch_chunk_size
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("batch_chunk_size must be a positive integer")
        if features[0].device != self.device:
            raise ValueError(f"Student features are on {features[0].device}; decoder is on {self.device}")

        def decode_one(*values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # Keep casts and interpolation inside checkpoint: retain the original
            # adapter outputs rather than four expanded FP32 teacher-grid tensors.
            values = tuple(value.float() for value in values)
            current_batch = values[0].shape[0]
            if source != target:
                upsampled: list[torch.Tensor] = []
                for feature in values:
                    value = feature.permute(0, 1, 3, 2).reshape(
                        current_batch * num_views, DA3_FEATURE_DIM_FLOW2ACT, source[0], source[1],
                    )
                    value = torch.nn.functional.interpolate(
                        value, size=target, mode="bilinear", align_corners=False,
                    )
                    value = value.reshape(
                        current_batch, num_views, DA3_FEATURE_DIM_FLOW2ACT, target[0] * target[1],
                    ).permute(0, 1, 3, 2)
                    upsampled.append(value)
                values = tuple(upsampled)
            self._validate_features(values, target)
            output = self.head(
                [(value, None) for value in values],
                height,
                width,
                patch_start_idx=0,
                chunk_size=num_views,
                depth_only=True,
            )
            return output["depth"], output["depth_conf"]

        depth_parts: list[torch.Tensor] = []
        confidence_parts: list[torch.Tensor] = []
        with torch.autocast(device_type=self.device.type, enabled=False):
            for start in range(0, batch_size, chunk_size):
                values = tuple(feature[start : start + chunk_size] for feature in features)
                if use_checkpoint and torch.is_grad_enabled() and any(value.requires_grad for value in values):
                    depth, confidence = checkpoint(decode_one, *values, use_reentrant=False)
                else:
                    depth, confidence = decode_one(*values)
                depth_parts.append(depth)
                confidence_parts.append(confidence)
        depth = torch.cat(depth_parts)
        confidence = torch.cat(confidence_parts)
        expected = (batch_size, num_views, height, width)
        if depth.shape != expected or confidence.shape != expected:
            raise RuntimeError(
                f"DualDPTFlow2Act output must be {expected}; got depth={depth.shape}, confidence={confidence.shape}"
            )
        return {"depth": depth, "confidence": confidence, "depth_conf": confidence}

    def encode(
        self,
        images: torch.Tensor,
        view_mask: torch.Tensor | None = None,
        *,
        batch_chunk_size: int | None = None,
        decode_depth: bool | None = None,
    ) -> dict[str, Any]:
        """Jointly encode all synchronized views in each ``[B,V]`` group."""

        if images.ndim != 5 or images.dtype is not torch.uint8 or images.shape[2] != 3:
            raise ValueError("teacher images must be uint8 [B,V,3,H,W]")
        batch_size, num_views = images.shape[:2]
        if num_views not in (2, 3):
            raise ValueError("DA3 teacher supports exactly two or three valid views")
        if view_mask is not None:
            view_mask = torch.as_tensor(view_mask, dtype=torch.bool, device=self.device)
            if view_mask.shape != (batch_size, num_views):
                raise ValueError("Every structural DA3 view must be valid")
            torch._assert_async(torch.all(view_mask), "Every structural DA3 view must be valid")
        grid = self._grid_from_images(images)
        chunk_size = batch_size if batch_chunk_size is None else batch_chunk_size
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("batch_chunk_size must be a positive integer")

        if decode_depth is None:
            decode_depth = self.head is not None
        if decode_depth and self.head is None:
            raise RuntimeError("decode_depth=True requires a teacher constructed with include_head=True")
        feature_parts: list[list[torch.Tensor]] = [[] for _ in self.layer_ids]
        depth_parts: list[torch.Tensor] = []
        confidence_parts: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, batch_size, chunk_size):
                value = images[start : start + chunk_size].to(self.device, non_blocking=True).float().div_(255.0)
                value = value.sub(self.image_mean).div(self.image_std)
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if self.device.type == "cuda"
                    else nullcontext()
                )
                with autocast:
                    pyramid, _ = self.backbone(
                        value,
                        cam_token=None,
                        export_feat_layers=[],
                        ref_view_strategy="first",
                    )
                features = tuple(entry[0] for entry in pyramid)
                self._validate_features(features, grid, layer_ids=self.layer_ids)
                for parts, feature in zip(feature_parts, features, strict=True):
                    parts.append(feature.detach())
                if decode_depth:
                    decoded = self.decode_student(
                        features,
                        grid,
                        teacher_grid_hw=grid,
                        output_hw=images.shape[-2:],
                        use_checkpoint=False,
                        # Backbone still jointly encodes B samples and all views;
                        # the sample-independent depth head decodes one B at a time.
                        batch_chunk_size=1,
                    )
                    depth_parts.append(decoded["depth"].detach())
                    confidence_parts.append(decoded["confidence"].detach())
        features = tuple(torch.cat(parts) for parts in feature_parts)
        patch_grid = torch.tensor(grid, dtype=torch.long, device=self.device).view(1, 1, 2)
        patch_grid = patch_grid.expand(batch_size, num_views, 2)
        output: dict[str, Any] = {
            "features": features,
            "patch_grid": patch_grid,
            "layer_ids": self.layer_ids,
        }
        if decode_depth:
            depth = torch.cat(depth_parts)
            confidence = torch.cat(confidence_parts)
            output.update({"depth": depth, "confidence": confidence, "depth_conf": confidence})
        return output

    encode_teacher = encode


def build_teacher_flow2act(
    weights_dir: str | Path,
    device: str | torch.device,
    *,
    include_head: bool = True,
    layer_ids: Sequence[int] | None = None,
) -> DA3AnyViewTeacherFlow2Act:
    root = Path(weights_dir).expanduser().resolve()
    return DA3AnyViewTeacherFlow2Act(
        backbone_weights=root / DA3_BACKBONE_FILENAME_FLOW2ACT,
        head_weights=(root / DA3_HEAD_FILENAME_FLOW2ACT) if include_head else None,
        device=device,
        # Keep the old default for existing callers, but allow four-layer
        # feature supervision without ever constructing a training depth head.
        layer_ids=(
            tuple(layer_ids) if layer_ids is not None
            else (DA3_LAYER_IDS_FLOW2ACT if include_head else (DA3_LAYER_IDS_FLOW2ACT[-1],))
        ),
    )


class FrozenDepthDiagnosticsFlow2Act:
    """CPU-resident, rank-zero-only head; detached diagnostics, never a loss."""

    def __init__(self, weights_dir: str | Path) -> None:
        # Construction must not advance the training RNG on rank zero only.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            self.head = build_da3_dualdpt_flow2act()
        strict_load_safetensors_flow2act(
            self.head, Path(weights_dir) / DA3_HEAD_FILENAME_FLOW2ACT
        )
        self.head.eval().requires_grad_(False)

    @torch.no_grad()
    def decode_pair_flow2act(
        self,
        teacher_features: Sequence[torch.Tensor],
        student_features: Sequence[torch.Tensor],
        *,
        grid_hw: tuple[int, int],
        output_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(size * 14 for size in grid_hw) != tuple(output_hw):
            raise ValueError("Depth diagnostic output must use the DA3 native grid")
        batch, views = DA3AnyViewTeacherFlow2Act._validate_features(teacher_features, grid_hw)
        if batch != 1 or DA3AnyViewTeacherFlow2Act._validate_features(student_features, grid_hw) != (batch, views):
            raise ValueError("Depth diagnostics require one paired sample and all its views")
        device = teacher_features[0].device
        if any(value.requires_grad for value in (*teacher_features, *student_features)):
            raise ValueError("Depth diagnostics must receive detached feature snapshots")
        if any(value.device != device for value in (*teacher_features, *student_features)):
            raise ValueError("Paired diagnostic features must be on the same device")
        try:
            self.head.to(device=device, dtype=torch.float32).eval()
            results = []
            with torch.autocast(device_type=device.type, enabled=False):
                for features in (teacher_features, student_features):
                    output = self.head(
                        [(value.float(), None) for value in features],
                        *output_hw,
                        patch_start_idx=0,
                        chunk_size=views,
                        depth_only=True,
                    )
                    depth = output["depth"]
                    if depth.shape != (1, views, *output_hw) or not torch.isfinite(depth).all():
                        raise ValueError("Invalid diagnostic depth shape or non-finite output")
                    results.append(depth.detach().float().cpu())
                    del output, depth
            return results[0], results[1]
        finally:
            self.head.to(device="cpu")
