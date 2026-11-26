# ruff: noqa: N999, N802, N812, FBT001, FBT002, FBT003, SLF001
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP32 feature and relative-depth losses for Flow2Act distillation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

FEATURE_DIM_FLOW2ACT = 3072
DEPTH_GRADIENT_STRIDES_FLOW2ACT = (1, 2, 4)


def _require_finite_flow2act(name: str, tensor: torch.Tensor) -> None:
    torch._assert_async(torch.isfinite(tensor).all(), f"{name} contains non-finite values")


def _validate_view_mask_flow2act(
    valid_view_mask: torch.Tensor,
    batch_size: int,
    num_views: int,
) -> torch.Tensor:
    if valid_view_mask.shape != (batch_size, num_views):
        raise ValueError(
            f"valid_view_mask must have shape [{batch_size},{num_views}], got {tuple(valid_view_mask.shape)}"
        )
    if valid_view_mask.dtype != torch.bool:
        raise TypeError(f"valid_view_mask must be bool, got {valid_view_mask.dtype}")
    valid_counts = valid_view_mask.sum(dim=1)
    torch._assert_async(torch.all(valid_counts > 0), "valid_view_mask has a sample with no valid view")
    return valid_view_mask


def _uniform_grid_flow2act(
    name: str,
    grid_hw: Sequence[int] | torch.Tensor,
    *,
    batch_size: int,
    num_views: int,
    token_count: int,
) -> tuple[int, int]:
    grid = torch.as_tensor(grid_hw, dtype=torch.long)
    if grid.shape == (2,):
        grid = grid.view(1, 1, 2).expand(batch_size, num_views, 2)
    elif grid.shape == (num_views, 2):
        grid = grid.unsqueeze(0).expand(batch_size, num_views, 2)
    if grid.shape != (batch_size, num_views, 2):
        raise ValueError(f"{name} must be [2], [V,2], or [B,V,2]")
    flat = grid.reshape(-1, 2)
    if torch.any(flat <= 0) or not torch.all(flat == flat[0]):
        raise ValueError(f"{name} must contain one positive uniform grid")
    height, width = (int(value.item()) for value in flat[0])
    if height * width != token_count:
        raise ValueError(f"{name} {(height, width)} does not match {token_count} tokens")
    return height, width


def downsample_teacher_features_flow2act(
    teacher_features: torch.Tensor,
    *,
    student_grid_hw: Sequence[int] | torch.Tensor,
    teacher_grid_hw: Sequence[int] | torch.Tensor,
    student_token_count: int,
) -> torch.Tensor:
    """Downsample detached DA3 tokens to the native student patch grid."""

    if teacher_features.ndim != 4:
        raise ValueError("teacher_features must be [B,V,P,D]")
    batch_size, num_views, teacher_token_count, channels = teacher_features.shape
    source_height, source_width = _uniform_grid_flow2act(
        "teacher_grid_hw",
        teacher_grid_hw,
        batch_size=batch_size,
        num_views=num_views,
        token_count=teacher_token_count,
    )
    target_height, target_width = _uniform_grid_flow2act(
        "student_grid_hw",
        student_grid_hw,
        batch_size=batch_size,
        num_views=num_views,
        token_count=student_token_count,
    )
    value = teacher_features.detach().float()
    value = value.permute(0, 1, 3, 2).reshape(
        batch_size * num_views,
        channels,
        source_height,
        source_width,
    )
    if (source_height, source_width) != (target_height, target_width):
        value = F.interpolate(value, size=(target_height, target_width), mode="bilinear", align_corners=False)
    return value.reshape(batch_size, num_views, channels, student_token_count).permute(0, 1, 3, 2)


def _prepare_feature_pair_flow2act(
    predicted: torch.Tensor,
    teacher: torch.Tensor,
    *,
    layer_index: int,
    feature_dim: int,
    student_grid_hw: Sequence[int] | torch.Tensor | None,
    teacher_grid_hw: Sequence[int] | torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if predicted.ndim != 4 or teacher.ndim != 4:
        raise ValueError(f"Feature layer {layer_index} must contain [B,V,P,D] tensors")
    if predicted.shape[:2] != teacher.shape[:2] or predicted.shape[-1] != teacher.shape[-1]:
        raise ValueError(
            f"Feature batch/view/channel mismatch at layer {layer_index}: "
            f"predicted={tuple(predicted.shape)}, teacher={tuple(teacher.shape)}"
        )
    if predicted.shape[-1] != feature_dim:
        raise ValueError(f"Feature layer {layer_index} must retain all {feature_dim} channels")
    if predicted.shape[2] == 0 or teacher.shape[2] == 0:
        raise ValueError(f"Feature layer {layer_index} has an empty token axis")
    if predicted.shape[2] == teacher.shape[2] and student_grid_hw is None and teacher_grid_hw is None:
        teacher_native = teacher.detach().float()
    else:
        if student_grid_hw is None or teacher_grid_hw is None:
            raise ValueError("student_grid_hw and teacher_grid_hw are required when token grids differ")
        teacher_native = downsample_teacher_features_flow2act(
            teacher,
            student_grid_hw=student_grid_hw,
            teacher_grid_hw=teacher_grid_hw,
            student_token_count=predicted.shape[2],
        )
    predicted_fp32 = predicted.float()
    _require_finite_flow2act(f"predicted_features[{layer_index}]", predicted_fp32)
    _require_finite_flow2act(f"teacher_features[{layer_index}]", teacher_native)
    return predicted_fp32, teacher_native


def _hierarchical_feature_reduce_flow2act(
    token_loss: torch.Tensor,
    valid_view_mask: torch.Tensor,
) -> torch.Tensor:
    """Reduce token -> valid view -> sample, preserving equal sample weights."""

    batch_size, num_views, _ = token_loss.shape
    _validate_view_mask_flow2act(valid_view_mask, batch_size, num_views)
    view_loss = token_loss.mean(dim=-1)
    view_weights = valid_view_mask.to(device=view_loss.device, dtype=view_loss.dtype)
    sample_loss = (view_loss * view_weights).sum(dim=1) / view_weights.sum(dim=1)
    return sample_loss.mean()


def feature_distillation_loss_flow2act(
    predicted_features: Sequence[torch.Tensor],
    teacher_features: Sequence[torch.Tensor],
    valid_view_mask: torch.Tensor,
    *,
    student_grid_hw: Sequence[int] | torch.Tensor | None = None,
    teacher_grid_hw: Sequence[int] | torch.Tensor | None = None,
    feature_dim: int = FEATURE_DIM_FLOW2ACT,
    smooth_l1_beta: float = 1.0,
    layer_norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Four-layer 0.5 normalized-cosine + 0.5 normalized-SmoothL1 loss."""

    if len(predicted_features) != 4 or len(teacher_features) != 4:
        raise ValueError("four_layer_da3 feature loss requires exactly four student and teacher layers")
    if smooth_l1_beta <= 0:
        raise ValueError(f"smooth_l1_beta must be positive, got {smooth_l1_beta}")
    layer_losses: list[torch.Tensor] = []
    expected_batch_views: tuple[int, int] | None = None
    for layer_index, (predicted, teacher) in enumerate(zip(predicted_features, teacher_features, strict=True)):
        predicted_fp32, teacher_native = _prepare_feature_pair_flow2act(
            predicted,
            teacher,
            layer_index=layer_index,
            feature_dim=feature_dim,
            student_grid_hw=student_grid_hw,
            teacher_grid_hw=teacher_grid_hw,
        )
        batch_views = predicted_fp32.shape[:2]
        if expected_batch_views is None:
            expected_batch_views = batch_views
        elif batch_views != expected_batch_views:
            raise ValueError("All feature layers must share batch and view axes")
        predicted_direction = F.normalize(predicted_fp32, dim=-1, eps=1e-6)
        teacher_direction = F.normalize(teacher_native, dim=-1, eps=1e-6)
        cosine = 0.5 * (predicted_direction - teacher_direction).square().sum(dim=-1)
        channels = predicted_fp32.shape[-1]
        predicted_normalized = F.layer_norm(
            predicted_fp32,
            (channels,),
            weight=None,
            bias=None,
            eps=layer_norm_eps,
        )
        teacher_normalized = F.layer_norm(
            teacher_native,
            (channels,),
            weight=None,
            bias=None,
            eps=layer_norm_eps,
        )
        smooth_l1 = F.smooth_l1_loss(
            predicted_normalized,
            teacher_normalized,
            reduction="none",
            beta=smooth_l1_beta,
        ).mean(dim=-1)
        layer_losses.append(
            _hierarchical_feature_reduce_flow2act(
                0.5 * cosine + 0.5 * smooth_l1,
                valid_view_mask,
            )
        )
    loss = torch.stack(layer_losses).mean()
    _require_finite_flow2act("Flow2Act four-layer feature loss", loss)
    return loss


def single_layer_cosine_distillation_loss_flow2act(
    predicted_features: Sequence[torch.Tensor],
    teacher_features: Sequence[torch.Tensor],
    valid_view_mask: torch.Tensor,
    *,
    student_grid_hw: Sequence[int] | torch.Tensor | None = None,
    teacher_grid_hw: Sequence[int] | torch.Tensor | None = None,
    feature_dim: int = FEATURE_DIM_FLOW2ACT,
) -> torch.Tensor:
    """Pure normalized-cosine loss for post-13 -> DA3 post-39 only."""

    if len(predicted_features) != 1 or len(teacher_features) != 1:
        raise ValueError("single_layer_da3_sf requires exactly one student and teacher feature")
    predicted_fp32, teacher_native = _prepare_feature_pair_flow2act(
        predicted_features[0],
        teacher_features[0],
        layer_index=0,
        feature_dim=feature_dim,
        student_grid_hw=student_grid_hw,
        teacher_grid_hw=teacher_grid_hw,
    )
    predicted_direction = F.normalize(predicted_fp32, dim=-1, eps=1e-6)
    teacher_direction = F.normalize(teacher_native, dim=-1, eps=1e-6)
    cosine = 0.5 * (predicted_direction - teacher_direction).square().sum(dim=-1)
    loss = _hierarchical_feature_reduce_flow2act(cosine, valid_view_mask)
    _require_finite_flow2act("Flow2Act single-layer cosine loss", loss)
    return loss


def _canonical_depth_shape_flow2act(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 5 and tensor.shape[2] == 1:
        tensor = tensor.squeeze(2)
    if tensor.ndim != 4:
        raise ValueError(f"{name} must be [B,V,H,W] or [B,V,1,H,W], got {tuple(tensor.shape)}")
    if tensor.shape[-2] == 0 or tensor.shape[-1] == 0:
        raise ValueError(f"{name} has an empty spatial axis")
    return tensor


def bounded_confidence_weight_flow2act(confidence: torch.Tensor) -> torch.Tensor:
    """Map positive DA3 confidence to the bounded weight ``(c - 1) / c``."""

    confidence_fp32 = confidence.detach().float()
    valid = torch.isfinite(confidence_fp32) & (confidence_fp32 > 1.0)
    safe_confidence = torch.where(valid, confidence_fp32, torch.ones_like(confidence_fp32))
    weight = ((safe_confidence - 1.0) / safe_confidence).clamp_(min=0.0, max=1.0)
    return torch.where(valid, weight, torch.zeros_like(weight))


@dataclass(frozen=True)
class DepthLossOutputFlow2Act:
    loss: torch.Tensor
    si_log: torch.Tensor
    multiscale_gradient: torch.Tensor


def _weighted_sample_mean_flow2act(
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    context: str,
    skip_empty_teacher_evidence: bool,
) -> torch.Tensor:
    reduce_dims = tuple(range(1, values.ndim))
    denominator = weights.sum(dim=reduce_dims)
    empty_sample = denominator <= 0
    if not skip_empty_teacher_evidence and torch.any(empty_sample).item():
        invalid = torch.nonzero(denominator <= 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"{context} has no positive-weight pixels for samples {invalid}")
    numerator = (values * weights).sum(dim=reduce_dims)
    safe_denominator = torch.where(empty_sample, torch.ones_like(denominator), denominator)
    return numerator / safe_denominator


def _equal_valid_view_mean_flow2act(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    valid_view_mask: torch.Tensor,
    *,
    context: str,
    skip_empty_teacher_evidence: bool,
) -> torch.Tensor:
    """Reduce pixels within views, then views equally within each sample."""

    if numerator.shape != denominator.shape or numerator.shape != valid_view_mask.shape:
        raise ValueError(
            f"{context} internal [B,V] shape mismatch: numerator={tuple(numerator.shape)}, "
            f"denominator={tuple(denominator.shape)}, mask={tuple(valid_view_mask.shape)}"
        )
    empty_valid_view = valid_view_mask & (denominator <= 0)
    if not skip_empty_teacher_evidence and torch.any(empty_valid_view).item():
        invalid = torch.nonzero(empty_valid_view, as_tuple=False).tolist()
        raise ValueError(f"{context} has no positive-weight pixels for [sample,view] {invalid}")
    eligible_view_mask = valid_view_mask & (denominator > 0)
    safe_denominator = torch.where(
        eligible_view_mask,
        denominator,
        torch.ones_like(denominator),
    )
    per_view = numerator / safe_denominator
    view_weights = eligible_view_mask.to(per_view.dtype)
    eligible_view_count = view_weights.sum(dim=1)
    per_sample = (per_view * view_weights).sum(dim=1) / eligible_view_count.clamp_min(1.0)
    eligible_sample_mask = eligible_view_count > 0
    # Samples without Teacher evidence contribute a graph-connected zero.
    # Keeping the batch denominator fixed prevents the same global batch from
    # producing different DDP/GA gradients under a different partition.
    return torch.where(eligible_sample_mask, per_sample, torch.zeros_like(per_sample)).mean()


def _equal_available_gradient_mean_flow2act(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    valid_view_mask: torch.Tensor,
) -> torch.Tensor:
    """Average a gradient scale over views and samples that contain valid pairs.

    A camera can contain positive-confidence depth pixels without containing a
    positive-confidence pair at every spatial stride.  Such a view has no
    gradient evidence at that stride and must be excluded from that stride's
    view mean instead of terminating the complete distributed training job.
    Samples with no eligible view at this stride are likewise excluded.  If
    the whole batch has no eligible pair, return a graph-connected zero and
    let the SI-log term continue supervising the batch.
    """

    if numerator.shape != denominator.shape or numerator.shape != valid_view_mask.shape:
        raise ValueError(
            "Flow2Act depth gradient internal [B,V] shape mismatch: "
            f"numerator={tuple(numerator.shape)}, denominator={tuple(denominator.shape)}, "
            f"mask={tuple(valid_view_mask.shape)}"
        )

    eligible_view_mask = valid_view_mask & (denominator > 0)
    safe_denominator = torch.where(
        eligible_view_mask,
        denominator,
        torch.ones_like(denominator),
    )
    per_view = numerator / safe_denominator
    view_weights = eligible_view_mask.to(per_view.dtype)
    eligible_view_count = view_weights.sum(dim=1)
    per_sample = (per_view * view_weights).sum(dim=1) / eligible_view_count.clamp_min(1.0)
    eligible_sample_mask = eligible_view_count > 0
    return torch.where(eligible_sample_mask, per_sample, torch.zeros_like(per_sample)).mean()


def relative_depth_loss_flow2act(
    predicted_depth: torch.Tensor,
    teacher_depth: torch.Tensor,
    teacher_confidence: torch.Tensor,
    valid_view_mask: torch.Tensor,
    *,
    gradient_strides: Sequence[int] = DEPTH_GRADIENT_STRIDES_FLOW2ACT,
    gradient_weight: float = 0.5,
    skip_empty_teacher_evidence: bool = False,
) -> DepthLossOutputFlow2Act:
    """Shared-per-sample scale-invariant relative-depth loss.

    One log-scale offset is estimated from every valid pixel in every valid
    camera of a sample.  Estimating an offset independently per camera would
    hide cross-view scale drift and is therefore intentionally not supported.

    ``skip_empty_teacher_evidence`` is an explicit opt-in for datasets where a
    structurally present camera can receive no positive-confidence dense-depth
    prediction from the frozen teacher.  Only that view's depth term is
    excluded; the caller's structural view mask, and therefore feature
    supervision, remain unchanged.  A sample with no depth evidence contributes
    zero to the fixed per-sample batch mean.  This fixed denominator is required
    for equivalent gradients across different DDP/GA partitions.
    """

    predicted_depth = _canonical_depth_shape_flow2act("predicted_depth", predicted_depth).float()
    teacher_depth = _canonical_depth_shape_flow2act("teacher_depth", teacher_depth).detach().float()
    teacher_confidence = _canonical_depth_shape_flow2act(
        "teacher_confidence",
        teacher_confidence,
    ).detach().float()
    if predicted_depth.shape != teacher_depth.shape:
        raise ValueError(
            f"Depth shape mismatch: predicted={tuple(predicted_depth.shape)}, teacher={tuple(teacher_depth.shape)}"
        )
    if teacher_confidence.shape != teacher_depth.shape:
        raise ValueError(
            "teacher_confidence shape must match teacher_depth, got "
            f"{tuple(teacher_confidence.shape)} and {tuple(teacher_depth.shape)}"
        )
    batch_size, num_views, height, width = teacher_depth.shape
    _validate_view_mask_flow2act(valid_view_mask, batch_size, num_views)
    _require_finite_flow2act("predicted_depth", predicted_depth)
    torch._assert_async(
        torch.all(predicted_depth > 1e-6),
        "predicted_depth must be finite and greater than 1e-6 before taking log",
    )
    if gradient_weight < 0:
        raise ValueError(f"gradient_weight must be non-negative, got {gradient_weight}")
    if not isinstance(skip_empty_teacher_evidence, bool):
        raise TypeError("skip_empty_teacher_evidence must be bool")

    strides = tuple(int(stride) for stride in gradient_strides)
    if not strides or any(stride <= 0 for stride in strides):
        raise ValueError(f"gradient_strides must contain positive integers, got {strides}")
    if any(stride >= height or stride >= width for stride in strides):
        raise ValueError(f"gradient stride must be smaller than H and W ({height},{width}), got {strides}")

    teacher_pixel_valid = (
        torch.isfinite(teacher_depth)
        & (teacher_depth > 1e-6)
        & torch.isfinite(teacher_confidence)
        & (teacher_confidence > 1.0)
    )
    confidence_weight = bounded_confidence_weight_flow2act(teacher_confidence)
    view_weight = valid_view_mask.to(device=confidence_weight.device, dtype=confidence_weight.dtype)[:, :, None, None]
    pixel_weight = confidence_weight * teacher_pixel_valid.to(confidence_weight.dtype) * view_weight

    safe_teacher_depth = torch.where(
        teacher_pixel_valid,
        teacher_depth,
        torch.ones_like(teacher_depth),
    )
    log_residual = predicted_depth.log() - safe_teacher_depth.log()
    scale_offset = _weighted_sample_mean_flow2act(
        log_residual,
        pixel_weight,
        context="Flow2Act shared log-scale offset",
        skip_empty_teacher_evidence=skip_empty_teacher_evidence,
    )
    centered_residual = log_residual - scale_offset[:, None, None, None]
    si_numerator = (centered_residual.square() * pixel_weight).sum(dim=(2, 3))
    si_denominator = pixel_weight.sum(dim=(2, 3))
    si_log = _equal_valid_view_mean_flow2act(
        si_numerator,
        si_denominator,
        valid_view_mask,
        context="Flow2Act SI-log loss",
        skip_empty_teacher_evidence=skip_empty_teacher_evidence,
    )

    scale_gradient_losses: list[torch.Tensor] = []
    for stride in strides:
        residual_dx = centered_residual[..., :, stride:] - centered_residual[..., :, :-stride]
        weight_dx = torch.minimum(pixel_weight[..., :, stride:], pixel_weight[..., :, :-stride])
        residual_dy = centered_residual[..., stride:, :] - centered_residual[..., :-stride, :]
        weight_dy = torch.minimum(pixel_weight[..., stride:, :], pixel_weight[..., :-stride, :])

        numerator = (residual_dx.abs() * weight_dx).sum(dim=(2, 3))
        numerator = numerator + (residual_dy.abs() * weight_dy).sum(dim=(2, 3))
        denominator = weight_dx.sum(dim=(2, 3)) + weight_dy.sum(dim=(2, 3))
        scale_gradient_losses.append(
            _equal_available_gradient_mean_flow2act(
                numerator,
                denominator,
                valid_view_mask,
            )
        )

    multiscale_gradient = torch.stack(scale_gradient_losses).mean()
    loss = si_log + gradient_weight * multiscale_gradient
    _require_finite_flow2act("Flow2Act SI-log loss", si_log)
    _require_finite_flow2act("Flow2Act multiscale gradient loss", multiscale_gradient)
    _require_finite_flow2act("Flow2Act relative-depth loss", loss)
    return DepthLossOutputFlow2Act(
        loss=loss,
        si_log=si_log,
        multiscale_gradient=multiscale_gradient,
    )

