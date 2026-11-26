"""Independent depth-only adapter with a frozen policy and frozen DA3 decoder.

This module intentionally has no feature distillation target or feature loss.
Depth is expressed in the teacher's pseudo-depth scale, not physical metres.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .probe_common import (
    episode_frame_indices_flow2act,
    stratified_episode_split_flow2act,
)

@dataclass(frozen=True)
class Flow2ActDepthProbeConfig:
    input_dim: int = 2560
    width: int = 512
    teacher_dim: int = 3072
    output_grid: int = 18

    def __post_init__(self):
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in (self.input_dim, self.width, self.teacher_dim, self.output_grid)):
            raise ValueError("Depth-only adapter dimensions must be positive integers")

    def to_dict(self):
        return asdict(self)


class Flow2ActDepthProbe(nn.Module):
    """Shared channel/spatial trunk, four independent full-width DA3 heads."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or Flow2ActDepthProbeConfig()
        cfg = self.config
        self.input_norm = nn.LayerNorm(cfg.input_dim)
        self.channel = nn.Linear(cfg.input_dim, cfg.width)
        self.refine1 = nn.Conv2d(cfg.width, cfg.width, 3, padding=1)
        self.refine2 = nn.Conv2d(cfg.width, cfg.width, 3, padding=1)
        self.heads = nn.ModuleList(nn.Linear(cfg.width, cfg.teacher_dim) for _ in range(4))

    def forward(self, tap_grids):
        if len(tap_grids) != 4:
            raise ValueError("The independent probe requires four VLM visual taps")
        outputs = []
        for grid, head in zip(tap_grids, self.heads, strict=True):
            if grid.ndim != 5 or grid.shape[-1] != self.config.input_dim:
                raise ValueError("Probe taps must be [B,V,H,W,input_dim]")
            b, v, h, w, _ = grid.shape
            # Freeze the upstream representation even when called outside the trainer.
            x = F.gelu(self.channel(self.input_norm(grid.detach().float())))
            x = x.permute(0, 1, 4, 2, 3).reshape(b*v, self.config.width, h, w)
            x = F.interpolate(x, size=(self.config.output_grid,)*2, mode="bilinear", align_corners=False)
            x = F.gelu(self.refine1(x))
            x = F.gelu(self.refine2(x))
            x = x.reshape(b, v, self.config.width, -1).permute(0, 1, 3, 2)
            outputs.append(head(x))
        return tuple(outputs)


def _depth_inputs_flow2act(depth, teacher_depth, confidence):
    """Canonical FP32 depths and strictly teacher-only confidence weights."""
    from mibot.models.VLA.losses_flow2act import (
        _canonical_depth_shape_flow2act,
        bounded_confidence_weight_flow2act,
    )
    depth = _canonical_depth_shape_flow2act("depth", depth).float()
    teacher_depth = _canonical_depth_shape_flow2act("teacher_depth", teacher_depth).detach().float()
    confidence = _canonical_depth_shape_flow2act("confidence", confidence).detach().float()
    if depth.shape != teacher_depth.shape or confidence.shape != teacher_depth.shape:
        raise ValueError("Depth-only prediction, teacher depth and confidence shapes must match")
    if depth.shape[0] == 0 or depth.shape[1] != 2:
        raise ValueError("Depth-only supervision requires a nonempty batch with main/wrist views")
    finite = torch.isfinite(depth)
    if not finite.all() or not (depth > 0).all():
        finite_values = depth.detach()[finite]
        minimum = float(finite_values.min()) if finite_values.numel() else None
        maximum = float(finite_values.max()) if finite_values.numel() else None
        raise FloatingPointError(
            "Student depth must be finite and strictly positive; "
            f"finite_min={minimum}, finite_max={maximum}, "
            f"nonfinite={int((~finite).sum())}, nonpositive={int((finite & (depth <= 0)).sum())}")
    valid = (torch.isfinite(teacher_depth) & (teacher_depth > 1e-6)
             & torch.isfinite(confidence) & (confidence > 1.0))
    weight = bounded_confidence_weight_flow2act(confidence) * valid.to(depth.dtype)
    safe_teacher = torch.where(valid, teacher_depth, torch.ones_like(teacher_depth))
    return depth, safe_teacher, confidence, valid, weight


def depth_only_losses_flow2act(depth, teacher_depth, confidence, *,
                                    si_log_weight=1.0, gradient_weight=0.5, log_depth_weight=0.1):
    """SI-log + multiscale log gradients + weak unaligned log-depth L1.

    All terms use the same frozen teacher evidence. Cameras are averaged
    equally within each sample; empty samples contribute a graph-connected
    zero to a fixed batch denominator, preserving DDP/GA equivalence.
    """
    from mibot.models.VLA.losses_flow2act import (
        _equal_available_gradient_mean_flow2act,
        _equal_valid_view_mean_flow2act,
        _weighted_sample_mean_flow2act,
    )
    weights = (si_log_weight, gradient_weight, log_depth_weight)
    if any(not math.isfinite(value) or value < 0 for value in weights) or sum(weights) <= 0:
        raise ValueError("Depth-only loss weights must be finite and nonnegative, with one positive")
    prediction, safe_teacher, _, _, pixel_weight = _depth_inputs_flow2act(
        depth, teacher_depth, confidence)
    view_mask = torch.ones(prediction.shape[:2], dtype=torch.bool, device=prediction.device)
    strides = (1, 2, 4)
    if any(stride >= prediction.shape[-2] or stride >= prediction.shape[-1] for stride in strides):
        raise ValueError("Depth-only gradient strides 1/2/4 require H and W greater than four")
    # A frozen decoder can produce small positive depths. They remain valid
    # predictions: the teacher-evidence threshold must not reject the student.
    log_residual = prediction.clamp_min(torch.finfo(torch.float32).tiny).log() - safe_teacher.log()
    scale_offset = _weighted_sample_mean_flow2act(
        log_residual, pixel_weight, context="Depth-only shared log-scale offset",
        skip_empty_teacher_evidence=True)
    centered = log_residual - scale_offset[:, None, None, None]
    si_log = _equal_valid_view_mean_flow2act(
        (centered.square()*pixel_weight).sum(dim=(2, 3)), pixel_weight.sum(dim=(2, 3)),
        view_mask, context="Depth-only SI-log", skip_empty_teacher_evidence=True)
    gradient_scales = []
    for stride in strides:
        dx = centered[..., :, stride:] - centered[..., :, :-stride]
        dy = centered[..., stride:, :] - centered[..., :-stride, :]
        weight_dx = torch.minimum(pixel_weight[..., :, stride:], pixel_weight[..., :, :-stride])
        weight_dy = torch.minimum(pixel_weight[..., stride:, :], pixel_weight[..., :-stride, :])
        numerator = (dx.abs()*weight_dx).sum(dim=(2, 3)) + (dy.abs()*weight_dy).sum(dim=(2, 3))
        denominator = weight_dx.sum(dim=(2, 3)) + weight_dy.sum(dim=(2, 3))
        gradient_scales.append(_equal_available_gradient_mean_flow2act(
            numerator, denominator, view_mask))
    multiscale_gradient = torch.stack(gradient_scales).mean()
    log_depth = _equal_valid_view_mean_flow2act(
        (log_residual.abs()*pixel_weight).sum(dim=(2, 3)), pixel_weight.sum(dim=(2, 3)),
        view_mask, context="Depth-only unaligned log-depth L1", skip_empty_teacher_evidence=True)
    loss = (si_log_weight*si_log + gradient_weight*multiscale_gradient
            + log_depth_weight*log_depth)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite depth-only adapter loss")
    return {"loss": loss, "si_log": si_log,
            "multiscale_gradient": multiscale_gradient, "log_depth": log_depth}


@torch.no_grad()
def encode_probe_teacher_depthonly_flow2act(teacher, raw_images):
    """Same float resize and joint-view backbone as policy training; no requantization."""
    if raw_images.dtype != torch.uint8 or raw_images.ndim != 5 or raw_images.shape[1:3] != (2, 3):
        raise ValueError("Probe teacher requires raw uint8 [B,2,3,H,W]")
    rgb = raw_images.to(teacher.device).float().div(255.0)
    b = rgb.shape[0]
    rgb = F.interpolate(rgb.flatten(0, 1), (252, 252), mode="bilinear", align_corners=False,
                        antialias=True).reshape(b, 2, 3, 252, 252)
    tolerance = 8*torch.finfo(torch.float32).eps
    if not torch.isfinite(rgb).all() or (rgb < -tolerance).any() or (rgb > 1+tolerance).any():
        raise ValueError("Teacher RGB resize exceeded FP32 rounding tolerance")
    rgb = rgb.clamp(0, 1)
    depths, confidences = [], []
    for sample in rgb.split(1):
        value = (sample-teacher.image_mean)/teacher.image_std
        amp = torch.autocast("cuda", dtype=torch.float16) if teacher.device.type == "cuda" else nullcontext()
        with amp:
            pyramid, _ = teacher.backbone(value, cam_token=None, export_feat_layers=[], ref_view_strategy="first")
        if len(pyramid) != 4:
            raise ValueError("Teacher depth decoding requires four DA3 feature taps")
        features = tuple(entry[0].detach() for entry in pyramid)
        decoded = teacher.decode_student(features, (18, 18), output_hw=(252, 252),
                                         use_checkpoint=False, batch_chunk_size=1)
        depths.append(decoded["depth"].detach())
        confidences.append(decoded["confidence"].detach())
        del pyramid, features, decoded, value
    # Do not return or retain a dense teacher feature target in this route.
    return {"depth": torch.cat(depths), "confidence": torch.cat(confidences)}


@torch.no_grad()
def depth_metric_sums_flow2act(depth, teacher_depth, confidence):
    """Additive, pixel-weighted depth metrics for exact distributed aggregation.

    Confidence defines the evaluation mask; metric values themselves use
    uniform valid-pixel weighting. Only scale estimation uses bounded teacher
    confidence, with a single shared scale for both cameras of each sample.
    """
    prediction, truth, _, valid, weight = _depth_inputs_flow2act(depth, teacher_depth, confidence)
    prediction, truth, weight = (value.double() for value in (prediction, truth, weight))
    residual = prediction.log()-truth.log()
    denominator = weight.sum(dim=(1, 2, 3))
    offset = (residual*weight).sum(dim=(1, 2, 3))/denominator.clamp_min(1e-30)
    aligned = prediction*torch.exp(-offset[:, None, None, None])
    result = {}
    for prefix, selection in (("", slice(None)), ("main_", slice(0, 1)), ("wrist_", slice(1, 2))):
        selected_valid = valid[:, selection]
        selected_truth = truth[:, selection]
        result[prefix+"valid_pixels"] = selected_valid.double().sum()
        result[prefix+"total_pixels"] = torch.tensor(selected_valid.numel(), device=depth.device,
                                                     dtype=torch.float64)
        for name, value in (("raw", prediction), ("aligned", aligned)):
            selected = value[:, selection]
            difference = selected-selected_truth
            ratio = torch.maximum(selected/selected_truth, selected_truth/selected)
            result[prefix+name+"_absrel_sum"] = torch.where(
                selected_valid, difference.abs()/selected_truth, 0).sum()
            result[prefix+name+"_squared_error_sum"] = torch.where(
                selected_valid, difference.square(), 0).sum()
            result[prefix+name+"_delta1_sum"] = ((ratio < 1.25) & selected_valid).double().sum()
    return result


def finalize_depth_metrics_flow2act(sums):
    """Convert all-reduced sums to JSON values; unsupported metrics are null."""
    def summarize(prefix):
        count = float(sums[prefix+"valid_pixels"])
        total = float(sums[prefix+"total_pixels"])
        metrics = {"valid_depth_pixels": int(count), "total_depth_pixels": int(total),
                   "valid_depth_fraction": count/total if total else 0.0}
        for name in ("raw", "aligned"):
            metrics["depth_absrel_"+name] = float(sums[prefix+name+"_absrel_sum"])/count if count else None
            metrics["depth_rmse_"+name] = math.sqrt(float(sums[prefix+name+"_squared_error_sum"])/count) if count else None
            metrics["depth_delta1_"+name] = float(sums[prefix+name+"_delta1_sum"])/count if count else None
        return metrics
    result = summarize("")
    result["per_view"] = {name: summarize(name+"_") for name in ("main", "wrist")}
    return result
