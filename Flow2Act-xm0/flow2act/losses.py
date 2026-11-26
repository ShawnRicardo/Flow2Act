"""Fixed teacher pooling, pure cosine and masked trajectory supervision."""
import math
import torch
from torch.nn import functional as F


def downsample_teacher_features_flow2act(teacher, size):
    """Pool detached raw features before cosine normalization; never learn targets."""
    teacher = teacher.detach().float()
    if teacher.ndim == 4:
        b, v, patches, channels = teacher.shape
        edge = math.isqrt(patches)
        if edge * edge != patches:
            raise ValueError("Flattened teacher tokens must describe a square grid")
        teacher = teacher.reshape(b, v, edge, edge, channels)
    if teacher.ndim != 5:
        raise ValueError("Teacher features must be [B,V,H,W,D] or square [B,V,P,D]")
    b, v, h, w, channels = teacher.shape
    value = teacher.permute(0, 1, 4, 2, 3).reshape(b * v, channels, h, w)
    if (h, w) != tuple(size):
        if size[0] > h or size[1] > w:
            raise ValueError("Static targets are downsampled to the native student grid")
        value = F.interpolate(value, tuple(size), mode="bilinear", align_corners=False, antialias=True)
    return value.reshape(b, v, channels, *size).permute(0, 1, 3, 4, 2)


def feature_loss_flow2act(predicted, teacher, view_mask=None):
    """Equal tap/camera/sample cosine on native VLM grids, with full channels."""
    if len(predicted) != 4 or len(teacher) != 4:
        raise ValueError("Four spatial feature pairs are required")
    result = []
    for p, raw_teacher in zip(predicted, teacher, strict=True):
        if p.ndim != 5:
            raise ValueError("Student features must be [B,V,H,W,D]")
        p = p.float()
        t = downsample_teacher_features_flow2act(raw_teacher, p.shape[2:4])
        if p.shape != t.shape:
            raise ValueError(f"Pooled teacher and student disagree: {p.shape}, {t.shape}")
        if not torch.isfinite(p).all() or not torch.isfinite(t).all():
            raise FloatingPointError("Non-finite static features")
        cosine = 1.0 - (F.normalize(p, dim=-1, eps=1e-6) * F.normalize(t, dim=-1, eps=1e-6)).sum(-1)
        per_view = cosine.mean((-1, -2))
        valid = torch.ones_like(per_view, dtype=torch.bool) if view_mask is None else view_mask.bool()
        if valid.shape != per_view.shape or not valid.any(-1).all():
            raise ValueError("Every sample requires a matching mask and at least one current view")
        result.append(((per_view * valid).sum(-1) / valid.sum(-1)).mean())
    return torch.stack(result).mean()


def _average_horizons(values, valid):
    count = valid.sum(-1)
    per_sample = (values * valid).sum(-1) / count.clamp_min(1)
    samples = count > 0
    return (per_sample * samples).sum() / samples.sum().clamp_min(1)


def trajectory_loss_flow2act(prediction, target, valid, *, beta, moving_target=None,
                                   image_size=256, moving_threshold_px=1.0):
    """Equal samples/horizons; 2D additionally balances moving/static groups."""
    prediction, target = prediction.float(), target.detach().float()
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("Trajectory tensors must match [B,H,C,G,G]")
    valid = valid.bool()
    if valid.shape != prediction.shape[:2] + prediction.shape[-2:]:
        raise ValueError("Trajectory mask must have shape [B,H,G,G]")
    valid = valid & torch.isfinite(target).all(2)
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Non-finite Student trajectory")
    # Mask before arithmetic: NaN * 0 is not zero.
    target = torch.where(valid.unsqueeze(2), target, torch.zeros_like(target))
    error = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta).mean(2)
    if moving_target is None:
        count = valid.sum((-1, -2))
        per_horizon = (error * valid).sum((-1, -2)) / count.clamp_min(1)
        return _average_horizons(per_horizon, count > 0)
    displacement = torch.nan_to_num(moving_target.detach().float()) * image_size
    moving = torch.linalg.vector_norm(displacement, dim=2) >= moving_threshold_px
    group_losses, group_valid = [], []
    for group in (moving, ~moving):
        mask = valid & group
        count = mask.sum((-1, -2))
        group_losses.append((error * mask).sum((-1, -2)) / count.clamp_min(1))
        group_valid.append(count > 0)
    losses, present = torch.stack(group_losses, -1), torch.stack(group_valid, -1)
    per_horizon = (losses * present).sum(-1) / present.sum(-1).clamp_min(1)
    return _average_horizons(per_horizon, present.any(-1))


@torch.no_grad()
def motion_metrics_flow2act(predictions, targets, *, image_size=256, moving_threshold_px=1.0):
    """Raw teacher-reference diagnostics; missing populations have zero/count=0.

    Metrics never alter losses, targets, predictions or attention masks. 3D uses
    current-camera displacement / anchor median depth, not metric world units.
    """
    p = predictions["motion_2d"].detach().float()
    t = targets["motion_2d"].detach().float()
    valid = targets["valid_2d"].detach().bool() & torch.isfinite(t).all(2)
    if p.shape != t.shape or p.ndim != 5 or p.shape[2] != 2 or valid.shape != p.shape[:2] + p.shape[-2:]:
        raise ValueError("Motion metrics require matching 2D fields and validity")
    if not torch.isfinite(p).all():
        raise FloatingPointError("Non-finite Student motion metrics")
    t = torch.where(valid[:, :, None], t, torch.zeros_like(t))
    p_px, t_px = p * image_size, t * image_size
    epe = torch.linalg.vector_norm(p_px - t_px, dim=2)
    predicted_norm = torch.linalg.vector_norm(p_px, dim=2)
    target_norm = torch.linalg.vector_norm(t_px, dim=2)
    moving = valid & (target_norm >= moving_threshold_px)
    static = valid & ~moving
    result = {}

    def emit(name, values, mask):
        count = mask.sum()
        result[f"metric_{name}"] = torch.where(mask, values, torch.zeros_like(values)).sum() / count.clamp_min(1)
        result[f"metric_{name}_count"] = count.float()

    result["metric_valid_2d_fraction"] = valid.float().mean()
    result["metric_moving_2d_fraction"] = moving.sum().float() / valid.sum().clamp_min(1)
    emit("motion_2d_epe_px", epe, valid)
    emit("motion_2d_moving_epe_px", epe, moving)
    emit("motion_2d_static_drift_px", predicted_norm, static)
    emit("motion_2d_zero_baseline_epe_px", target_norm, valid)
    nonzero_reference = valid & (target_norm > 1e-6)
    directional = nonzero_reference & (predicted_norm > 1e-6)
    denom = (predicted_norm * target_norm).clamp_min(1e-12)
    cosine = ((p_px * t_px).sum(2) / denom).clamp(-1, 1)
    emit("motion_2d_direction_deg", torch.rad2deg(torch.acos(cosine)), directional)
    result["metric_motion_2d_direction_coverage"] = directional.sum().float() / nonzero_reference.sum().clamp_min(1)
    result["metric_motion_2d_zero_prediction_fraction"] = ((predicted_norm <= 1e-6) & valid).sum().float() / valid.sum().clamp_min(1)
    for index in range(p.shape[1]):
        tag = f"h{index+1:02d}"
        emit(f"motion_2d_epe_px_{tag}", epe[:, index], valid[:, index])
        emit(f"motion_2d_moving_epe_px_{tag}", epe[:, index], moving[:, index])
        emit(f"motion_2d_static_drift_px_{tag}", predicted_norm[:, index], static[:, index])
        result[f"metric_valid_2d_fraction_{tag}"] = valid[:, index].float().mean()
    if "motion_3d" in predictions:
        p3 = predictions["motion_3d"].detach().float()
        t3 = targets["motion_3d"].detach().float()
        valid3 = targets["valid_3d"].detach().bool() & torch.isfinite(t3).all(2)
        if p3.shape != t3.shape or p3.shape[:2] + p3.shape[-2:] != valid.shape or p3.shape[2] != 3:
            raise ValueError("3D metric fields must match the 2D anchor grid and horizons")
        if (valid3 & ~valid).any():
            raise ValueError("3D validity must be a subset of 2D validity")
        if not torch.isfinite(p3).all():
            raise FloatingPointError("Non-finite Student 3D motion metrics")
        t3 = torch.where(valid3[:, :, None], t3, torch.zeros_like(t3))
        epe3 = torch.linalg.vector_norm(p3 - t3, dim=2)
        drift3 = torch.linalg.vector_norm(p3, dim=2)
        result["metric_valid_3d_fraction"] = valid3.float().mean()
        emit("motion_3d_epe_normalized", epe3, valid3)
        emit("motion_3d_moving_epe_normalized", epe3, valid3 & moving)
        emit("motion_3d_static_drift_normalized", drift3, valid3 & static)
        for index in range(p3.shape[1]):
            tag = f"h{index+1:02d}"
            emit(f"motion_3d_epe_normalized_{tag}", epe3[:, index], valid3[:, index])
            result[f"metric_valid_3d_fraction_{tag}"] = valid3[:, index].float().mean()
    return result
