"""Native-grid static distillation and causal VLM suffix for staged ablations.

No external temporal decoder, action bridge or spatial upsampler belongs to the
policy. Future queries are contextualized by the caller's original VLM.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .config import Flow2ActConfig
from .losses import (
    feature_loss_flow2act, motion_metrics_flow2act, trajectory_loss_flow2act,
)


def _positions(value, width):
    pairs = (width + 1) // 2
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(pairs, device=value.device) / max(pairs - 1, 1))
    phase = value.float().unsqueeze(-1) * frequencies
    return torch.cat((phase.sin(), phase.cos()), -1)[..., :width]


def _grid_positions(height, width, channels, device):
    y, x = torch.meshgrid(torch.linspace(-1, 1, height, device=device),
                          torch.linspace(-1, 1, width, device=device), indexing="ij")
    first = channels // 2
    return torch.cat((_positions(x, first), _positions(y, channels - first)), -1)


class SharedChannelAdapterFlow2Act(nn.Module):
    """One complete LN/Linear/GELU/Linear reused by every tap and camera."""

    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.vlm_hidden_dim)
        self.input_projection = nn.Linear(config.vlm_hidden_dim, config.adapter_hidden_dim)
        self.output_projection = nn.Linear(config.adapter_hidden_dim, config.teacher_dim)

    def forward(self, grid):
        d = grid.shape[-1]
        value = F.layer_norm(grid.float(), (d,), self.norm.weight.float(), self.norm.bias.float(), self.norm.eps)
        value = F.gelu(self.input_projection(value.to(self.input_projection.weight.dtype)))
        return self.output_projection(value)


class Flow2ActGeometryMotion(nn.Module):
    def __init__(self, config: Flow2ActConfig):
        super().__init__()
        self.config = config
        if config.has_static:
            self.shared_adapter = SharedChannelAdapterFlow2Act(config)
        if config.has_motion:
            d = config.vlm_hidden_dim
            self.future_queries = nn.Parameter(torch.randn(config.future_horizon, d) * 0.02)
            self.memory_norm = nn.LayerNorm(d)
            self.memory_projection = nn.Linear(d, d)
            self.state_projection = nn.Linear(config.state_dim, d)
            self.camera_embedding = nn.Embedding(2, d)
            self.anchor_projection = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, config.width), nn.GELU())
            self.readout_future = nn.Linear(d, config.width)
            self.readout = nn.Sequential(nn.LayerNorm(config.width),
                                         nn.Linear(config.width, config.readout_hidden_dim), nn.GELU())
            self.motion_2d_head = nn.Linear(config.readout_hidden_dim, 2)
            if config.has_3d:
                self.motion_3d_head = nn.Linear(config.readout_hidden_dim, 3)

    def _validate_grid(self, value):
        if value.ndim != 5 or value.shape[1] != 2 or value.shape[-1] != self.config.vlm_hidden_dim:
            raise ValueError("Visual features must be [B,2,H,W,vlm_hidden_dim]")
        if min(value.shape) < 1:
            raise ValueError("Visual feature grids must be nonempty")

    def static_predictions(self, tap_grids):
        if not self.config.has_static:
            return ()
        if len(tap_grids) != 4:
            raise ValueError("Static distillation requires four VLM tap grids")
        for grid in tap_grids:
            self._validate_grid(grid)
        return tuple(self.shared_adapter(grid) for grid in tap_grids)

    def build_suffix(self, history_grids, history_states, history_valid, history_ages, current_state):
        cfg = self.config
        if current_state.ndim != 2 or current_state.shape[-1] != cfg.state_dim:
            raise ValueError("Current state must be [B,state_dim]")
        b = current_state.shape[0]
        if not cfg.has_motion:
            return current_state.new_empty((b, 0, cfg.vlm_hidden_dim)), torch.empty((b, 0), device=current_state.device, dtype=torch.bool)
        expected = cfg.memory_length - 1
        if len(history_grids) != expected or history_states.shape != (b, expected, cfg.state_dim):
            raise ValueError("History grids/states must contain exactly memory_length-1 observations")
        if history_valid.shape != (b, expected) or history_ages.shape != history_valid.shape:
            raise ValueError("History validity and ages must be [B,memory_length-1]")
        valid = history_valid.bool()
        if (history_ages[valid] <= 0).any():
            raise ValueError("Valid history ages must be positive past decision offsets")
        parts, masks = [], []
        dtype = self.memory_projection.weight.dtype
        for j, grid in enumerate(history_grids):
            self._validate_grid(grid)
            if grid.shape[0] != b:
                raise ValueError("History batch differs from current state")
            keep = valid[:, j]
            safe_grid = torch.where(keep[:, None, None, None, None], grid, torch.zeros_like(grid))
            value = self.memory_projection(self.memory_norm(safe_grid.to(dtype)))
            h, w = grid.shape[2:4]
            positions = _grid_positions(h, w, cfg.vlm_hidden_dim, grid.device).to(dtype)
            cameras = self.camera_embedding(torch.arange(2, device=grid.device))
            age = torch.where(keep, history_ages[:, j], torch.zeros_like(history_ages[:, j]))
            age_embed = _positions(age, cfg.vlm_hidden_dim).to(dtype)
            value = value + positions[None, None] + cameras[None, :, None, None] + age_embed[:, None, None, None]
            value = value.reshape(b, 2 * h * w, cfg.vlm_hidden_dim)
            old_state = torch.where(keep[:, None], history_states[:, j], torch.zeros_like(history_states[:, j]))
            state_token = self.state_projection(old_state.to(dtype)) + age_embed
            value = torch.cat((value, state_token[:, None]), 1)
            mask = keep[:, None].expand(-1, value.shape[1])
            parts.append(torch.where(mask[..., None], value, torch.zeros_like(value)))
            masks.append(mask)
        state_token = self.state_projection(current_state.to(dtype))[:, None]
        parts.extend((state_token, self.future_queries[None].expand(b, -1, -1)))
        masks.extend((torch.ones((b, 1), device=current_state.device, dtype=torch.bool),
                      torch.ones((b, cfg.future_horizon), device=current_state.device, dtype=torch.bool)))
        return torch.cat(parts, 1), torch.cat(masks, 1)

    def read_motion(self, query_hidden, current_visual_grid):
        cfg = self.config
        if not cfg.has_motion:
            return {}
        self._validate_grid(current_visual_grid)
        b = current_visual_grid.shape[0]
        if query_hidden.shape != (b, cfg.future_horizon, cfg.vlm_hidden_dim):
            raise ValueError("VLM future-query hidden states must be [B,H_F,vlm_hidden_dim]")
        anchor = self.anchor_projection(current_visual_grid[:, 0].to(self.readout_future.weight.dtype))
        anchor = F.interpolate(anchor.permute(0, 3, 1, 2), (cfg.motion_grid_size,) * 2,
                               mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        positions = _grid_positions(cfg.motion_grid_size, cfg.motion_grid_size, cfg.width, anchor.device).to(anchor.dtype)
        anchor = anchor + positions[None]
        chunks2, chunks3 = [], []
        for start in range(0, cfg.future_horizon, cfg.readout_horizon_chunk):
            future = self.readout_future(query_hidden[:, start:start + cfg.readout_horizon_chunk].to(self.readout_future.weight.dtype))
            value = self.readout(anchor[:, None] + future[:, :, None, None])
            chunks2.append(self.motion_2d_head(value).permute(0, 1, 4, 2, 3))
            if cfg.has_3d:
                chunks3.append(self.motion_3d_head(value).permute(0, 1, 4, 2, 3))
        result = {"motion_2d": torch.cat(chunks2, 1)}
        if chunks3:
            result["motion_3d"] = torch.cat(chunks3, 1)
        return result

    def auxiliary_losses(self, tap_grids, teacher_features, predictions, targets, view_mask=None):
        cfg = self.config
        if tap_grids:
            zero = tap_grids[0].float().sum() * 0.0
        elif predictions:
            zero = next(iter(predictions.values())).float().sum() * 0.0
        else:
            zero = torch.tensor(0.0)
        result = {"loss": zero}
        if cfg.has_static:
            geo = feature_loss_flow2act(self.static_predictions(tap_grids), teacher_features, view_mask)
            result.update(loss=cfg.geo_loss_weight * geo, loss_geo=geo)
        if cfg.has_motion:
            motion2 = trajectory_loss_flow2act(
                predictions["motion_2d"], targets["motion_2d"], targets["valid_2d"], beta=cfg.motion_2d_beta,
                moving_target=targets["motion_2d"], image_size=cfg.image_size, moving_threshold_px=cfg.moving_threshold_px)
            result["loss_motion_2d"] = motion2
            result["loss"] = result["loss"] + cfg.motion_2d_loss_weight * motion2
            if cfg.has_3d:
                motion3 = trajectory_loss_flow2act(
                    predictions["motion_3d"], targets["motion_3d"], targets["valid_3d"], beta=cfg.motion_3d_beta)
                result["loss_motion_3d"] = motion3
                result["loss"] = result["loss"] + cfg.motion_3d_loss_weight * motion3
            result.update(motion_metrics_flow2act(predictions, targets, image_size=cfg.image_size,
                                                       moving_threshold_px=cfg.moving_threshold_px))
        return result
