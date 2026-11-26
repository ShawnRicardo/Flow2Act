"""Frozen rank-local joint-view DA3 teacher, excluded from student checkpoints."""
from contextlib import nullcontext
from pathlib import Path
import torch
from torch import nn

from .da3.da3_anyview_teacher import (
    DA3_BACKBONE_FILENAME_FLOW2ACT,
    build_da3_anyview_giant_flow2act,
    strict_load_safetensors_flow2act,
)


class FrozenDA3TeacherFlow2Act(nn.Module):
    def __init__(self, weights_dir, device="cpu", mmap_weights=False):
        super().__init__()
        if type(mmap_weights) is not bool:
            raise TypeError("mmap_weights must be boolean")
        device = torch.device(device)
        if mmap_weights and device.type != "cpu":
            raise ValueError("Shared DA3 weights support CPU compute only")
        self.mmap_weights = mmap_weights
        with torch.device("meta"):
            self.backbone = build_da3_anyview_giant_flow2act((19, 27, 33, 39))
        if not mmap_weights:
            self.backbone.to_empty(device=device)
        path = Path(weights_dir)
        if path.is_dir():
            path = path / DA3_BACKBONE_FILENAME_FLOW2ACT
        strict_load_safetensors_flow2act(self.backbone, path, assign_cpu=mmap_weights)
        self.backbone.eval().requires_grad_(False)
        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, rgb):
        if rgb.ndim != 5 or rgb.shape[1:] != (2, 3, 252, 252):
            raise ValueError("DA3 expects current joint main/wrist [B,2,3,252,252]")
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0
        if not torch.isfinite(rgb).all() or (rgb < 0).any() or (rgb > 1).any():
            raise ValueError("Teacher RGB must be unaugmented float [0,1] or uint8")
        device = self.image_mean.device
        parts = [[], [], [], []]
        for sample in rgb.split(1):
            value = (sample.to(device=device, dtype=torch.float32) - self.image_mean) / self.image_std
            autocast = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
            with autocast:
                pyramid, _ = self.backbone(value, cam_token=None, export_feat_layers=[], ref_view_strategy="first")
            for layer, entry in zip(parts, pyramid, strict=True):
                feature = entry[0]
                if feature.shape != (1, 2, 324, 3072):
                    raise ValueError(f"Unexpected DA3 feature shape {feature.shape}")
                layer.append(feature.reshape(1, 2, 18, 18, 3072).detach())
        return tuple(torch.cat(layer) for layer in parts)
