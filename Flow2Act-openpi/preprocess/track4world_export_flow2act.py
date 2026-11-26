"""Canonical-window native UV teacher; shared extraction, S16/stride8 decoder."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as functional

CONTEXT_FLOW2ACT = 128
OWNERSHIP_FLOW2ACT = 78
HORIZON_FLOW2ACT = 50
WINDOW_FLOW2ACT = 16
STRIDE_FLOW2ACT = 8


def canonical_windows_flow2act(length):
    if length < 1:
        raise ValueError("An episode must contain real frames")
    for start in range(0, length, OWNERSHIP_FLOW2ACT):
        stop = min(start + CONTEXT_FLOW2ACT, length)
        yield start, stop, min(start + OWNERSHIP_FLOW2ACT, length)


def gather_frames_flow2act(anchor, real_length):
    if not 0 <= anchor < real_length:
        raise IndexError(anchor)
    targets = anchor + np.arange(HORIZON_FLOW2ACT + 1)
    return np.minimum(targets, real_length - 1), targets[1:] < real_length


def _padded_sequence(value, indices, padded_length):
    result = value.index_select(0, indices)
    if len(result) < padded_length:
        result = torch.cat((result, result[-1:].expand(padded_length - len(result), *result.shape[1:])), 0)
    return result


@torch.no_grad()
def decode_anchor_flow2act(model, features, anchor, real_length, iters=4):
    """Use native iterative/time-attention updates, including source self slot."""
    indices, valid_horizon = gather_frames_flow2act(anchor, real_length)
    length = HORIZON_FLOW2ACT + 1
    starts = list(range(0, length - WINDOW_FLOW2ACT, STRIDE_FLOW2ACT))
    # The final overlapping window must cover the last real decoder slot.
    if not starts or starts[-1] + WINDOW_FLOW2ACT < length:
        starts.append(0 if not starts else starts[-1] + STRIDE_FLOW2ACT)
    padded_length = starts[-1] + WINDOW_FLOW2ACT
    device = features["fmaps"].device
    gather = torch.as_tensor(indices, device=device, dtype=torch.long)
    sequence = {
        name: _padded_sequence(features[name], gather, padded_length)[None]
        for name in ("fmaps", "ctxfeats", "fmaps3d_detail", "pms")
    }
    b, _, _, height, width = sequence["fmaps"].shape
    dtype = sequence["fmaps"].dtype
    flow = torch.zeros((b, WINDOW_FLOW2ACT, 2, height, width), device=device, dtype=dtype)
    xyz = torch.zeros((b, WINDOW_FLOW2ACT, 3, height, width), device=device, dtype=dtype)
    vis = torch.zeros_like(flow)
    result_flow = torch.empty((padded_length, 2, 32, 32), dtype=torch.float32)
    result_vis = torch.empty_like(result_flow)
    seen = torch.zeros(padded_length, dtype=torch.bool)
    for window_index, start in enumerate(starts):
        if window_index:
            # Native S16 overlap warm-start: retain last eight, repeat last state eight times.
            flow = torch.cat((flow[:, 8:16], flow[:, 15:16].expand(-1, 8, -1, -1, -1)), 1)
            xyz = torch.cat((xyz[:, 8:16], xyz[:, 15:16].expand(-1, 8, -1, -1, -1)), 1)
            vis = torch.cat((vis[:, 8:16], vis[:, 15:16].expand(-1, 8, -1, -1, -1)), 1)
        span = slice(start, start + WINDOW_FLOW2ACT)
        outputs = model.forward_window_unified(
            fmap1_single=sequence["fmaps"][:, 0],
            fmap2=sequence["fmaps"][:, span],
            fmaps3d_detail1_single=sequence["fmaps3d_detail"][:, 0],
            fmaps3d_detail2=sequence["fmaps3d_detail"][:, span],
            cxt1_single=sequence["ctxfeats"][:, 0],
            cxt2=sequence["ctxfeats"][:, span],
            pm1_single=sequence["pms"][:, 0].detach(),
            pm2=sequence["pms"][:, span].detach(),
            visconfs8=vis.reshape(b * 16, 2, height, width).detach(),
            flow2ds8=flow.reshape(b * 16, 2, height, width).detach(),
            flow3ds8=xyz.reshape(b * 16, 3, height, width).detach(),
            iters=iters,
            is_training=False,
            tracking3d=False,
        )
        flow_predictions, _, vis_predictions, flow, xyz, _ = outputs
        # Native learned full-resolution upsampling first; then sample current pixel-centre grid32.
        sampled = functional.interpolate(flow_predictions[-1].float(), (32, 32), mode="bilinear", align_corners=False)
        sampled_vis = functional.interpolate(
            vis_predictions[-1].float().sigmoid(), (32, 32), mode="bilinear", align_corners=False
        )
        # Native forward_sliding likewise fills only unvisited slots (model.py:1736-1746).
        # Overlap refinements update the warm-start state, not an already published slot.
        local = (~seen[span]).nonzero().flatten()
        absolute = local + start
        result_flow[absolute] = sampled[local.to(sampled.device)].cpu()
        result_vis[absolute] = sampled_vis[local.to(sampled_vis.device)].cpu()
        seen[span] = True
        flow = flow.reshape(b, 16, 2, height, width)
        xyz = xyz.reshape(b, 16, 3, height, width)
        # Native warm-start uses the decoder's low-resolution accumulated visibility logits.
        # forward_window_unified mutates visconfs8 by addition but returns only its prediction;
        # capture the final raw low-resolution state via the dedicated hook set by our native copy.
        vis = model.native_vis_state_flow2act.reshape(b, 16, 2, height, width)
    return result_flow[1:length], result_vis[1:length], valid_horizon


class NativeUVTeacherFlow2Act:
    def __init__(self, track4world_root, checkpoint=None, device="cuda:0", *, keep_allocator_cache=False):
        if not isinstance(keep_allocator_cache, bool):
            raise TypeError("keep_allocator_cache must be an explicit bool")
        self.root = Path(track4world_root)
        sys.path.insert(0, str(self.root))
        from .track4world_model_flow2act import Track4WorldFlow2Act

        settings = json.loads((self.root / "track4world/config/eval/v1.json").read_text())["model"]
        self.model = Track4WorldFlow2Act(**settings, seqlen=16, use_3d=False, use_model="depthanythingv3")
        self.model.keep_allocator_cache = keep_allocator_cache
        checkpoint = Path(checkpoint) if checkpoint else self.root / "checkpoints/track4world_da3.pth"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = self.model.load_pretrained_with_remap(state)
        # The optional nested metric model comes from its official local pretrained checkpoint.
        bad_missing = [name for name in missing if not name.startswith("backbone.model.da3_metric.")]
        absent3d = ("flow_aggregator3d.", "dot_fc2.", "flow3d_head.", "flow_3d_upsample_weight.", "time_emb3d")
        bad_unexpected = [name for name in unexpected if not name.startswith(absent3d)]
        if bad_missing or bad_unexpected:
            raise RuntimeError(f"Native 2D teacher weight mismatch: missing={bad_missing}, unexpected={bad_unexpected}")
        self.device = torch.device(device)
        self.model.to(self.device).eval().requires_grad_(requires_grad=False)

    @torch.no_grad()
    def extract_context(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1:] != (3, 256, 256) or not 1 <= len(rgb) <= 128:
            raise ValueError("One real canonical context must be [T<=128,3,256,256]")
        raw = rgb.to(self.device, dtype=torch.float32)
        normalized = (raw / 255.0 - self.model.image_mean) / self.model.image_std
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            values = self.model.forward_point(
                normalized[None].to(torch.float16 if self.device.type == "cuda" else torch.float32),
                current_batch_size=1,
                for_flow=True,
            )
        names = ("fmaps", "ctxfeats", "fmaps3d_detail", "pms")
        features = {name: value.detach() for name, value in zip(names, values[:4], strict=True)}
        geometry = {key: value.detach() for key, value in self.model.native_geometry_flow2act.items()}
        if len(geometry["depth"]) != len(rgb):
            raise ValueError("Geometry extraction must see only real frames")
        return features, geometry

    @torch.no_grad()
    def decode_anchor(self, features, anchor, real_length, iters=4):
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            return decode_anchor_flow2act(self.model, features, anchor, real_length, iters)


def self_test_flow2act():
    for length in (1, 2, 50, 77, 78, 79, 128, 129, 155, 156, 157, 300):
        owned = []
        for start, stop, end in canonical_windows_flow2act(length):
            assert stop - start <= 128
            assert end - start <= 78
            owned.extend(range(start, end))
            for anchor in range(start, end):
                ids, valid = gather_frames_flow2act(anchor - start, stop - start)
                assert ids[0] == anchor - start
                assert len(ids) == 51
                assert np.array_equal(valid, anchor + np.arange(1, 51) < length)
                assert np.all(ids[1:][valid] == anchor - start + np.arange(1, 51)[valid])
        assert owned == list(range(length))

    class Fake:
        def __init__(self):
            self.calls = []

        def forward_window_unified(self, **kwargs):
            assert kwargs["tracking3d"] is False
            target = kwargs["fmap2"][0, :, 0, 0, 0]
            if self.calls:
                assert torch.equal(kwargs["flow2ds8"][:8, 0, 0, 0], self.calls[-1][8:16])
            self.calls.append(target.clone())
            flows = target[:, None, None, None].expand(16, 2, 256, 256).clone()
            f = target[:, None, None, None].expand(16, 2, 1, 1).clone()
            self.native_vis_state_flow2act = torch.zeros_like(f)
            return [flows], None, [torch.zeros_like(flows)], f, torch.zeros(16, 3, 1, 1), None

    features = {
        name: torch.arange(128).float()[:, None, None, None] for name in ("fmaps", "ctxfeats", "fmaps3d_detail", "pms")
    }
    fake = Fake()
    flow, vis, valid = decode_anchor_flow2act(fake, features, 17, 128)
    assert torch.equal(flow[:, 0, 0, 0], torch.arange(18, 68).float())
    assert flow.shape == (50, 2, 32, 32)
    assert torch.equal(vis, torch.full_like(vis, 0.5))
    assert len(fake.calls) == 6
    assert valid.all()
    print(
        "canonical128/ownership78, source-self+h1..50, S16/stride8, warm-start, full-resolution-centre-grid checks passed"
    )
