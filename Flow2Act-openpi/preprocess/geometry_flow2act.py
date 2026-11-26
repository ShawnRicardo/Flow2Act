"""UV-guided depth lifting in a single camera/scale convention (not raw XYZ)."""

import torch
import torch.nn.functional as functional


def anchor_grid_flow2act(image_hw=(256, 256), grid_hw=(32, 32), *, device=None):
    h, w = image_hw
    gh, gw = grid_hw
    y = (torch.arange(gh, device=device).float() + 0.5) * h / gh - 0.5
    x = (torch.arange(gw, device=device).float() + 0.5) * w / gw - 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), -1)


def _sample_depth(depth, confidence, xy):
    n, h, w = depth.shape
    finite_xy = torch.isfinite(xy).all(-1)
    safe_xy = torch.where(finite_xy[..., None], xy, torch.zeros_like(xy))
    grid = torch.stack(((safe_xy[..., 0] + 0.5) * 2 / w - 1, (safe_xy[..., 1] + 0.5) * 2 / h - 1), -1)
    finite = torch.isfinite(depth) & (depth > 0) & torch.isfinite(confidence) & (confidence > 2.0)
    safe_depth = torch.where(finite, depth, torch.zeros_like(depth))
    sampled = functional.grid_sample(
        safe_depth[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )[:, 0]
    coverage = functional.grid_sample(
        finite.float()[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )[:, 0]
    inside = (xy[..., 0] >= 0) & (xy[..., 0] <= w - 1) & (xy[..., 1] >= 0) & (xy[..., 1] <= h - 1)
    return sampled, finite_xy & inside & (coverage >= 1 - 1e-5) & (sampled > 0)


def lift_trajectories_flow2act(
    displacement_pixels, depths, intrinsics, camera_to_world, depth_confidence, *, image_hw=(256, 256)
):
    """Return normalized current-camera displacement for one anchor and H targets.

    displacement_pixels: [H,2,G,G], geometries: [H+1,...] (source then targets).
    Depth and all C2W translations must use the same scale. Full per-frame K is
    expressed on the original DA3 depth grid, before any XYZ interpolation.
    """
    displacement_pixels = displacement_pixels.float()
    depths, intrinsics, camera_to_world = depths.float(), intrinsics.float(), camera_to_world.float()
    depth_confidence = depth_confidence.float()
    horizon, _, gh, gw = displacement_pixels.shape
    if (
        len(depths) != horizon + 1
        or intrinsics.shape != (horizon + 1, 3, 3)
        or camera_to_world.shape != (horizon + 1, 4, 4)
    ):
        raise ValueError("Geometry must contain source and the exact H future frames")
    # A bad camera invalidates only its 3D labels. The independent UV labels
    # remain usable, so never abort a 2D export for a non-invertible camera.
    eye3 = torch.eye(3, device=intrinsics.device, dtype=intrinsics.dtype)
    eye4 = torch.eye(4, device=camera_to_world.device, dtype=camera_to_world.dtype)
    finite_k = torch.isfinite(intrinsics).all(dim=(-2, -1))
    finite_c = torch.isfinite(camera_to_world).all(dim=(-2, -1))
    safe_k = torch.where(finite_k[:, None, None], intrinsics, eye3)
    safe_c = torch.where(finite_c[:, None, None], camera_to_world, eye4)
    inverse_k, info_k = torch.linalg.inv_ex(safe_k)
    inverse_c, info_c = torch.linalg.inv_ex(safe_c)
    valid_k = finite_k & (info_k == 0) & torch.isfinite(inverse_k).all(dim=(-2, -1))
    valid_k &= (safe_k[:, 0, 0] > 0) & (safe_k[:, 1, 1] > 0)
    valid_c = finite_c & (info_c == 0) & torch.isfinite(inverse_c).all(dim=(-2, -1))
    valid_c &= torch.isclose(safe_c[:, 3], eye4[3], atol=1e-4, rtol=0).all(-1)
    valid_c &= torch.linalg.det(safe_c[:, :3, :3]) > 1e-6
    inverse_k = torch.where(valid_k[:, None, None], inverse_k, eye3)
    safe_c = torch.where(valid_c[:, None, None], safe_c, eye4)
    inverse_c = torch.where(valid_c[:, None, None], inverse_c, eye4)
    valid_camera = valid_k & valid_c
    dh, dw = depths.shape[-2:]
    ih, iw = image_hw
    p = anchor_grid_flow2act(image_hw, (gh, gw), device=depths.device)
    q = p[None] + displacement_pixels.permute(0, 2, 3, 1)
    xy = torch.cat((p[None], q), 0)
    resize = depths.new_tensor([dw / iw, dh / ih])
    xy_depth = (xy + 0.5) * resize - 0.5
    z, valid = _sample_depth(depths, depth_confidence, xy_depth)
    safe_xy_depth = torch.where(torch.isfinite(xy_depth), xy_depth, torch.zeros_like(xy_depth))
    homogeneous = torch.cat((safe_xy_depth, torch.ones_like(safe_xy_depth[..., :1])), -1)
    rays = torch.einsum("nij,nhwj->nhwi", inverse_k, homogeneous)
    points = rays * z[..., None]
    transform = inverse_c[0][None] @ safe_c[1:]
    endpoints = torch.einsum("nij,nhwj->nhwi", transform[:, :3, :3], points[1:]) + transform[:, None, None, :3, 3]
    displacement = endpoints - points[0][None]
    source_valid_depth = depths[0][
        torch.isfinite(depths[0]) & (depths[0] > 0) & torch.isfinite(depth_confidence[0]) & (depth_confidence[0] > 2)
    ]
    scale = source_valid_depth.median() if source_valid_depth.numel() else depths.new_tensor(0.0)
    valid3 = valid[0][None] & valid[1:] & torch.isfinite(displacement).all(-1) & (scale > 1e-6)
    valid3 &= (valid_camera[0] & valid_camera[1:])[:, None, None]
    normalized = displacement / scale.clamp_min(1e-6)
    normalized = torch.where(valid3[..., None], normalized, torch.zeros_like(normalized))
    return {
        "motion_3d": normalized.permute(0, 3, 1, 2),
        "valid_3d": valid3,
        "anchor_scale": scale,
        "source_points": points[0],
        "endpoints_current": endpoints,
        "endpoint_depth_xy": xy_depth[1:],
    }
