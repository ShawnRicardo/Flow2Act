"""Shared cache names for standalone preprocessing and the JAX data loader."""

import json
from pathlib import Path

SCHEMA_FLOW2ACT = "Flow2Act-motion-v1"
COMPATIBLE_SCHEMAS_FLOW2ACT = (SCHEMA_FLOW2ACT, "flow2act-motion-v1")
NATIVE_UV_CONTRACT_FLOW2ACT = {
    "coordinate": "displacement_xy_div_wh",
    "image_hw": [256, 256],
    "grid_hw": [32, 32],
    "source_camera": "observation.images.image",
    "canonical_context_frames": 128,
    "anchor_ownership": 78,
    "decoder_sequence_length": 16,
    "decoder_stride": 8,
    "decoder_iterations": 4,
    "decoder_temporal_attention": True,
    "decoder_overlap_warm_start": True,
    "geometry_batch": 1,
    "geometry_real_frames_only": True,
    "decoder_source_self_slot": True,
    "visibility": "native_sigmoid_visibility_and_confidence",
    "endpoint_bounds_before_float16": True,
    "optional_3d": "native_UV_plus_original_DA3_depth_fullK_C2W_in_current_camera_div_anchor_median_depth",
    "raw_xyz_flow_used": False,
}
FILES_FLOW2ACT = {
    "motion_2d": "displacement_normalized_flow2act.npy",
    "visconf": "native_visconf_flow2act.npy",
    "in_bounds": "in_bounds_flow2act.npy",
    "horizon_valid": "horizon_valid_flow2act.npy",
    "motion_3d": "delta_xyz_normalized_flow2act.npy",
    "valid_3d": "valid3d_flow2act.npy",
    "anchor_scale": "anchor_scale_flow2act.npy",
}


def read_cache_layout_flow2act(root):
    """Read one committed header; derive the existing array suffix without writing anything."""
    root = Path(root)
    headers = sorted(root.glob("metadata*.json"))
    if len(headers) != 1:
        raise ValueError(f"Expected exactly one completed metadata*.json in {root}; found {len(headers)}")
    header = headers[0]
    metadata = json.loads(header.read_text())
    if not isinstance(metadata, dict):
        raise ValueError("Motion cache metadata must be a JSON object")
    schema = metadata.get("schema")
    if not isinstance(schema, str) or not schema:
        raise ValueError("Motion cache must declare a nonempty schema")
    # Archived names are accepted only with the full native-UV scientific contract.
    basic = ("coordinate", "image_hw", "grid_hw", "source_camera")
    required = tuple(
        field
        for field in NATIVE_UV_CONTRACT_FLOW2ACT
        if schema not in COMPATIBLE_SCHEMAS_FLOW2ACT or field in basic or field in metadata
    )
    for field in required:
        expected = NATIVE_UV_CONTRACT_FLOW2ACT[field]
        actual = metadata.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"Motion cache contract mismatch: {field}")
    if type(metadata.get("horizon")) is not int or metadata["horizon"] < 1:
        raise ValueError("Motion cache horizon must be a positive integer")
    if type(metadata.get("has_motion_3d")) is not bool:
        raise ValueError("Motion cache must declare whether 3D labels are available")
    return header, metadata, header.stem.removeprefix("metadata")


def canonical_suite_flow2act(suite):
    aliases = {
        "10": "libero_10",
        "long": "libero_10",
        "goal": "libero_goal",
        "object": "libero_object",
        "spatial": "libero_spatial",
    }
    suite = aliases.get(str(suite), str(suite))
    if suite not in ("libero_10", "libero_goal", "libero_object", "libero_spatial"):
        raise ValueError(f"Unknown LIBERO suite: {suite!r}")
    return suite
