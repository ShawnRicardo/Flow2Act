"""The shipped exporter produces the exact cache and Teacher artifacts JAX expects."""

import json
import math

import numpy as np
import pytest
import torch

from openpi.flow2act.cache_flow2act import COMPATIBLE_SCHEMAS_FLOW2ACT
from openpi.flow2act.cache_flow2act import SCHEMA_FLOW2ACT
from openpi.flow2act.data_flow2act import Flow2ActMotionCache
from openpi.flow2act.teacher_contract_flow2act import DEFAULT_TEACHER_FLOW2ACT
from openpi.flow2act.teacher_contract_flow2act import expected_keys_flow2act
from preprocess.export_motion_flow2act import cache_metadata_flow2act
from preprocess.export_motion_flow2act import export_episode_flow2act
from preprocess.extract_da3_weights_flow2act import ARTIFACT_SPECS
from preprocess.extract_da3_weights_flow2act import ArtifactSpec
from preprocess.extract_da3_weights_flow2act import extract_all
from preprocess.track4world_export_flow2act import gather_frames_flow2act
from preprocess.track4world_export_flow2act import self_test_flow2act


class FakeTeacherFlow2Act:
    def extract_context(self, frames):
        count = len(frames)
        intrinsics = torch.eye(3).repeat(count, 1, 1)
        intrinsics[:, 0, 0] = intrinsics[:, 1, 1] = 200
        intrinsics[:, :2, 2] = 127.5
        return {}, {
            "depth": torch.full((count, 256, 256), 2.0),
            "depth_confidence": torch.full((count, 256, 256), 3.0),
            "intrinsics": intrinsics,
            "camera_to_world": torch.eye(4).repeat(count, 1, 1),
        }

    def decode_anchor(self, features, anchor, real_length, iters=4):
        _, valid = gather_frames_flow2act(anchor, real_length)
        flow = torch.zeros(50, 2, 32, 32)
        flow[:, 0] = torch.arange(1, 51)[:, None, None]
        return flow, torch.ones_like(flow), valid


def test_canonical_decoder_without_weights_flow2act():
    self_test_flow2act()


@pytest.mark.parametrize("schema", COMPATIBLE_SCHEMAS_FLOW2ACT)
def test_export_to_jax_reader_and_3d_geometry_flow2act(tmp_path, schema):
    metadata = cache_metadata_flow2act(motion_3d_enabled=True)
    assert metadata["schema"] == SCHEMA_FLOW2ACT
    metadata["schema"] = schema
    (tmp_path / "metadata_flow2act.json").write_text(json.dumps(metadata))
    episode = tmp_path / "libero_10/episode_000000"
    export_episode_flow2act(FakeTeacherFlow2Act(), torch.zeros(3, 3, 256, 256), episode, motion_3d=True)
    reader = Flow2ActMotionCache(tmp_path, 30, three_dimensional=True)
    sample = reader.load("libero_10", 0, 0, 3)
    assert sample["motion2d"].shape == (30, 2, 32, 32)
    assert sample["motion3d"].shape == (30, 3, 32, 32)
    assert sample["mask2d"][:2].all()
    assert not sample["mask2d"][2:].any()
    assert sample["mask3d"][:2].all()
    assert not sample["mask3d"][2:].any()
    np.testing.assert_allclose(sample["motion2d"][0, 0], 1 / 256, atol=1e-6)
    np.testing.assert_allclose(sample["motion3d"][0, 0], 1 / 200, atol=3e-6)
    assert np.count_nonzero(sample["motion3d"][0, 1:]) == 0
    assert not reader.load("libero_10", 0, 2, 3)["mask2d"].any()
    assert not list(episode.glob("*.partial"))
    with pytest.raises(FileExistsError):
        export_episode_flow2act(FakeTeacherFlow2Act(), torch.zeros(3, 3, 256, 256), episode, motion_3d=True)
    for path in episode.iterdir():
        assert path.stat().st_mode & 0o777 == 0o777


def test_extractor_shared_contract_and_exact_values_flow2act(tmp_path):
    giant = ARTIFACT_SPECS[0]
    assert giant.filename == "da3_giant_flow2act.safetensors"
    assert giant.expected_shapes == expected_keys_flow2act(DEFAULT_TEACHER_FLOW2ACT)
    assert giant.expected_key_count == len(giant.expected_shapes)
    assert giant.expected_numel == sum(math.prod(shape) for shape in giant.expected_shapes.values())
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    source = tmp_path / "source.pth"
    torch.save({"state_dict": {"backbone.model.backbone.pretrained.test": tensor}}, source)
    spec = ArtifactSpec("test.safetensors", "backbone.model.backbone.", 1, 6, {"pretrained.test": (2, 3)})
    output = tmp_path / "extracted"
    paths = extract_all(source, output, specs=(spec,))
    from safetensors import safe_open

    with safe_open(paths[0], framework="pt") as artifact:
        assert torch.equal(artifact.get_tensor("pretrained.test"), tensor)
    # A second explicit extraction may reuse identical values, but never replace different weights.
    extract_all(source, output, specs=(spec,))
    torch.save({"backbone.model.backbone.pretrained.test": tensor + 1}, source)
    with pytest.raises(RuntimeError, match="different tensor values"):
        extract_all(source, output, specs=(spec,))
