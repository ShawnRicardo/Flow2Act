"""Archived cache reuse is read-only and preserves labels, masks and dataset alignment."""

import json

import numpy as np
import pytest

from openpi.flow2act.cache_flow2act import FILES_FLOW2ACT
from openpi.flow2act.cache_flow2act import NATIVE_UV_CONTRACT_FLOW2ACT
from openpi.flow2act.cache_flow2act import SCHEMA_FLOW2ACT
from openpi.flow2act.data_flow2act import SUITES_FLOW2ACT
from openpi.flow2act.data_flow2act import Flow2ActMotionCache
from scripts.check_motion_cache_flow2act import check_motion_cache_flow2act


@pytest.fixture
def shared_cache_flow2act(tmp_path):
    cache, data = tmp_path / "shared cache", tmp_path / "data"
    cache.mkdir()
    metadata = {"schema": SCHEMA_FLOW2ACT, "horizon": 5, "has_motion_3d": True, **NATIVE_UV_CONTRACT_FLOW2ACT}
    (cache / "metadata_flow2act.json").write_text(json.dumps(metadata))
    for suite in SUITES_FLOW2ACT:
        episode = cache / suite / "episode_000000"
        episode.mkdir(parents=True)
        values = {
            "motion_2d": np.arange(4 * 5 * 2 * 32 * 32, dtype=np.float32).reshape(4, 5, 2, 32, 32) / 1e6,
            "visconf": np.ones((4, 5, 2, 32, 32), np.float16),
            "in_bounds": np.ones((4, 5, 32, 32), bool),
            "horizon_valid": np.arange(4)[:, None] + np.arange(1, 6) < 4,
            "motion_3d": np.arange(4 * 5 * 3 * 32 * 32, dtype=np.float32).reshape(4, 5, 3, 32, 32) / 1e6,
            "valid_3d": np.ones((4, 5, 32, 32), bool),
        }
        for key, array in values.items():
            np.save(episode / FILES_FLOW2ACT[key], array)
        directory = data / f"{suite}_no_noops_1.0.0_lerobot" / "meta"
        directory.mkdir(parents=True)
        (directory / "info.json").write_text(json.dumps({"codebase_version": "v2.1", "fps": 20}))
        (directory / "episodes.jsonl").write_text(
            json.dumps({"episode_index": 0, "length": 4}) + "\n" + json.dumps({"episode_index": 1, "length": 1}) + "\n"
        )
    return cache, data


def archive_names_flow2act(cache):
    header = cache / "metadata_flow2act.json"
    metadata = json.loads(header.read_text())
    metadata["schema"] = "ArchivedNativeUV-v1"
    metadata["motion_3d_uv_precision"] = "source_float16_normalized_UV_times_256_no_redecode"
    archived = cache / "metadata_archived.json"
    header.rename(archived)
    archived.write_text(json.dumps(metadata))
    for path in cache.glob("libero_*/episode_*/*_flow2act.npy"):
        path.rename(path.with_name(path.name.replace("_flow2act.npy", "_archived.npy")))


def snapshot_flow2act(root):
    return {
        path.relative_to(root): (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_mode)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_archived_labels_masks_and_full_check_are_read_only_flow2act(shared_cache_flow2act, monkeypatch):
    cache, data = shared_cache_flow2act
    current = Flow2ActMotionCache(cache, 3, three_dimensional=True)
    expected = {(suite, frame): current.load(suite, 0, frame, 4) for suite in SUITES_FLOW2ACT for frame in (0, 2)}
    archive_names_flow2act(cache)
    before = snapshot_flow2act(cache)
    load = np.load

    def read_only(path, *args, **kwargs):
        assert kwargs["mmap_mode"] == "r"
        assert kwargs["allow_pickle"] is False
        array = load(path, *args, **kwargs)
        assert not array.flags.writeable
        return array

    monkeypatch.setattr(np, "load", read_only)
    reader = Flow2ActMotionCache(cache, 3, three_dimensional=True)
    for (suite, frame), original in expected.items():
        result = reader.load(suite, 0, frame, 4)
        for key in original:
            np.testing.assert_array_equal(result[key], original[key])
    report = check_motion_cache_flow2act(cache, data, action_horizon=2, future_horizon=3)
    assert report["episodes"] == 4
    assert report["complete_windows"] == 12
    assert report["cache_horizon"] == 5
    assert snapshot_flow2act(cache) == before
    assert reader.metadata["motion_3d_uv_precision"] == "source_float16_normalized_UV_times_256_no_redecode"


@pytest.mark.parametrize(
    ("field", "value"), [("decoder_stride", 4), ("raw_xyz_flow_used", True), ("grid_hw", [16, 16])]
)
def test_reject_incompatible_archived_contract_flow2act(shared_cache_flow2act, field, value):
    cache, _ = shared_cache_flow2act
    archive_names_flow2act(cache)
    header = cache / "metadata_archived.json"
    metadata = json.loads(header.read_text())
    metadata[field] = value
    header.write_text(json.dumps(metadata))
    before = snapshot_flow2act(cache)
    with pytest.raises(ValueError, match="contract mismatch"):
        Flow2ActMotionCache(cache, 3, three_dimensional=True)
    assert snapshot_flow2act(cache) == before


def test_reject_ambiguous_headers_and_missing_data_flow2act(shared_cache_flow2act):
    cache, data = shared_cache_flow2act
    other = cache / "metadata_another.json"
    other.write_text((cache / "metadata_flow2act.json").read_text())
    with pytest.raises(ValueError, match="exactly one"):
        Flow2ActMotionCache(cache, 3, three_dimensional=True)
    other.unlink()
    for suite in SUITES_FLOW2ACT:
        (cache / suite / "episode_000000" / FILES_FLOW2ACT["motion_3d"]).unlink()
    # The 2D ablation must not read the absent 3D arrays.
    report = check_motion_cache_flow2act(cache, data, action_horizon=2, future_horizon=3, motion_3d=False)
    assert report["episodes"] == 4
    with pytest.raises(FileNotFoundError):
        check_motion_cache_flow2act(cache, data, action_horizon=2, future_horizon=3)


def test_array_headers_must_match_dataset_frames_flow2act(shared_cache_flow2act):
    cache, data = shared_cache_flow2act
    path = cache / "libero_10/episode_000000" / FILES_FLOW2ACT["motion_2d"]
    np.save(path, np.zeros((5, 5, 2, 32, 32), np.float16))
    with pytest.raises(ValueError, match="Malformed motion cache"):
        check_motion_cache_flow2act(cache, data, action_horizon=2, future_horizon=3)


def test_missing_cache_never_starts_generation_flow2act(tmp_path):
    cache = tmp_path / "absent"
    with pytest.raises(ValueError, match="exactly one"):
        Flow2ActMotionCache(cache, 30, three_dimensional=True)
    assert not cache.exists()
