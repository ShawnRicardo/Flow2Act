# ruff: noqa: N801, N802, N999
"""Real LIBERO/cache alignment, branch isolation and deterministic batches."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from openpi.flow2act.config_flow2act import Flow2ActConfig
from openpi.flow2act.data_flow2act import Flow2ActDataset
from openpi.flow2act.data_flow2act import norm_pad_flow2act
from openpi.flow2act.data_flow2act import observation_flow2act
from openpi.flow2act.data_flow2act import parquet_flow2act


@pytest.mark.local_data
def test_h30_tail_and_cache_branch_isolation_flow2act(monkeypatch):
    paths = []
    load = np.load

    def record(path, *args, **kwargs):
        paths.append(str(path))
        return load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", record)
    cfg = dataclasses.replace(Flow2ActConfig(), stage="motion2d", workers=0, augment=False, memory_length=3)
    dataset = Flow2ActDataset(cfg)
    sample = dataset.sample((184, 3))  # last complete action window in first episode
    (_, directory, info, episode), frame = dataset.windows.locate(184)
    data = parquet_flow2act(directory, info, episode)
    assert frame == episode["length"] - 30
    expected = norm_pad_flow2act(data["action"][frame:], dataset.stats["actions"])
    np.testing.assert_array_equal(sample["actions"], expected)
    assert sample["actions"].shape == (30, 32)
    assert not sample["mask2d"][-1].any()
    assert np.count_nonzero(sample["motion2d"][-1]) == 0
    assert "motion3d" not in sample
    assert paths
    assert all("delta_xyz" not in path and "valid3d" not in path for path in paths)
    # No change to raw LIBERO gripper representation or supervised padding.
    assert (data["action"][:, -1] >= 0).all()
    assert (data["action"][:, -1] <= 1).all()
    assert np.count_nonzero(sample["actions"][:, 7:]) == 0
    beginning = dataset.sample((0, 0))
    assert not beginning["history_valid"].any()
    assert np.count_nonzero(beginning["history_state"]) == 0
    assert np.count_nonzero(beginning["history_rgb"]) == 0
    paths.clear()
    static = Flow2ActDataset(dataclasses.replace(cfg, stage="static"))
    assert static.cache is None
    static.sample((0, 0))
    assert paths == []
    baseline = Flow2ActDataset(dataclasses.replace(cfg, stage="baseline"))
    assert "teacher_rgb" not in baseline.sample((0, 0))


@pytest.mark.local_data
def test_batch_rng_across_workers_flow2act():
    cfg = dataclasses.replace(Flow2ActConfig(), stage="static", workers=0, augment=True)
    serial = Flow2ActDataset(cfg)
    parallel = Flow2ActDataset(dataclasses.replace(cfg, workers=2))
    indices = np.array([0, 90478, 130108, 224367])
    first, repeated = serial.batch(indices, 1280), parallel.batch(indices, 1280)
    for key in first:
        np.testing.assert_array_equal(first[key], repeated[key])
    unchanged_teacher = serial.batch(indices, 1281)
    np.testing.assert_array_equal(first["teacher_rgb"], unchanged_teacher["teacher_rgb"])
    assert not np.array_equal(first["rgb"], unchanged_teacher["rgb"])
    serial.close()
    parallel.close()


def test_full_fov_preprocessing_flow2act():
    image = np.zeros((1, 2, 256, 256, 3), np.uint8)
    # Single-pixel outer boundaries would disappear under any crop; distinct
    # corners also detect rotation. No padding can replace these output edges.
    image[:, :, 0, :, 0] = 255
    image[:, :, -1, :, 2] = 255
    image[:, :, :, 0, 1] = 255
    image[:, :, :, -1, :2] = 255
    obs = observation_flow2act(
        jnp.asarray(image), jnp.zeros((1, 32)), jnp.ones((1, 8), jnp.int32), jnp.ones((1, 8), bool)
    )
    output = (np.asarray(obs.images["base_0_rgb"])[0] + 1) / 2
    assert output.shape == (224, 224, 3)
    assert output[0, 100, 0] > 0.5
    assert output[-1, 100, 2] > 0.5
    assert output[100, 0, 1] > 0.5
    assert output[100, -1, 0] > 0.5
    assert output[100, -1, 1] > 0.5
    assert np.count_nonzero(output[32:-32, 32:-32]) == 0
    assert not obs.image_masks["right_wrist_0_rgb"].any()
