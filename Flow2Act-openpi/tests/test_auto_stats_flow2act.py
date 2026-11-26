# ruff: noqa: N802, N999
"""Automatic normalization preserves complete-window weighting and validated cached values."""

from concurrent.futures import ThreadPoolExecutor
import json
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.flow2act import stats_flow2act
from openpi.flow2act.data_flow2act import SUITES_FLOW2ACT
from openpi.flow2act.data_flow2act import Flow2ActWindows
from openpi.shared import normalize


@pytest.fixture
def training_windows_flow2act(tmp_path):
    root = tmp_path / "libero data"
    records = []
    for suite_index, suite in enumerate(SUITES_FLOW2ACT):
        directory = root / f"{suite}_no_noops_1.0.0_lerobot"
        (directory / "meta").mkdir(parents=True)
        info = {
            "codebase_version": "v2.1",
            "fps": 20,
            "chunks_size": 1000,
            "data_path": "data/episode_{episode_index:06d}.parquet",
        }
        (directory / "meta/info.json").write_text(json.dumps(info))
        episodes = [{"episode_index": 0, "length": 32}, {"episode_index": 1, "length": 29}]
        (directory / "meta/episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes))
        state = np.arange(32 * 8, dtype=np.float32).reshape(32, 8) / 64 + suite_index
        action = np.arange(32 * 7, dtype=np.float32).reshape(32, 7) / 128 - suite_index
        action[:, -1] = np.arange(32) % 2
        table = pa.table(
            {
                "observation.state": state.tolist(),
                "action": action.tolist(),
                "frame_index": list(range(32)),
                "episode_index": [0] * 32,
            }
        )
        (directory / "data").mkdir()
        pq.write_table(table, directory / "data/episode_000000.parquet")
        # The short episode has no parquet or videos: it must never be read.
        records.append((state.astype(np.float64), action.astype(np.float64)))
    return Flow2ActWindows(root, 30), records


def test_fresh_stats_match_previous_algorithm_flow2act(tmp_path, training_windows_flow2act):
    windows, records = training_windows_flow2act
    output = tmp_path / "stats"
    before = np.random.get_state()
    content = stats_flow2act.prepare_stats_flow2act(windows, output, if_missing=True)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    result = json.loads(content)
    assert result["sampling"]["complete_windows"] == [3, 3, 3, 3]
    assert result["sampling"]["total_windows"] == 12
    expected_states = np.concatenate([state[:3] for state, _ in records])
    expected_actions = np.concatenate([action[t : t + 30] for _, action in records for t in range(3)])
    np.testing.assert_allclose(result["norm_stats"]["state"]["mean"], expected_states.mean(0))
    np.testing.assert_allclose(result["norm_stats"]["actions"]["mean"], expected_actions.mean(0))
    # Exact legacy histogram/FP64 parity, including q01/q99 and episode order.
    state_stats, action_stats = normalize.RunningStats(), normalize.RunningStats()
    for state, action in records:
        state_stats.update(state[:3])
        action_stats.update(np.stack([action[t : t + 30] for t in range(3)]))
    previous = json.loads(
        normalize.serialize_json({"state": state_stats.get_statistics(), "actions": action_stats.get_statistics()})
    )
    previous["sampling"] = windows.sampling_spec()
    assert content == json.dumps(previous, indent=2) + "\n"
    assert (output / "norm_stats.json").stat().st_mode & 0o777 == 0o777
    assert [p.name for p in output.iterdir()] == ["norm_stats.json"]


def fail_scan_flow2act(*args, **kwargs):
    pytest.fail("Existing statistics must not scan training parquet")


def test_reuse_without_scanning_or_overwrite_flow2act(tmp_path, training_windows_flow2act, monkeypatch):
    windows, _ = training_windows_flow2act
    output = tmp_path / "stats"
    content = stats_flow2act.prepare_stats_flow2act(windows, output, if_missing=True)
    path = output / "norm_stats.json"
    original = path.stat()
    monkeypatch.setattr(stats_flow2act, "parquet_flow2act", fail_scan_flow2act)
    assert stats_flow2act.prepare_stats_flow2act(windows, output, if_missing=True) == content
    assert path.stat().st_ino == original.st_ino
    assert path.stat().st_mtime_ns == original.st_mtime_ns
    with pytest.raises(FileExistsError, match="overwrite"):
        stats_flow2act.prepare_stats_flow2act(windows, output)


@pytest.mark.parametrize("corruption", ["horizon", "shape", "finite"])
def test_reject_mismatched_stats_flow2act(tmp_path, training_windows_flow2act, monkeypatch, corruption):
    windows, _ = training_windows_flow2act
    output = tmp_path / "stats"
    content = json.loads(stats_flow2act.prepare_stats_flow2act(windows, output, if_missing=True))
    if corruption == "horizon":
        content["sampling"]["action_horizon"] = 29
    elif corruption == "shape":
        content["norm_stats"]["actions"]["mean"] = [0.0]
    else:
        content["norm_stats"]["state"]["q01"][0] = float("nan")
    path = output / "norm_stats.json"
    path.write_text(json.dumps(content))
    before = path.read_bytes()
    monkeypatch.setattr(stats_flow2act, "parquet_flow2act", fail_scan_flow2act)
    with pytest.raises(ValueError, match="[Nn]orm stats"):
        stats_flow2act.prepare_stats_flow2act(windows, output, if_missing=True)
    assert path.read_bytes() == before


def test_concurrent_launches_compute_once_flow2act(tmp_path, training_windows_flow2act, monkeypatch):
    windows, _ = training_windows_flow2act
    compute = stats_flow2act.compute_stats_flow2act
    calls = []

    def counted(index):
        calls.append(index)
        time.sleep(0.05)
        return compute(index)

    monkeypatch.setattr(stats_flow2act, "compute_stats_flow2act", counted)
    output = tmp_path / "shared cache"
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(stats_flow2act.prepare_stats_flow2act, windows, output, if_missing=True) for _ in range(2)]
        results = [job.result(timeout=20) for job in jobs]
    assert len(calls) == 1
    assert results[0] == results[1] == (output / "norm_stats.json").read_text()
    assert [path.name for path in output.iterdir()] == ["norm_stats.json"]
