"""Compute horizon-specific normalization for the four LIBERO LeRobot suites."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pyarrow.parquet as pq

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


def compute_stats(root: Path, action_length: int) -> dict:
    sums = np.zeros((action_length, 7), dtype=np.float64)
    squares = np.zeros_like(sums)
    num_windows = 0
    suite_windows = {}
    for suite in SUITES:
        directory = root / f"{suite}_no_noops_1.0.0_lerobot"
        info = json.loads((directory / "meta/info.json").read_text())
        episodes = [json.loads(line) for line in (directory / "meta/episodes.jsonl").read_text().splitlines()]
        start, end = info["splits"]["train"].split(":")
        start, end = int(start or 0), int(end or info["total_episodes"])
        suite_windows[suite] = 0
        for episode in episodes:
            index = episode["episode_index"]
            if not start <= index < end or episode["length"] < action_length:
                continue
            path = directory / info["data_path"].format(
                episode_chunk=index // info["chunks_size"], episode_index=index,
            )
            actions = np.asarray(pq.read_table(path, columns=["action"])["action"].to_pylist(), dtype=np.float64)
            if actions.shape != (episode["length"], 7) or not np.isfinite(actions).all():
                raise ValueError(f"Invalid action array in {path}: {actions.shape}")
            if not np.isin(actions[:, 6], (0.0, 1.0)).all():
                raise ValueError(f"Expected source gripper commands in {{0,1}} in {path}")
            actions[:, 6] = 1.0 - 2.0 * actions[:, 6]
            count = len(actions) - action_length + 1
            cumulative = np.vstack([np.zeros((1, 7)), np.cumsum(actions, axis=0)])
            cumulative_squared = np.vstack([np.zeros((1, 7)), np.cumsum(actions * actions, axis=0)])
            offsets = np.arange(action_length)
            sums += cumulative[offsets + count] - cumulative[offsets]
            squares += cumulative_squared[offsets + count] - cumulative_squared[offsets]
            num_windows += count
            suite_windows[suite] += count
        print(f"{suite}: {suite_windows[suite]} complete action windows", flush=True)
    if not num_windows:
        raise ValueError("No complete LIBERO training windows found")
    mean = np.zeros((action_length, 32), dtype=np.float32)
    std = np.zeros_like(mean)
    mean[:, :7] = sums / num_windows
    std[:, :7] = np.sqrt(np.maximum(squares / num_windows - (sums / num_windows) ** 2, 0.0))
    return {
        "mean": mean.tolist(), "std": std.tolist(),
        "normalization": "zscore", "action_length": action_length,
        "action_encoding": "libero_delta_xyz_axis_angle_gripper_signed_v1",
        "gripper_encoding": "signed_1_minus_2g_v1",
        "valid_action_dims": list(range(7)), "suites": list(SUITES),
        "num_windows": num_windows, "suite_windows": suite_windows,
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=repo / "data/libero_lerobot")
    parser.add_argument("--output", type=Path, default=repo / "checkpoints/Flow2Act/libero4mix_H10_norm_stats_flow2act.json")
    parser.add_argument("--action-length", type=int, default=10)
    parser.add_argument("--if-missing", action="store_true", help="Reuse existing statistics without scanning the dataset")
    args = parser.parse_args()
    if args.action_length <= 0:
        parser.error("--action-length must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Serialize concurrent launches without leaving a separate lock file.
    directory_fd = os.open(args.output.parent, os.O_RDONLY)
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        if args.if_missing and args.output.is_file():
            print(f"Using existing action statistics: {args.output}", flush=True)
            return
        stats = compute_stats(args.data_root, args.action_length)
        # Readers must see the complete JSON even when other jobs start together.
        with tempfile.TemporaryDirectory(prefix=".flow2act_stats_", dir=args.output.parent) as temporary:
            pending = Path(temporary) / args.output.name
            pending.write_text(json.dumps(stats, indent=2) + "\n")
            pending.replace(args.output)
        print(f"Saved ({args.action_length}, 32) statistics to {args.output}", flush=True)
    finally:
        os.close(directory_fd)


if __name__ == "__main__":
    main()
