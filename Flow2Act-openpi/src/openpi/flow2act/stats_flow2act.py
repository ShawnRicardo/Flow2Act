# ruff: noqa: N802, N999
"""Prepare or reuse native normalization over the exact Flow2Act training windows."""

import fcntl
import json
import logging
import os
from pathlib import Path

import numpy as np

from openpi.flow2act.data_flow2act import load_stats_flow2act
from openpi.flow2act.data_flow2act import parquet_flow2act
from openpi.shared import normalize


def compute_stats_flow2act(windows):
    state, action = normalize.RunningStats(), normalize.RunningStats()
    for _, directory, info, episode in windows.episodes:
        data = parquet_flow2act(directory, info, episode)
        count = episode["length"] - windows.horizon + 1
        state.update(data["observation.state"][:count].astype(np.float64))
        labels = np.stack([data["action"][t : t + windows.horizon] for t in range(count)]).astype(np.float64)
        action.update(labels)
    content = json.loads(
        normalize.serialize_json({"state": state.get_statistics(), "actions": action.get_statistics()})
    )
    content["sampling"] = windows.sampling_spec()
    return json.dumps(content, indent=2) + "\n"


def prepare_stats_flow2act(windows, output_dir, *, if_missing=False, resume_checkpoint=""):
    """Compute once for fresh runs; resume always uses the checkpoint's exact statistics."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    final = output / "norm_stats.json"
    # A directory lock serializes concurrent experiments without leaving a lock file.
    directory_fd = os.open(output, os.O_RDONLY)
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        saved = None
        if resume_checkpoint:
            load_stats_flow2act(resume_checkpoint, windows)
            saved = (Path(resume_checkpoint) / "norm_stats.json").read_text()
        if final.exists():
            if not if_missing:
                raise FileExistsError(f"Refusing to overwrite existing norm stats: {final}")
            load_stats_flow2act(output, windows)
            content = final.read_text()
            if saved is not None and json.loads(content) != json.loads(saved):
                raise ValueError("Resume normalization statistics mismatch")
            logging.info("Using existing normalization statistics: %s", final)
            return saved if saved is not None else content
        if any(output.iterdir()):
            raise FileExistsError(f"Refusing to write into nonempty stats directory: {output}")
        if saved is None:
            logging.info(
                "Computing normalization statistics over %d complete H%d windows (CPU, no video decoding)",
                len(windows),
                windows.horizon,
            )
            content = compute_stats_flow2act(windows)
        else:
            logging.info("Restoring normalization statistics from %s", resume_checkpoint)
            content = saved
        output.chmod(0o777)
        temporary = output / "norm_stats_flow2act.tmp"
        stream = temporary.open("x")
        try:
            with stream:
                stream.write(content)
            temporary.replace(final)
            final.chmod(0o777)
        finally:
            temporary.unlink(missing_ok=True)
        load_stats_flow2act(output, windows)
        logging.info("Saved normalization statistics: %s", final)
        return content
    finally:
        os.close(directory_fd)
