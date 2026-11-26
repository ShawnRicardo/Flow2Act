# Copyright (C) 2026 Xiaomi Corporation.
"""Read the shared LIBERO LeRobot v2.1 episodes without converting or copying them."""
from __future__ import annotations

import bisect
import json
import logging
import os
import random
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision.transforms.functional as vision_f
from torch.utils.data import Dataset

from mibot.utils.io import ACTION_DIM, STATE_DIM, normalize_action, resize_image, validate_stats

logger = logging.getLogger(__name__)
LIBERO_SUITES = tuple(
    f"libero_{suite}_no_noops_1.0.0_lerobot" for suite in ("10", "goal", "object", "spatial")
)
CAMERA_KEYS = ("observation.images.image", "observation.images.wrist_image")


class LiberoDatasetFlow2Act(Dataset):
    """Frame-proportional four-suite mixture with complete, episode-local action windows.

    LIBERO stores controller deltas already: xyz, axis-angle rotation, binary gripper (converted to signed once before z-score).
    Its 8D state is xyz, axis-angle orientation and two gripper joint positions.
    These values map to the first 7 / 8 dimensions of XR0, with the rest zero padded.
    """

    def __init__(self, params):
        data = params["train_datasets"]
        self.probe_only = bool(data.get("probe_only", False))
        self.stage = data.get("stage", "baseline")
        if self.stage not in ("baseline", "static", "motion2d", "motion3d"):
            raise ValueError("stage must be baseline/static/motion2d/motion3d")
        for name, default in (("action_length", 10), ("future_horizon", 10), ("memory_length", 1), ("execute_horizon", 10)):
            value = data.get(name, default)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.future_horizon = data.get("future_horizon", 10)
        self.gradient_accumulation_steps = int(params.get("gradient_accumulation_steps", 1))
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        self.action_length = int(data.get("action_length", 10))
        if self.action_length < 1:
            raise ValueError("action_length must be positive")
        self.memory_length = int(data.get("memory_length", 1))
        self.execute_horizon = int(data.get("execute_horizon", 10))
        if self.memory_length < 1:
            raise ValueError("memory_length must be a positive integer including current")
        if self.stage in ("baseline", "static") and self.memory_length != 1:
            raise ValueError("baseline/static require MEMORY_LENGTH=1")
        if self.probe_only:
            self.memory_length = 1
        if not 1 <= self.execute_horizon <= self.action_length:
            raise ValueError("execute_horizon must match actual LIBERO replan steps in [1,H_A]")
        self.history_offsets = tuple(self.execute_horizon * j for j in range(1, self.memory_length))
        self.motion_cache = None
        if not self.probe_only and self.stage in ("motion2d", "motion3d"):
            if not data.get("motion_cache_root"):
                raise ValueError("Motion stages require an existing native-UV motion cache")
            from flow2act.cache import MotionCacheFlow2Act
            self.motion_cache = MotionCacheFlow2Act(
                data["motion_cache_root"], horizon=self.future_horizon,
                motion_3d_enabled=self.stage == "motion3d")
        self.augment = bool(data.get("augment", True))
        self.video_threads = int(data.get("video_threads", 2))
        if not self.probe_only:
            stats_path = data.get("stats_path")
            if stats_path:
                with open(stats_path, "r") as file:
                    stats = json.load(file)
            else:
                stats = data
            if stats.get("action_encoding") != "libero_delta_xyz_axis_angle_gripper_signed_v1":
                raise ValueError("Use independent _flow2act signed-gripper statistics")
            if stats.get("gripper_encoding") != "signed_1_minus_2g_v1":
                raise ValueError("Source gripper 0/1 must be converted exactly once before normalization")
            self.mean, self.std = validate_stats(stats["mean"], stats["std"], self.action_length)
            if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std < 0):
                raise ValueError("LIBERO action mean/std must be finite and std nonnegative")
            if np.any(self.std[:, :7] <= 0) or np.any(self.mean[:, 7:] != 0):
                raise ValueError("LIBERO requires positive std in its 7 action dimensions and zero padding mean")
        root = Path(data["root"])
        self.episodes = []
        counts = []
        for suite in data.get("suites", LIBERO_SUITES):
            suite_root = root / suite
            with (suite_root / "meta/info.json").open() as file:
                info = json.load(file)
            if info["codebase_version"] != "v2.1":
                raise ValueError(f"Expected LeRobot v2.1 in {suite_root}")
            train_start, train_stop = map(int, info["splits"]["train"].split(":"))
            for key, shape in (("action", [7]), ("observation.state", [8])):
                if info["features"][key]["shape"] != shape:
                    raise ValueError(f"Unsupported {key} shape in {suite_root}")
            with (suite_root / "meta/tasks.jsonl").open() as file:
                tasks = {int(row["task_index"]): row["task"] for row in map(json.loads, file)}
            suite_count = 0
            with (suite_root / "meta/episodes.jsonl").open() as file:
                episodes = sorted(map(json.loads, file), key=lambda row: int(row["episode_index"]))
            for episode in episodes:
                index = int(episode["episode_index"])
                if not train_start <= index < train_stop:
                    continue
                count = int(episode["length"]) if self.probe_only else int(episode["length"]) - self.action_length + 1
                if count <= 0:
                    continue
                fields = {"episode_index": index, "episode_chunk": index // int(info["chunks_size"])}
                self.episodes.append(
                    {
                        "path": str(suite_root / info["data_path"].format(**fields)),
                        "videos": tuple(
                            str(suite_root / info["video_path"].format(video_key=key, **fields))
                            for key in CAMERA_KEYS
                        ),
                        "length": int(episode["length"]),
                        "suite": suite.removesuffix("_no_noops_1.0.0_lerobot"),
                        "episode_index": index,
                        "tasks": tasks,
                        "task": episode.get("tasks", [""])[0],
                        "task_index": next((key for key, text in tasks.items() if text == episode.get("tasks", [""])[0]), -1),
                        "fps": float(info["fps"]),
                    }
                )
                counts.append(count)
                suite_count += count
            logger.info("LIBERO %s: %d complete action windows", suite, suite_count)
        if not counts:
            raise ValueError(f"No complete LIBERO action windows found at {root}")
        self.sample_ends = np.cumsum(counts, dtype=np.int64)
        self.raw_samples = int(self.sample_ends[-1])
        # Include accumulation so max_steps optimizer updates finish within one epoch.
        # Repeat whole mixtures without favoring a suite prefix or dropping anchors.
        required = int(params.get("max_steps", 1000)) * int(data.get("batch_size", 16))
        required *= int(os.environ.get("WORLD_SIZE", 1)) * self.gradient_accumulation_steps
        repeats = max(1, (required + self.raw_samples - 1) // self.raw_samples)
        self.num_samples = self.raw_samples if self.probe_only else repeats * self.raw_samples
        logger.info("LIBERO mixture: %d raw windows, %d training samples", self.raw_samples, self.num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if index < 0 or index >= self.num_samples:
            raise IndexError(index)
        index %= self.raw_samples
        episode_id = bisect.bisect_right(self.sample_ends, index)
        frame = index - (int(self.sample_ends[episode_id - 1]) if episode_id else 0)
        episode = self.episodes[episode_id]
        arrays = (self._read_probe_episode(episode["path"]) if self.probe_only else self._read_episode(episode["path"]))
        if len(arrays["state"]) != episode["length"]:
            raise ValueError(f"Episode length disagrees with metadata: {episode['path']}")
        if int(arrays["frame_index"][frame]) != frame:
            raise ValueError(f"Noncontiguous frame_index: {episode['path']}")
        state = np.zeros((1, STATE_DIM), dtype=np.float32)
        state[0, :8] = arrays["state"][frame]
        if not self.probe_only:
            actions = arrays["action"][frame : frame + self.action_length]
            action = np.zeros((self.action_length, ACTION_DIM), dtype=np.float32)
            action[:, :7] = actions
            action_mask = np.zeros_like(action, dtype=np.int32)
            action_mask[:, :7] = 1
        raw_current = [self._video_frame(path, frame, episode["fps"]) for path in episode["videos"]]
        teacher_images = None
        if self.probe_only or self.stage != "baseline":
            teacher_images = torch.stack([
                torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(2, 0, 1)
                for image in raw_current])
        history_valid = np.asarray([frame >= offset for offset in self.history_offsets], dtype=np.bool_)
        history_state = np.zeros((len(self.history_offsets), 1, STATE_DIM), dtype=np.float32)
        groups = [raw_current]
        for j, (offset, valid) in enumerate(zip(self.history_offsets, history_valid)):
            if valid:
                history_state[j, 0, :8] = arrays["state"][frame - offset]
                groups.append([self._video_frame(path, frame - offset, episode["fps"]) for path in episode["videos"]])
            else:
                groups.append([image.copy() for image in raw_current])
        prepared = self._prepare_images([image for group in groups for image in group])
        groups = [prepared[j:j+2] for j in range(0, len(prepared), 2)]
        instruction = episode["tasks"][int(arrays["task_index"][frame])].capitalize()
        sample = {
            "messages": self._messages(instruction, groups[0]),
            "history_messages": [self._messages(instruction, group) for group in groups[1:]],
            "state": torch.from_numpy(state),
            "history_state": torch.from_numpy(history_state),
            "history_valid": torch.from_numpy(history_valid),
            "history_ages": torch.tensor(list(self.history_offsets), dtype=torch.int64),
            "suite": episode["suite"], "episode_index": episode["episode_index"], "frame_index": frame,
            "task_index": int(arrays["task_index"][frame]),
        }
        if teacher_images is not None:
            sample["teacher_images"] = teacher_images
        if not self.probe_only:
            sample["action"] = torch.from_numpy(normalize_action(action, self.mean, self.std))
            sample["action_mask"] = torch.from_numpy(action_mask)
        if self.motion_cache is not None:
            targets = self.motion_cache.load(episode["suite"], episode["episode_index"], frame)
            sample.update({key: torch.from_numpy(value) for key, value in targets.items()})
        return sample

    @staticmethod
    @lru_cache(maxsize=32)
    def _read_probe_episode(path):
        table = pq.read_table(path, columns=["observation.state", "frame_index", "task_index"])
        arrays = {"state": np.asarray(table["observation.state"].to_pylist(), dtype=np.float32),
                  "frame_index": table["frame_index"].to_numpy(), "task_index": table["task_index"].to_numpy()}
        if arrays["state"].shape != (table.num_rows, 8) or not np.isfinite(arrays["state"]).all():
            raise ValueError("Invalid probe state array")
        if not np.array_equal(arrays["frame_index"], np.arange(table.num_rows)):
            raise ValueError("Probe frames must be contiguous")
        return arrays

    @staticmethod
    @lru_cache(maxsize=32)
    def _read_episode(path):
        table = pq.read_table(path, columns=["action", "observation.state", "frame_index", "task_index"])
        arrays = {
            "action": np.asarray(table["action"].to_pylist(), dtype=np.float32),
            "state": np.asarray(table["observation.state"].to_pylist(), dtype=np.float32),
            "frame_index": table["frame_index"].to_numpy(),
            "task_index": table["task_index"].to_numpy(),
        }
        if arrays["action"].shape != (table.num_rows, 7) or arrays["state"].shape != (table.num_rows, 8):
            raise ValueError(f"Unsupported LIBERO action/state shape in {path}")
        if not np.isfinite(arrays["action"]).all() or not np.isfinite(arrays["state"]).all():
            raise ValueError(f"Non-finite LIBERO action/state in {path}")
        if not np.array_equal(arrays["frame_index"], np.arange(table.num_rows)):
            raise ValueError(f"Noncontiguous frame_index in {path}")
        if not np.isin(arrays["action"][:, 6], (0.0, 1.0)).all():
            raise ValueError(f"Expected source gripper commands in {{0,1}} in {path}")
        arrays["action"][:, 6] = 1.0 - 2.0 * arrays["action"][:, 6]
        return arrays

    def _video_frame(self, path, frame_index, fps):
        # Shared LeRobot videos are AV1; PyAV/libdav1d supports them while Decord does not.
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = self.video_threads
            target_pts = int(frame_index / fps / stream.time_base)
            container.seek(target_pts, stream=stream, backward=True, any_frame=False)
            for frame in container.decode(video=0):
                decoded_index = round(float(frame.pts * stream.time_base) * fps)
                if decoded_index == frame_index:
                    # RLDS -> LeRobot preserves the training orientation. The simulator
                    # observation is rotated in eval_libero, so no second flip belongs here.
                    return frame.to_image().convert("RGB")
                if decoded_index > frame_index:
                    break
        raise IndexError(f"Video {path} has no frame {frame_index} at {fps} fps")

    def _prepare_images(self, images):
        images = [resize_image(image, factor=32, max_pixels=90000) for image in images]
        if not self.augment:
            return images
        # Full FOV preserves current-anchor trajectory coordinates.
        # Share only photometric transforms across every decision and view.
        ops = (
            (vision_f.adjust_brightness, 1.0 + random.uniform(-32.0 / 255.0, 32.0 / 255.0)),
            (vision_f.adjust_contrast, random.uniform(0.5, 1.5)),
            (vision_f.adjust_saturation, random.uniform(0.5, 1.5)),
            (vision_f.adjust_hue, 0.0),
        )
        flags = [random.randint(0, 1) == 0 for _ in ops]
        result = []
        for image in images:
            for use, (op, value) in zip(flags, ops):
                if use:
                    image = op(image, value)
            result.append(image)
        return result

    @staticmethod
    def _messages(instruction, images):
        # Same camera order and prompt as the repository's LIBERO deployment interface.
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "The following observations are captured from multiple views.\n# Base View\n"},
                    {"type": "image", "image": images[0]},
                    {"type": "text", "text": "\n# Left-Wrist View\n"},
                    {"type": "image", "image": images[1]},
                    {"type": "text", "text": f"\nGenerate robot actions for the task:\n{instruction} /no_cot"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]
