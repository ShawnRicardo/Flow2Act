# ruff: noqa: N801, N802, N999
"""Offline LeRobot v2.1 LIBERO4Mix windows, deterministic sampler and native stats."""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

import av
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from PIL import ImageEnhance
import pyarrow.parquet as pq
import sentencepiece

from openpi.flow2act.cache_flow2act import read_cache_layout_flow2act
from openpi.models.model import Observation
from openpi.shared import normalize

SUITES_FLOW2ACT = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
CAMERAS_FLOW2ACT = ("observation.images.image", "observation.images.wrist_image")
IMAGE_KEYS_FLOW2ACT = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


class Flow2ActWindows:
    def __init__(self, root, horizon):
        self.root, self.horizon = Path(root), horizon
        self.episodes, self.ends, self.counts = [], [], []
        total = 0
        for suite in SUITES_FLOW2ACT:
            directory = self.root / f"{suite}_no_noops_1.0.0_lerobot"
            info = json.loads((directory / "meta/info.json").read_text())
            if info["codebase_version"] != "v2.1" or info["fps"] != 20:
                raise ValueError("Expected LIBERO LeRobot v2.1 at 20 fps")
            before = total
            for line in (directory / "meta/episodes.jsonl").read_text().splitlines():
                episode = json.loads(line)
                count = max(0, episode["length"] - horizon + 1)
                if not count:
                    continue
                self.episodes.append((suite, directory, info, episode))
                total += count
                self.ends.append(total)
            self.counts.append(total - before)
        self.ends = np.asarray(self.ends, np.int64)

    def __len__(self):
        return int(self.ends[-1])

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self.ends, index, side="right"))
        start = 0 if episode == 0 else self.ends[episode - 1]
        return self.episodes[episode], int(index - start)

    def sampling_spec(self):
        return {
            "suites": list(SUITES_FLOW2ACT),
            "action_horizon": self.horizon,
            "complete_windows": self.counts,
            "total_windows": len(self),
            "action_representation": "raw_libero_delta_gripper",
            "state_dim": 8,
            "action_dim": 7,
            "action_statistics": "all_labels_in_complete_windows",
            "state_statistics": "window_anchors",
        }


class Flow2ActSampler:
    """Permutation by epoch; only consumed samples advance the checkpointed cursor."""

    def __init__(self, size, seed, position=0):
        self.size, self.seed, self.position = size, seed, position
        self.epoch, self.order = None, None

    def take(self, count):
        result = []
        while count:
            epoch, offset = divmod(self.position, self.size)
            if self.epoch != epoch:
                self.order = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, 0x53414D])).permutation(
                    self.size
                )
                self.epoch = epoch
            n = min(count, self.size - offset)
            result.extend(self.order[offset : offset + n])
            self.position += n
            count -= n
        return np.asarray(result, np.int64)


class Flow2ActTokenizer:
    def __init__(self, path, max_len=200):
        self.processor = sentencepiece.SentencePieceProcessor(model_proto=Path(path).read_bytes())
        self.max_len = max_len

    def __call__(self, prompt):
        # Exact native PaligemmaTokenizer state=None format, using a local file.
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        tokens = self.processor.encode(cleaned, add_bos=True) + self.processor.encode("\n")
        n = min(len(tokens), self.max_len)
        return (np.pad(np.asarray(tokens[:n], np.int32), (0, self.max_len - n)), np.arange(self.max_len) < n)


class Flow2ActMotionCache:
    def __init__(self, path, horizon, three_dimensional):
        self.root, self.horizon, self.three_dimensional = Path(path), horizon, three_dimensional
        self.metadata_path, self.metadata, self.file_suffix = read_cache_layout_flow2act(self.root)
        if horizon > self.metadata["horizon"] or (three_dimensional and not self.metadata.get("has_motion_3d")):
            raise ValueError("Motion cache does not cover requested supervision")
        self.local = threading.local()

    def _episode_arrays(self, suite, episode, length):
        arrays = getattr(self.local, "arrays", None)
        if arrays is None:
            arrays = self.local.arrays = OrderedDict()
        key = (suite, episode)
        if key not in arrays:
            directory = self.root / suite / f"episode_{episode:06d}"
            names = {
                "motion2d": "displacement_normalized",
                "visconf": "native_visconf",
                "bounds": "in_bounds",
                "horizon": "horizon_valid",
            }
            if self.three_dimensional:
                names.update(motion3d="delta_xyz_normalized", valid3d="valid3d")
            value = {
                k: np.load(directory / f"{v}{self.file_suffix}.npy", mmap_mode="r", allow_pickle=False)
                for k, v in names.items()
            }
            h = self.metadata["horizon"]
            shapes = {
                "motion2d": (length, h, 2, 32, 32),
                "visconf": (length, h, 2, 32, 32),
                "bounds": (length, h, 32, 32),
                "horizon": (length, h),
                "motion3d": (length, h, 3, 32, 32),
                "valid3d": (length, h, 32, 32),
            }
            for name, data in value.items():
                floats = name in ("motion2d", "motion3d", "visconf")
                if data.shape != shapes[name] or (
                    data.dtype not in (np.float16, np.float32) if floats else data.dtype != bool
                ):
                    raise ValueError(f"Malformed motion cache {directory}/{name}: {data.shape}/{data.dtype}")
            arrays[key] = value
            if len(arrays) > 16:
                arrays.popitem(last=False)
        arrays.move_to_end(key)
        return arrays[key]

    def validate_episode(self, suite, episode, length):
        """Validate required array headers against dataset length without scanning labels."""
        self._episode_arrays(suite, episode, length)

    def load(self, suite, episode, frame, length):
        value = self._episode_arrays(suite, episode, length)
        horizon = self.horizon
        flow = np.asarray(value["motion2d"][frame, :horizon], np.float32)
        prob = np.asarray(value["visconf"][frame, :horizon], np.float32)
        # Explicit dataset boundary independent from action-window boundary.
        hv = (frame + np.arange(1, horizon + 1) < length) & value["horizon"][frame, :horizon]
        valid = hv[:, None, None] & value["bounds"][frame, :horizon]
        valid &= np.isfinite(flow).all(1) & np.isfinite(prob).all(1) & (prob >= 0.5).all(1)
        result = {"motion2d": np.where(valid[:, None], flow, 0), "mask2d": valid}
        if self.three_dimensional:
            xyz = np.asarray(value["motion3d"][frame, :horizon], np.float32)
            valid3 = valid & value["valid3d"][frame, :horizon] & np.isfinite(xyz).all(1)
            result.update(motion3d=np.where(valid3[:, None], xyz, 0), mask3d=valid3)
        return result


def parquet_flow2act(directory, info, episode):
    index = episode["episode_index"]
    path = directory / info["data_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index)
    table = pq.read_table(path, columns=["observation.state", "action", "frame_index", "episode_index"])
    result = {key: np.asarray(table[key].to_pylist()) for key in table.column_names}
    if (
        not np.array_equal(result["frame_index"], np.arange(episode["length"]))
        or not (result["episode_index"] == index).all()
    ):
        raise ValueError(f"LeRobot frame/episode alignment mismatch: {path}")
    return result


def video_frames_flow2act(path, indices, fps):
    wanted = set(indices)
    output = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "SLICE"
        stream.codec_context.thread_count = 1
        container.seek(max(0, int(min(wanted) / fps / stream.time_base)), stream=stream, backward=True)
        for frame in container.decode(stream):
            index = round(float(frame.pts * stream.time_base) * fps)
            if index in wanted:
                output[index] = frame.to_ndarray(format="rgb24")
            if index >= max(wanted):
                break
    if set(output) != wanted or any(v.shape != (256, 256, 3) for v in output.values()):
        raise ValueError(f"Missing frames or wrong full-FOV geometry: {path}, {wanted - set(output)}")
    return output


def photometric_flow2act(images, seed):
    rng = np.random.default_rng(seed)
    values = (1 + rng.uniform(-32 / 255, 32 / 255), rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5))
    flags = rng.integers(0, 2, 4) == 0
    result = []
    for image in images:
        value = Image.fromarray(image)
        for use, cls, factor in zip(
            flags[:3], (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color), values, strict=True
        ):
            if use:
                value = cls(value).enhance(factor)
        result.append(np.asarray(value))
    return np.stack(result)


def load_stats_flow2act(path, windows):
    content = json.loads((Path(path) / "norm_stats.json").read_text())
    if content.get("sampling") != windows.sampling_spec():
        raise ValueError("Norm stats do not match these complete action windows; run prepare_stats_flow2act.py")
    stats = normalize.deserialize_json(json.dumps(content))
    for key, dimension in (("state", 8), ("actions", 7)):
        for field in ("mean", "std", "q01", "q99"):
            value = getattr(stats[key], field)
            if value is None or value.shape != (dimension,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid native norm stats: {key}.{field}")
    return stats


def norm_pad_flow2act(value, stats):
    value = (value - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2 - 1
    return np.pad(value.astype(np.float32), [(0, 0)] * (value.ndim - 1) + [(0, 32 - value.shape[-1])])


class Flow2ActDataset:
    def __init__(self, cfg, windows=None):
        self.cfg = cfg
        self.windows = windows or Flow2ActWindows(cfg.data_root, cfg.action_horizon)
        self.stats = load_stats_flow2act(cfg.norm_stats, self.windows)
        self.tokenizer = Flow2ActTokenizer(cfg.tokenizer_path)
        self.cache = (
            Flow2ActMotionCache(cfg.motion_cache, cfg.future_horizon, cfg.stage == "motion3d") if cfg.motion else None
        )
        self.local = threading.local()
        self.pool = ThreadPoolExecutor(max_workers=cfg.workers) if cfg.workers else None

    def sample(self, item):
        index, serial = item
        (suite, directory, info, episode), frame = self.windows.locate(index)
        key = (suite, episode["episode_index"])
        arrays = getattr(self.local, "arrays", None)
        if arrays is None:
            arrays = self.local.arrays = OrderedDict()
        if key not in arrays:
            arrays[key] = parquet_flow2act(directory, info, episode)
            if len(arrays) > 32:
                arrays.popitem(last=False)
        arrays.move_to_end(key)
        data = arrays[key]
        history = (
            [frame - h * self.cfg.execute_horizon for h in range(1, self.cfg.memory_length)] if self.cfg.motion else []
        )
        indices = [frame, *(h for h in history if h >= 0)]
        images = []
        for camera in CAMERAS_FLOW2ACT:
            path = directory / info["video_path"].format(
                episode_chunk=key[1] // info["chunks_size"], episode_index=key[1], video_key=camera
            )
            images.append(video_frames_flow2act(path, indices, info["fps"]))
        raw = np.stack([images[v][frame] for v in range(2)])
        all_rgb = np.stack(
            [images[v][h] if h >= 0 else np.zeros_like(raw[v]) for h in [frame, *history] for v in range(2)]
        )
        if self.cfg.augment:
            all_rgb = photometric_flow2act(all_rgb, np.random.SeedSequence([self.cfg.seed, serial, 0x415547]))
        tokens, token_mask = self.tokenizer(episode["tasks"][0])
        result = {
            "rgb": all_rgb[:2],
            "state": norm_pad_flow2act(data["observation.state"][frame], self.stats["state"]),
            "actions": norm_pad_flow2act(
                data["action"][frame : frame + self.cfg.action_horizon], self.stats["actions"]
            ),
            "tokens": tokens,
            "token_mask": token_mask,
        }
        if self.cfg.stage != "baseline":
            result["teacher_rgb"] = raw.astype(np.float32) / 255
        if history:
            result["history_rgb"] = all_rgb[2:].reshape(len(history), 2, 256, 256, 3)
            result["history_valid"] = np.asarray(history) >= 0
            result["history_state"] = np.stack(
                [
                    norm_pad_flow2act(data["observation.state"][h], self.stats["state"])
                    if h >= 0
                    else np.zeros(32, np.float32)
                    for h in history
                ]
            )
        if self.cache is not None:
            result.update(self.cache.load(suite, key[1], frame, episode["length"]))
        return result

    def batch(self, indices, serial_start):
        items = list(zip(indices, range(serial_start, serial_start + len(indices)), strict=True))
        samples = list(self.pool.map(self.sample, items)) if self.pool else [self.sample(item) for item in items]
        return {k: np.stack([sample[k] for sample in samples]) for k in samples[0]}

    def close(self):
        if self.pool:
            self.pool.shutdown()


def observation_flow2act(rgb, state, tokens, token_mask, valid=None):
    b = state.shape[0]
    rgb = jax.image.resize(rgb.astype(jnp.float32) / 255, (b, 2, 224, 224, 3), "linear", antialias=True) * 2 - 1
    if valid is None:
        valid = jnp.ones(b, bool)
    return Observation(
        images={
            IMAGE_KEYS_FLOW2ACT[0]: rgb[:, 0],
            IMAGE_KEYS_FLOW2ACT[1]: rgb[:, 1],
            IMAGE_KEYS_FLOW2ACT[2]: jnp.zeros_like(rgb[:, 0]),
        },
        image_masks={
            IMAGE_KEYS_FLOW2ACT[0]: valid,
            IMAGE_KEYS_FLOW2ACT[1]: valid,
            IMAGE_KEYS_FLOW2ACT[2]: jnp.zeros(b, bool),
        },
        state=state,
        tokenized_prompt=tokens,
        tokenized_prompt_mask=token_mask,
    )


def model_batch_flow2act(raw):
    raw = {k: jnp.asarray(v) for k, v in raw.items()}
    result = {k: v for k, v in raw.items() if k in ("actions", "motion2d", "motion3d", "mask2d", "mask3d", "teacher")}
    result["observation"] = observation_flow2act(raw["rgb"], raw["state"], raw["tokens"], raw["token_mask"])
    if "history_rgb" in raw:
        b, length = raw["history_rgb"].shape[:2]
        tokens = jnp.repeat(raw["tokens"][:, None], length, 1).reshape(b * length, -1)
        mask = jnp.repeat(raw["token_mask"][:, None], length, 1).reshape(b * length, -1)
        result["history"] = observation_flow2act(
            raw["history_rgb"].reshape(b * length, 2, 256, 256, 3),
            raw["history_state"].reshape(b * length, 32),
            tokens,
            mask,
            raw["history_valid"].reshape(-1),
        )
        result["history_valid"] = raw["history_valid"]
    return result
