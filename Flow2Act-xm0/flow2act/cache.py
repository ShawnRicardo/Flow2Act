"""Episode mmap reader for native-UV trajectories, never for raw XYZ caches."""
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np

SCHEMA_FLOW2ACT = "Flow2Act-motion-v1"
FILES_FLOW2ACT = {
    "motion_2d": "displacement_normalized_flow2act.npy",
    "visconf": "native_visconf_flow2act.npy",
    "in_bounds": "in_bounds_flow2act.npy",
    "horizon_valid": "horizon_valid_flow2act.npy",
    "motion_3d": "delta_xyz_normalized_flow2act.npy",
    "valid_3d": "valid3d_flow2act.npy",
    "anchor_scale": "anchor_scale_flow2act.npy",
}


def canonical_suite_flow2act(suite):
    aliases = {"10": "libero_10", "long": "libero_10", "goal": "libero_goal",
               "object": "libero_object", "spatial": "libero_spatial"}
    suite = aliases.get(str(suite), str(suite))
    if suite not in ("libero_10", "libero_goal", "libero_object", "libero_spatial"):
        raise ValueError(f"Unknown LIBERO suite {suite!r}")
    return suite


class MotionCacheFlow2Act:
    def __init__(self, root, horizon, motion_3d_enabled=False):
        self.root = Path(root)
        self.horizon = int(horizon)
        self.motion_3d_enabled = bool(motion_3d_enabled)
        header = self.root / "metadata_flow2act.json"
        if not header.is_file():
            raise FileNotFoundError(f"Missing new native-UV cache {header}; run export_motion_flow2act.py first")
        self.metadata = json.loads(header.read_text())
        if self.metadata.get("schema") != SCHEMA_FLOW2ACT:
            raise ValueError("Cache is not a Flow2Act native-UV cache")
        if self.metadata.get("coordinate") != "displacement_xy_div_wh":
            raise ValueError("Expected normalized current-anchor displacement, not endpoint/raw XYZ")
        if self.metadata.get("image_hw") != [256, 256] or self.metadata.get("grid_hw") != [32, 32]:
            raise ValueError("This experiment fixes full-FOV image256 and current-anchor grid32")
        if not 1 <= self.horizon <= self.metadata["horizon"]:
            raise ValueError("Cache horizon does not cover configured action/future horizon")
        if self.motion_3d_enabled and not self.metadata.get("has_motion_3d", False):
            raise FileNotFoundError("3D supervision requested but absent. Re-export with --motion-3d-enabled")
        self._episodes = OrderedDict()

    def _episode(self, suite, episode_index):
        key = (canonical_suite_flow2act(suite), int(episode_index))
        if key not in self._episodes:
            directory = self.root / key[0] / f"episode_{key[1]:06d}"
            names = ["motion_2d", "visconf", "in_bounds", "horizon_valid"]
            if self.motion_3d_enabled:
                names += ["motion_3d", "valid_3d"]
            arrays = {}
            for name in names:
                path = directory / FILES_FLOW2ACT[name]
                if not path.is_file():
                    raise FileNotFoundError(f"Missing {path}; export this episode with matching 3D option")
                arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
            flow = arrays["motion_2d"]
            if flow.ndim != 5 or flow.shape[1:] != (self.metadata["horizon"], 2, 32, 32):
                raise ValueError(f"Malformed current-anchor cache {directory}: {flow.shape}")
            count, horizon = len(flow), self.metadata["horizon"]
            expected_shapes = {"motion_2d": (count,horizon,2,32,32),
                "visconf": (count,horizon,2,32,32), "in_bounds": (count,horizon,32,32),
                "horizon_valid": (count,horizon), "motion_3d": (count,horizon,3,32,32),
                "valid_3d": (count,horizon,32,32)}
            for name, array in arrays.items():
                floating = name in ("motion_2d", "visconf", "motion_3d")
                valid_dtype = array.dtype in (np.float16,np.float32) if floating else array.dtype == np.bool_
                if array.shape != expected_shapes[name] or not valid_dtype:
                    raise ValueError(f"Malformed {name} sidecar {directory}: shape={array.shape}, dtype={array.dtype}")
            self._episodes[key] = arrays
            if len(self._episodes) > 32:
                self._episodes.popitem(last=False)
        self._episodes.move_to_end(key)
        return self._episodes[key]

    def load(self, suite, episode_index, frame_index):
        arrays = self._episode(suite, episode_index)
        frame_index = int(frame_index)
        if not 0 <= frame_index < len(arrays["motion_2d"]):
            raise IndexError("Anchor frame lies outside the cached episode")
        h = self.horizon
        flow = np.array(arrays["motion_2d"][frame_index, :h], dtype=np.float32, copy=True)
        probabilities = np.array(arrays["visconf"][frame_index, :h], dtype=np.float32, copy=True)
        valid = np.asarray(arrays["horizon_valid"][frame_index, :h], dtype=bool)[:, None, None]
        valid = valid & np.asarray(arrays["in_bounds"][frame_index, :h], dtype=bool)
        valid = valid & np.isfinite(flow).all(1) & np.isfinite(probabilities).all(1)
        valid = valid & (probabilities[:, 0] >= 0.5) & (probabilities[:, 1] >= 0.5)
        result = {"motion_2d": np.where(valid[:, None], flow, 0.0), "valid_2d": np.array(valid, copy=True)}
        if self.motion_3d_enabled:
            xyz = np.array(arrays["motion_3d"][frame_index, :h], dtype=np.float32, copy=True)
            valid3 = valid & np.asarray(arrays["valid_3d"][frame_index, :h], dtype=bool) & np.isfinite(xyz).all(1)
            result.update(motion_3d=np.where(valid3[:, None], xyz, 0.0), valid_3d=np.array(valid3, copy=True))
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_episodes"] = OrderedDict()
        return state
