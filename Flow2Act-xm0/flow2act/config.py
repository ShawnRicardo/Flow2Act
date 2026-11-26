"""Scientific configuration for independent static, 2D and 3D ablations."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Flow2ActConfig:
    stage: str = "static"
    action_horizon: int = 10
    future_horizon: int = 10
    execute_horizon: int = 10
    memory_length: int = 1
    vlm_hidden_dim: int = 2560
    state_dim: int = 32
    student_taps: tuple[int, ...] = (17, 23, 29, 35)
    teacher_taps: tuple[int, ...] = (19, 27, 33, 39)
    teacher_dim: int = 3072
    adapter_hidden_dim: int = 1024
    width: int = 512
    readout_hidden_dim: int = 256
    motion_grid_size: int = 32
    image_size: int = 256
    geo_loss_weight: float = 0.02
    motion_2d_loss_weight: float = 1.0
    motion_3d_loss_weight: float = 0.1
    motion_2d_beta: float = 1.0 / 256.0
    motion_3d_beta: float = 0.01
    moving_threshold_px: float = 1.0
    readout_horizon_chunk: int = 4
    static_alignment: str = "teacher_bilinear_antialias_to_native_student_grid"
    static_loss: str = "cosine"
    teacher_view_context: str = "joint_main_wrist"
    query_location: str = "vlm_suffix_from_layer0"
    action_context: str = "current_prefix_and_future_query_kv"
    motion_target: str = "current_anchor_future_displacement"

    def __post_init__(self):
        if self.stage not in ("baseline", "static", "motion2d", "motion3d"):
            raise ValueError("stage must be baseline, static, motion2d or motion3d")
        for name in ("action_horizon", "future_horizon", "execute_horizon", "memory_length",
                     "vlm_hidden_dim", "state_dim", "teacher_dim", "adapter_hidden_dim", "width",
                     "readout_hidden_dim", "motion_grid_size", "image_size", "readout_horizon_chunk"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.execute_horizon <= self.action_horizon <= 50 or self.future_horizon > 50:
            raise ValueError("Require execution <= action horizon <= 50 and future horizon <= 50")
        if not self.has_motion and self.memory_length != 1:
            raise ValueError("Baseline/static stages have no temporal suffix; memory_length must be 1")
        if len(self.student_taps) != 4 or len(set(self.student_taps)) != 4 or len(self.teacher_taps) != 4:
            raise ValueError("Four distinct student taps and four teacher taps are required")
        for name in ("geo_loss_weight", "motion_2d_loss_weight", "motion_3d_loss_weight", "moving_threshold_px"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("motion_2d_beta", "motion_3d_beta"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def has_static(self):
        return self.stage != "baseline"

    @property
    def has_motion(self):
        return self.stage in ("motion2d", "motion3d")

    @property
    def has_3d(self):
        return self.stage == "motion3d"

    @property
    def motion_3d_enabled(self):
        return self.has_3d

    @property
    def history_offsets(self):
        return tuple(self.execute_horizon * j for j in range(1, self.memory_length))

    def to_dict(self):
        return asdict(self)
