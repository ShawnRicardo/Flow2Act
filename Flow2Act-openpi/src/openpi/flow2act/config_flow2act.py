# ruff: noqa: N801, N802, N999
"""Isolated, offline JAX pi05 experiment configuration."""

import dataclasses
from pathlib import Path

import jax.numpy as jnp

from openpi.models.pi0_config import Pi0Config


@dataclasses.dataclass(frozen=True)
class Flow2ActConfig:
    stage: str = "motion3d"
    action_horizon: int = 30
    future_horizon: int = 30
    execute_horizon: int = 10
    memory_length: int = 3
    num_gpus: int = 4
    fsdp_devices: int = 4
    global_batch_size: int = 256
    ga: int = 1
    max_steps: int = 30_000
    save_interval: int = 5_000
    lr: float = 5e-5
    final_lr: float = 5e-6
    warmup_steps: int = 2_000
    warmup_lr: float = 5e-7
    ema_decay: float = 0.999
    seed: int = 42
    student_taps: tuple[int, ...] = (8, 11, 14, 17)
    static_weight: float = 0.02
    motion2d_weight: float = 1.0
    motion3d_weight: float = 0.1
    motion_beta2d: float = 1 / 256
    motion_beta3d: float = 0.01
    moving_threshold: float = 1 / 256
    readout_chunk: int = 4
    workers: int = 8
    augment: bool = True
    log_interval: int = 10
    stop_after_updates: int = 0
    data_root: str = "data/libero_lerobot"
    base_params: str = "checkpoints/openpi-assets/checkpoints/pi05_base/params"
    tokenizer_path: str = "checkpoints/big_vision/paligemma_tokenizer.model"
    teacher_path: str = "checkpoints/da3_flow2act/da3_giant_flow2act.safetensors"
    motion_cache: str = "data/libero4mix_motion3d_flow2act"
    norm_stats: str = "assets/libero4mix_H30_flow2act"
    output_dir: str = ""

    def validate(self):
        if self.stage not in ("baseline", "static", "motion2d", "motion3d"):
            raise ValueError(f"Unknown STAGE: {self.stage}")
        for name in (
            "action_horizon",
            "future_horizon",
            "execute_horizon",
            "memory_length",
            "num_gpus",
            "fsdp_devices",
            "global_batch_size",
            "ga",
            "max_steps",
            "save_interval",
            "readout_chunk",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.execute_horizon > self.action_horizon:
            raise ValueError("execute_horizon exceeds action_horizon")
        if self.num_gpus % self.fsdp_devices or self.global_batch_size % (self.num_gpus * self.ga):
            raise ValueError("FSDP must divide GPUs; GBS must divide GPUs * GA exactly")
        if not 1 < self.warmup_steps < self.max_steps:
            raise ValueError("Require 1 < warmup_steps < max_steps")
        if not 0 <= self.ema_decay < 1 or not 0 < self.final_lr <= self.lr or not 0 < self.warmup_lr <= self.lr:
            raise ValueError("Invalid EMA or learning rates")
        if (
            len(self.student_taps) != 4
            or min(self.student_taps) < 0
            or tuple(sorted(set(self.student_taps))) != self.student_taps
        ):
            raise ValueError("Require four increasing, distinct student taps")
        if self.workers < 0 or self.seed < 0 or self.stop_after_updates < 0 or self.log_interval < 1:
            raise ValueError("Invalid workers/seed/stop_after_updates/log_interval")
        if min(self.motion_beta2d, self.motion_beta3d, self.moving_threshold) <= 0:
            raise ValueError("Motion Smooth L1 beta and moving threshold must be positive")
        if min(self.static_weight, self.motion2d_weight, self.motion3d_weight) < 0:
            raise ValueError("Auxiliary weights must be nonnegative")

    @property
    def motion(self):
        return self.stage in ("motion2d", "motion3d")

    def model_config(self):
        return Pi0Config(pi05=True, action_horizon=self.action_horizon, discrete_state_input=False)

    def schedule(self, count):
        """Optax count is zero-based; scientific update numbers are one-based."""
        count = jnp.asarray(count, jnp.float32)
        warm = self.warmup_lr + (self.lr - self.warmup_lr) * count / (self.warmup_steps - 1)
        progress = jnp.clip((count + 1 - self.warmup_steps) / (self.max_steps - self.warmup_steps), 0, 1)
        cosine = self.final_lr + (self.lr - self.final_lr) * 0.5 * (1 + jnp.cos(jnp.pi * progress))
        return jnp.where(
            count < self.warmup_steps, jnp.minimum(warm, self.lr), jnp.clip(cosine, self.final_lr, self.lr)
        )

    def default_output(self):
        name = (
            f"{self.stage}_mem{self.memory_length}_HA{self.action_horizon}_HF{self.future_horizon}"
            f"_M{self.execute_horizon}_gpu{self.num_gpus}_fsdp{self.fsdp_devices}"
            f"_GBS{self.global_batch_size}_GA{self.ga}_lr{self.lr:g}-{self.final_lr:g}"
            f"_wu{self.warmup_steps}-{self.warmup_lr:g}_ema{self.ema_decay:g}"
            f"_steps{self.max_steps}_save{self.save_interval}_seed{self.seed}"
            f"_taps{'-'.join(map(str, self.student_taps))}"
            f"_w{self.static_weight:g}-{self.motion2d_weight:g}-{self.motion3d_weight:g}"
        )
        return str(Path("outputs_flow2act/pi05/libero4mix") / name)
