# ruff: noqa: N801, N802, N999
"""Shared NoMixer static adapter and spatial Future readout."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp


def sincos_flow2act(length, width):
    return positions_flow2act(jnp.arange(length, dtype=jnp.float32), width)


def positions_flow2act(value, width):
    pairs = (width + 1) // 2
    angles = value[..., None] * jnp.exp(-jnp.log(10_000.0) * jnp.arange(pairs) / max(pairs - 1, 1))
    return jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), -1)


def spatial_flow2act(side, width):
    one = positions_flow2act(jnp.linspace(-1, 1, side), width // 2)
    return jnp.concatenate(
        (
            jnp.broadcast_to(one[None], (side, side, width // 2)),
            jnp.broadcast_to(one[:, None], (side, side, width // 2)),
        ),
        -1,
    )


class Flow2ActStaticAdapter(nnx.Module):
    def __init__(self, width, rngs):
        self.norm = nnx.LayerNorm(width, epsilon=1e-5, rngs=rngs)
        self.in_proj = nnx.Linear(width, 1024, rngs=rngs)
        self.out_proj = nnx.Linear(1024, 3072, rngs=rngs)

    def __call__(self, x):
        return self.out_proj(jax.nn.gelu(self.in_proj(self.norm(x.astype(jnp.float32))), approximate=False))


class Flow2ActFuture(nnx.Module):
    def __init__(self, width, cfg, rngs):
        self.queries = nnx.Param(jax.random.normal(rngs.params(), (cfg.future_horizon, width)) * 0.02)
        self.state = nnx.Linear(32, width, rngs=rngs)
        self.memory_norm = nnx.LayerNorm(width, epsilon=1e-5, rngs=rngs)
        self.memory_proj = nnx.Linear(width, width, rngs=rngs)
        self.camera = nnx.Param(jax.random.normal(rngs.params(), (2, width)) * 0.02)
        self.anchor_norm = nnx.LayerNorm(width, epsilon=1e-5, rngs=rngs)
        self.anchor_proj = nnx.Linear(width, 512, rngs=rngs)
        self.query_proj = nnx.Linear(width, 512, rngs=rngs)
        self.readout_norm = nnx.LayerNorm(512, epsilon=1e-5, rngs=rngs)
        self.readout_proj = nnx.Linear(512, 256, rngs=rngs)
        self.head2d = nnx.Linear(256, 2, rngs=rngs)
        self.head3d = nnx.Linear(256, 3, rngs=rngs) if cfg.stage == "motion3d" else None
        self.chunk = cfg.readout_chunk

    def readout(self, grid, queries):
        b, n, _ = grid.shape
        side = int(n**0.5)
        anchor = jax.nn.gelu(self.anchor_proj(self.anchor_norm(grid.astype(jnp.float32))), approximate=False)
        anchor = jax.image.resize(anchor.reshape(b, side, side, 512), (b, 32, 32, 512), "linear")
        anchor = anchor + spatial_flow2act(32, 512)
        queries = self.query_proj(queries.astype(jnp.float32))
        out2, out3 = [], []
        for start in range(0, queries.shape[1], self.chunk):
            x = anchor[:, None] + queries[:, start : start + self.chunk, None, None]
            x = jax.nn.gelu(self.readout_proj(self.readout_norm(x)), approximate=False)
            out2.append(jnp.moveaxis(self.head2d(x), -1, 2))
            if self.head3d is not None:
                out3.append(jnp.moveaxis(self.head3d(x), -1, 2))
        return jnp.concatenate(out2, 1), jnp.concatenate(out3, 1) if out3 else None


def static_loss_flow2act(student, teacher):
    """Both tensors [4,B,V,16,16,3072], teacher is unconditionally detached."""
    student = student.astype(jnp.float32)
    teacher = jax.lax.stop_gradient(teacher.astype(jnp.float32))
    # Clamp before sqrt to preserve F.normalize's finite zero-vector gradient.
    student *= jax.lax.rsqrt(jnp.maximum(jnp.sum(student**2, axis=-1, keepdims=True), 1e-12))
    teacher *= jax.lax.rsqrt(jnp.maximum(jnp.sum(teacher**2, axis=-1, keepdims=True), 1e-12))
    return jnp.mean(1 - jnp.sum(student * teacher, -1))


def motion_loss_flow2act(pred, target, mask, beta, *, balance=False, threshold=1 / 256, normalizer=None):
    """Equal valid horizons per sample, then equal samples; safe for empty masks."""
    if mask.ndim == 5:
        mask = mask[:, :, 0]
    mask = jax.lax.stop_gradient(mask.astype(bool) & jnp.isfinite(target).all(2))
    target = jax.lax.stop_gradient(jnp.where(mask[:, :, None], target, 0).astype(jnp.float32))
    delta = jnp.abs(pred.astype(jnp.float32) - target)
    error = jnp.mean(jnp.where(delta < beta, 0.5 * delta**2 / beta, delta - 0.5 * beta), axis=2)

    def reduce_pixels(weights):
        denom = jnp.sum(weights, (-2, -1))
        return jnp.sum(error * weights, (-2, -1)) / jnp.maximum(denom, 1), denom > 0

    if balance:
        moving = jnp.linalg.norm(target, axis=2) >= threshold
        moving_loss, mv = reduce_pixels(mask * moving)
        static_loss, sv = reduce_pixels(mask * ~moving)
        loss = (moving_loss * mv + static_loss * sv) / jnp.maximum(mv.astype(jnp.int32) + sv, 1)
        valid = mv | sv
    else:
        loss, valid = reduce_pixels(mask)
    per_sample = jnp.sum(loss * valid, 1) / jnp.maximum(jnp.sum(valid, 1), 1)
    if normalizer is None:
        return jnp.sum(per_sample) / jnp.maximum(jnp.sum(jnp.any(valid, 1)), 1)
    # Full effective-batch valid fraction, identical across all GA microbatches.
    return jnp.mean(per_sample) / jnp.maximum(normalizer, 1e-20)
