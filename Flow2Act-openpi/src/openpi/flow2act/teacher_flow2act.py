# ruff: noqa: N801, N802, N999
"""Pure JAX DA3-Giant feature teacher. No Torch, Hub, optimizer or checkpoint dependency.

Architecture follows the DA3/DINOv2 implementation (Meta Platforms, Inc. and
affiliates; see third_party/Track4World for the upstream license).
Weights keep their source dtype (FP32); linear/attention compute uses FP16 on GPU.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open

from openpi.flow2act.teacher_contract_flow2act import DEFAULT_TEACHER_FLOW2ACT
from openpi.flow2act.teacher_contract_flow2act import Flow2ActTeacherConfig  # noqa: F401 -- public compatibility import
from openpi.flow2act.teacher_contract_flow2act import expected_keys_flow2act


def resize_pos_flow2act(pos, side, offset=0.1):
    """Torch bicubic align_corners=False, antialias=False, including DINO's offset.

    Torch's cubic coefficient is -0.75; jax.image.resize uses a different kernel.
    This small fixed interpolation is done once, on CPU, at weight loading.
    """
    source = math.isqrt(pos.shape[1] - 1)
    if source == side:
        return np.asarray(pos)
    coordinate = (np.arange(side, dtype=np.float64) + 0.5) * source / (side + offset) - 0.5
    base = np.floor(coordinate).astype(int)
    index = base[:, None] + np.arange(-1, 3)
    distance = np.abs(coordinate[:, None] - index)
    a = -0.75
    weights = np.where(
        distance <= 1,
        (a + 2) * distance**3 - (a + 3) * distance**2 + 1,
        np.where(distance < 2, a * distance**3 - 5 * a * distance**2 + 8 * a * distance - 4 * a, 0),
    )
    matrix = np.zeros((side, source), np.float32)
    for row in range(side):
        np.add.at(matrix[row], np.clip(index[row], 0, source - 1), weights[row].astype(np.float32))
    grid = pos[:, 1:].reshape(source, source, -1)
    value = np.einsum("ij,jkc,lk->ilc", matrix, grid, matrix, optimize=True)
    return np.concatenate((pos[:, :1], value.reshape(1, side * side, -1)), 1)


def pack_weights_flow2act(raw, cfg):
    """Strict source validation precedes packing the scanned blocks."""
    expected = expected_keys_flow2act(cfg)
    if set(raw) != set(expected):
        raise ValueError(f"DA3 key mismatch: missing={set(expected) - set(raw)}, extra={set(raw) - set(expected)}")
    for name, shape in expected.items():
        value = raw[name]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Invalid DA3 weight {name}: {value.shape}, {value.dtype}; expected {shape}, FP32 finite")
    source = {k.removeprefix("pretrained."): v for k, v in raw.items()}
    names = [k.removeprefix(f"blocks.{cfg.depth - 1}.") for k in source if k.startswith(f"blocks.{cfg.depth - 1}.")]
    blocks = {}
    for name in names:
        values = []
        for i in range(cfg.depth):
            key = f"blocks.{i}.{name}"
            if key in source:
                values.append(source[key])
            elif "_norm.weight" in name:
                values.append(np.ones(cfg.width // cfg.heads, np.float32))
            else:
                values.append(np.zeros(cfg.width // cfg.heads, np.float32))
        blocks[name] = np.stack(values)
    weights = {k: v for k, v in source.items() if not k.startswith("blocks.")}
    weights["pos_embed"] = resize_pos_flow2act(weights["pos_embed"], cfg.grid, cfg.interpolate_offset)
    weights["blocks"] = blocks
    return weights


def load_teacher_flow2act(path, cfg=DEFAULT_TEACHER_FLOW2ACT):
    with safe_open(path, framework="numpy") as handle:
        raw = {k: handle.get_tensor(k) for k in handle.keys()}  # noqa: SIM118 -- safe_open is not iterable
    return pack_weights_flow2act(raw, cfg)


def layer_norm_flow2act(x, weight, bias, eps):
    dtype = x.dtype
    value = x.astype(jnp.float32)
    mean = jnp.mean(value, -1, keepdims=True)
    variance = jnp.mean(jnp.square(value - mean), -1, keepdims=True)
    return ((value - mean) * jax.lax.rsqrt(variance + eps) * weight + bias).astype(dtype)


def rope_flow2act(x, positions, frequency=100.0):
    quarter = x.shape[-1] // 4
    inv = frequency ** (-jnp.arange(quarter, dtype=jnp.float32) / quarter)
    parts = []
    for axis, part in enumerate(jnp.split(x, 2, -1)):
        angles = (positions[..., axis, None] * inv).astype(x.dtype)
        angles = jnp.concatenate((angles, angles), -1)[:, None]
        first, second = jnp.split(part, 2, -1)
        rotated = jnp.concatenate((-second, first), -1)
        parts.append(part * jnp.cos(angles) + rotated * jnp.sin(angles))
    return jnp.concatenate(parts, -1)


def block_flow2act(x, weights, positions, use_rope, cfg, dtype):
    def dense(value, name):
        # Autocast linear quantizes operands/bias, accumulates + adds bias in FP32,
        # and rounds once to FP16 at the output.
        product = jnp.matmul(
            value.astype(dtype), weights[name + ".weight"].T.astype(dtype), preferred_element_type=jnp.float32
        )
        return (product + weights[name + ".bias"].astype(dtype).astype(jnp.float32)).astype(dtype)

    def norm(value, name, eps):
        return layer_norm_flow2act(value, weights[name + ".weight"], weights[name + ".bias"], eps)

    qkv = dense(norm(x, "norm1", 1e-6), "attn.qkv")
    q, k, v = jnp.moveaxis(qkv.reshape(*qkv.shape[:2], 3, cfg.heads, cfg.width // cfg.heads), 2, 0)
    q, k, v = (jnp.swapaxes(y, 1, 2) for y in (q, k, v))
    q, k = jax.lax.cond(
        use_rope,
        lambda pair: (
            rope_flow2act(norm(pair[0].astype(jnp.float32), "attn.q_norm", 1e-5), positions, cfg.rope_frequency),
            rope_flow2act(norm(pair[1].astype(jnp.float32), "attn.k_norm", 1e-5), positions, cfg.rope_frequency),
        ),
        lambda pair: tuple(value.astype(jnp.float32) for value in pair),
        (q, k),
    )
    # CUDA autocast keeps QK LayerNorm and RoPE in FP32, then casts Q/K for SDPA.
    q, k = q.astype(dtype), k.astype(dtype)
    # SDPA accumulates logits in FP32 even for half inputs.
    logits = jnp.einsum("bhnd,bhmd->bhnm", q, k, preferred_element_type=jnp.float32) * (q.shape[-1] ** -0.5)
    prob = jax.nn.softmax(logits, -1).astype(dtype)
    attn = jnp.einsum("bhnm,bhmd->bhnd", prob, v).swapaxes(1, 2).reshape(x.shape)
    x = x + dense(attn, "attn.proj") * weights["ls1.gamma"]
    one, two = jnp.split(dense(norm(x, "norm2", 1e-6), "mlp.w12"), 2, -1)
    return x + dense(jax.nn.silu(one.astype(jnp.float32)).astype(dtype) * two, "mlp.w3") * weights["ls2.gamma"]


def teacher_features_flow2act(weights, rgb, cfg=DEFAULT_TEACHER_FLOW2ACT, *, compute_dtype=jnp.float16):
    """Joint views: rgb [B,V,H,W,3], float [0,1], raw unaugmented full FOV."""
    b, views = rgb.shape[:2]
    if views != 2:
        raise ValueError("LIBERO DA3 teacher requires the synchronized main/wrist pair")
    side = cfg.grid * cfg.patch
    rgb = jax.image.resize(rgb.astype(jnp.float32), (b, views, side, side, 3), "linear", antialias=True)
    rgb = (rgb - jnp.array([0.485, 0.456, 0.406])) / jnp.array([0.229, 0.224, 0.225])
    patch = jax.lax.conv_general_dilated(
        rgb.reshape(b * views, side, side, 3).astype(compute_dtype),
        weights["patch_embed.proj.weight"].transpose(2, 3, 1, 0).astype(compute_dtype),
        (cfg.patch, cfg.patch),
        "VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        preferred_element_type=jnp.float32,
    )
    patch = (patch + weights["patch_embed.proj.bias"].astype(compute_dtype).astype(jnp.float32)).astype(compute_dtype)
    n = cfg.grid**2 + 1
    x = jnp.concatenate(
        (jnp.broadcast_to(weights["cls_token"], (b * views, 1, cfg.width)), patch.reshape(b * views, n - 1, cfg.width)),
        1,
    )
    x = (x + weights["pos_embed"]).reshape(b, views, n, cfg.width)
    y, z = jnp.meshgrid(jnp.arange(cfg.grid), jnp.arange(cfg.grid), indexing="ij")
    local_pos = jnp.concatenate((jnp.zeros((1, 2), jnp.int32), jnp.stack((y.flatten() + 1, z.flatten() + 1), -1)), 0)
    global_pos = jnp.concatenate((jnp.zeros((1, 2), jnp.int32), jnp.ones((n - 1, 2), jnp.int32)), 0)
    features = jnp.zeros((len(cfg.taps), b, views, n - 1, 2 * cfg.width), jnp.float32)

    def scan(carry, item):
        x, local, features = carry
        index, block = item
        cameras = jnp.concatenate(
            (
                weights["camera_token"][:, :1],
                jnp.broadcast_to(weights["camera_token"][:, 1:], (1, views - 1, cfg.width)),
            ),
            1,
        )
        x = x.at[:, :, 0].set(jnp.where(index == cfg.alt_start, cameras, x[:, :, 0]))
        is_global = (index >= cfg.alt_start) & (index % 2 == 1)

        def apply_global(value):
            pos = jnp.broadcast_to(global_pos, (b, views, n, 2)).reshape(b, views * n, 2)
            return block_flow2act(
                value.reshape(b, views * n, cfg.width), block, pos, index >= cfg.alt_start, cfg, compute_dtype
            ).reshape(value.shape)

        def apply_local(value):
            pos = jnp.broadcast_to(local_pos, (b * views, n, 2))
            return block_flow2act(
                value.reshape(b * views, n, cfg.width), block, pos, index >= cfg.alt_start, cfg, compute_dtype
            ).reshape(value.shape)

        x = jax.lax.cond(is_global, apply_global, apply_local, x)
        local = jnp.where(is_global, local, x)
        normalized = layer_norm_flow2act(x, weights["norm.weight"], weights["norm.bias"], 1e-5)
        pair = jnp.concatenate((local, normalized), -1)[:, :, 1:]
        for slot, tap in enumerate(cfg.taps):
            features = features.at[slot].set(jnp.where(index == tap, pair, features[slot]))
        return (x, local, features), None

    (_, _, features), _ = jax.lax.scan(scan, (x, x, features), (jnp.arange(cfg.depth), weights["blocks"]))
    return jax.lax.stop_gradient(features.reshape(len(cfg.taps), b, views, cfg.grid, cfg.grid, 2 * cfg.width))


def teacher_targets_flow2act(weights, rgb, cfg=DEFAULT_TEACHER_FLOW2ACT, *, compute_dtype=jnp.float16):
    raw = teacher_features_flow2act(weights, rgb, cfg, compute_dtype=compute_dtype)
    # Bilinear antialias pooling occurs before any cosine normalization.
    target = jax.image.resize(raw, (*raw.shape[:3], 16, 16, raw.shape[-1]), "linear", antialias=True)
    return jax.lax.stop_gradient(target)
