# ruff: noqa: N801, N802, N999
"""Independent Torch numerical oracle for DA3 primitives (verification only)."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.flow2act.teacher_flow2act import Flow2ActTeacherConfig
from openpi.flow2act.teacher_flow2act import block_flow2act
from openpi.flow2act.teacher_flow2act import expected_keys_flow2act
from openpi.flow2act.teacher_flow2act import pack_weights_flow2act
from openpi.flow2act.teacher_flow2act import resize_pos_flow2act
from openpi.flow2act.teacher_flow2act import teacher_features_flow2act


def test_bicubic_and_antialias_pooling_flow2act():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(31)
    position = rng.normal(size=(1, 1370, 12)).astype(np.float32)
    grid = torch.from_numpy(position[:, 1:].reshape(1, 37, 37, 12)).permute(0, 3, 1, 2)
    expected = torch.nn.functional.interpolate(
        grid, scale_factor=(18.1 / 37, 18.1 / 37), mode="bicubic", antialias=False
    )
    actual = resize_pos_flow2act(position, 18)[:, 1:].reshape(1, 18, 18, 12)
    np.testing.assert_allclose(actual, expected.permute(0, 2, 3, 1).numpy(), atol=2e-5, rtol=2e-4)
    feature = rng.normal(size=(1, 2, 18, 18, 12)).astype(np.float32)
    reference = torch.nn.functional.interpolate(
        torch.from_numpy(feature.reshape(2, 18, 18, 12)).permute(0, 3, 1, 2), (16, 16), mode="bilinear", antialias=True
    )
    pooled = jax.image.resize(jnp.asarray(feature), (1, 2, 16, 16, 12), "linear", antialias=True)
    np.testing.assert_allclose(
        pooled.reshape(2, 16, 16, 12), reference.permute(0, 2, 3, 1).numpy(), atol=3e-6, rtol=1e-5
    )


@pytest.mark.parametrize("precision", ["float32", "cuda_float16"])
def test_da3_block_qknorm_rope_swiglu_flow2act(precision):
    torch = pytest.importorskip("torch")
    functional = torch.nn.functional
    cuda = precision == "cuda_float16"
    if cuda and not os.environ.get("FLOW2ACT_CUDA_PRIMITIVES"):
        pytest.skip("Set FLOW2ACT_CUDA_PRIMITIVES=1 to verify CUDA autocast primitives")
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        torch.cuda.set_per_process_memory_fraction(0.025)
    cfg = Flow2ActTeacherConfig(
        width=32, heads=2, hidden=80, depth=4, alt_start=1, source_grid=3, grid=2, patch=2, taps=(0, 1, 2, 3)
    )
    rng = np.random.default_rng(2)
    raw = {
        key: (rng.normal(size=shape) * 0.15).astype(np.float32) for key, shape in expected_keys_flow2act(cfg).items()
    }
    packed = pack_weights_flow2act(raw, cfg)
    block = {key: value[1] for key, value in packed["blocks"].items()}
    weights = {key: torch.from_numpy(value).to(device) for key, value in block.items()}
    value = rng.normal(size=(2, 5, 32)).astype(np.float32)
    positions = np.tile(np.array([[0, 0], [1, 1], [1, 2], [2, 1], [2, 2]]), (2, 1, 1))

    def norm(x, name, eps):
        return functional.layer_norm(x, (x.shape[-1],), weights[name + ".weight"], weights[name + ".bias"], eps)

    def dense(x, name):
        return functional.linear(x, weights[name + ".weight"], weights[name + ".bias"])

    def rope(x):
        parts = []
        inv = 100.0 ** (-torch.arange(4, device=device) / 4)
        for axis, part in enumerate(x.chunk(2, -1)):
            angle = torch.from_numpy(positions[..., axis, None]).to(device).float() * inv
            angle = torch.cat((angle, angle), -1)[:, None]
            one, two = part.chunk(2, -1)
            parts.append(part * angle.cos() + torch.cat((-two, one), -1) * angle.sin())
        return torch.cat(parts, -1)

    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=cuda):
        x = torch.from_numpy(value).to(device)
        q, k, v = dense(norm(x, "norm1", 1e-6), "attn.qkv").reshape(2, 5, 3, 2, 16).permute(2, 0, 3, 1, 4)
        q, k = rope(norm(q, "attn.q_norm", 1e-5)), rope(norm(k, "attn.k_norm", 1e-5))
        attention = functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(2, 5, 32)
        x = x + dense(attention, "attn.proj") * weights["ls1.gamma"]
        first, second = dense(norm(x, "norm2", 1e-6), "mlp.w12").chunk(2, -1)
        expected = x + dense(functional.silu(first) * second, "mlp.w3") * weights["ls2.gamma"]
    actual = block_flow2act(
        jnp.asarray(value),
        jax.tree.map(jnp.asarray, block),
        jnp.asarray(positions),
        use_rope=True,
        cfg=cfg,
        dtype=jnp.float16 if cuda else jnp.float32,
    )
    np.testing.assert_allclose(
        actual, expected.float().cpu().numpy(), atol=2e-4 if cuda else 2e-6, rtol=2e-3 if cuda else 2e-5
    )
    if cuda:
        return
    # Frozen targets have no gradients with respect to either pixels or weights.
    rgb = jnp.asarray(rng.uniform(size=(1, 2, 4, 4, 3)).astype(np.float32))
    packed = jax.tree.map(jnp.asarray, packed)

    def function(weights, image):
        return teacher_features_flow2act(weights, image, cfg, compute_dtype=jnp.float32).sum()

    grad_weights, grad_rgb = jax.grad(function, argnums=(0, 1))(packed, rgb)
    assert all(jnp.count_nonzero(leaf) == 0 for leaf in jax.tree.leaves(grad_weights))
    assert jnp.count_nonzero(grad_rgb) == 0
    first = teacher_features_flow2act(packed, rgb, cfg, compute_dtype=jnp.float32)
    changed = teacher_features_flow2act(packed, rgb.at[:, 1].set(0), cfg, compute_dtype=jnp.float32)
    assert jnp.linalg.norm(first[-1, :, 0] - changed[-1, :, 0]) > 0
    broken = dict(raw)
    broken["pretrained.norm.weight"] = broken["pretrained.norm.weight"].astype(np.float16)
    with pytest.raises(ValueError, match="Invalid DA3 weight"):
        pack_weights_flow2act(broken, cfg)
