# ruff: noqa: N801, N802, N999
"""Scientific contracts; CPU tests do not claim a full-size GPU training gate."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.flow2act.config_flow2act import Flow2ActConfig
from openpi.flow2act.data_flow2act import Flow2ActSampler
from openpi.flow2act.data_flow2act import Flow2ActWindows
from openpi.flow2act.gemma_flow2act import Flow2ActModule
from openpi.flow2act.modules_flow2act import motion_loss_flow2act
from openpi.flow2act.modules_flow2act import static_loss_flow2act
from openpi.flow2act.training_flow2act import Flow2ActState
from openpi.flow2act.training_flow2act import restore_checkpoint_flow2act
from openpi.flow2act.training_flow2act import save_checkpoint_flow2act
from openpi.flow2act.training_flow2act import update_flow2act
from openpi.models import gemma


def test_schedule_endpoints_flow2act():
    cfg = Flow2ActConfig()
    cfg.validate()
    with pytest.raises(ValueError, match="student taps"):
        dataclasses.replace(cfg, student_taps=(-1, 8, 11, 17)).validate()
    np.testing.assert_allclose(np.array([cfg.schedule(i) for i in (0, 1999, 79999)]), [5e-7, 5e-5, 5e-6], rtol=1e-6)
    values = np.asarray(jax.vmap(cfg.schedule)(jnp.arange(80000)))
    assert np.all(np.diff(values[:2000]) >= 0)
    assert np.all(np.diff(values[1999:]) <= 0)


@pytest.mark.local_data
def test_real_window_boundaries_flow2act():
    windows = Flow2ActWindows("data/libero_lerobot", 30)
    assert windows.counts == [90478, 39630, 53818, 40442]
    assert len(windows) == 224368
    for index in (0, 184, 90477, 90478, 130107, 130108, 224367):
        (_, _, _, episode), frame = windows.locate(index)
        assert frame >= 0
        assert frame + 29 < episode["length"]
    assert windows.locate(184)[1] + 29 == windows.locate(184)[0][3]["length"] - 1
    assert windows.locate(90478)[0][0] == "libero_goal"


def test_sampler_exact_resume_flow2act():
    original = Flow2ActSampler(17, 42)
    original.take(23)
    resumed = Flow2ActSampler(17, 42, original.position)
    np.testing.assert_array_equal(original.take(45), resumed.take(45))


def test_empty_mask_balancing_and_ga_flow2act():
    target = jnp.zeros((4, 3, 2, 2, 2)).at[1, :, :, 0].set(0.02)
    prediction = jnp.ones_like(target) * 0.01
    mask = jnp.ones((4, 3, 2, 2), bool).at[0].set(False).at[3, 1:].set(False)

    def full(p):
        return motion_loss_flow2act(p, target, mask, 1 / 256, balance=True)

    fraction = jnp.mean(jnp.any(mask, (1, 2, 3)))

    def accumulated(p):
        return sum(
            motion_loss_flow2act(
                p[i : i + 2], target[i : i + 2], mask[i : i + 2], 1 / 256, balance=True, normalizer=fraction
            )
            / 2
            for i in (0, 2)
        )

    np.testing.assert_allclose(full(prediction), accumulated(prediction), rtol=1e-6)
    np.testing.assert_allclose(jax.grad(full)(prediction), jax.grad(accumulated)(prediction), rtol=1e-6)
    empty = jnp.zeros_like(mask)
    assert motion_loss_flow2act(prediction, target.at[0].set(jnp.nan), empty, 0.01) == 0
    assert jnp.isfinite(jax.grad(lambda p: motion_loss_flow2act(p, target, empty, 0.01))(prediction)).all()


def test_zero_static_vector_and_detached_target_flow2act():
    student, teacher = jnp.zeros((4, 1, 2, 2, 2, 3)), jnp.ones((4, 1, 2, 2, 2, 3))
    grad_student, grad_teacher = jax.grad(static_loss_flow2act, argnums=(0, 1))(student, teacher)
    assert static_loss_flow2act(student, teacher) == 1
    assert jnp.isfinite(grad_student).all()
    assert jnp.linalg.norm(grad_student) > 0
    assert jnp.count_nonzero(grad_teacher) == 0


def test_gemma_native_scanned_paths_and_values_flow2act():
    config = gemma.get_config("dummy")
    original = gemma.Module(configs=[config, config], embed_dtype="bfloat16", adarms=True)
    derived = Flow2ActModule(configs=[config, config], embed_dtype="bfloat16", adarms=True, taps=(0, 1, 2, 3))
    key = jax.random.key(0)
    _, weights = original.init_with_output(key, use_adarms=[False, True], method=original.init)
    _, other = derived.init_with_output(key, use_adarms=[False, True], method=derived.init)
    assert jax.tree.structure(weights) == jax.tree.structure(other)
    for first, second in zip(jax.tree.leaves(weights), jax.tree.leaves(other), strict=True):
        np.testing.assert_array_equal(first, second)
    inputs = [jax.random.normal(key, (1, 6, 64)), jax.random.normal(key, (1, 3, 64))]
    pos, mask = jnp.arange(9)[None], jnp.ones((1, 9, 9), bool)
    kwargs = {"positions": pos, "mask": mask, "adarms_cond": [None, jnp.ones((1, 64))]}
    out1 = original.apply(weights, inputs, **kwargs)
    out2 = derived.apply(weights, inputs, **kwargs)
    for first, second in zip(jax.tree.leaves(out1), jax.tree.leaves(out2), strict=True):
        np.testing.assert_array_equal(first, second)

    def objective(w, module):
        return jnp.mean(module.apply(w, inputs, **kwargs)[0][1].astype(jnp.float32) ** 2)

    grads1 = jax.grad(objective)(weights, original)
    grads2 = jax.grad(objective)(weights, derived)
    for first, second in zip(jax.tree.leaves(grads1), jax.tree.leaves(grads2), strict=True):
        np.testing.assert_array_equal(first, second)


def test_ema_checkpoint_exact_next_update_flow2act(tmp_path):
    cfg = dataclasses.replace(Flow2ActConfig(), num_gpus=1, fsdp_devices=1, global_batch_size=4)
    tx = optax.adamw(cfg.schedule, b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10)
    params = {"w": jnp.ones((2, 2))}
    state = Flow2ActState(jnp.int32(0), params, params, tx.init(params), jnp.array([0, 42], jnp.uint32), jnp.int32(0))
    grads = {"w": jnp.full((2, 2), 0.5)}
    state, _ = update_flow2act(cfg, tx, state, grads, 4)
    np.testing.assert_allclose(state.ema["w"], 0.999 + 0.001 * state.params["w"])
    norm = '{"norm_stats": {}}'
    path = save_checkpoint_flow2act(tmp_path, state, cfg, norm)
    assert path.name == "checkpoint-1"
    resumed = restore_checkpoint_flow2act(path, state, cfg, norm)
    for first, second in zip(jax.tree.leaves(state), jax.tree.leaves(resumed), strict=True):
        np.testing.assert_array_equal(first, second)
    first, _ = update_flow2act(cfg, tx, state, grads, 8)
    second, _ = update_flow2act(cfg, tx, resumed, grads, 8)
    for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="configuration"):
        restore_checkpoint_flow2act(path, state, dataclasses.replace(cfg, memory_length=2), norm)
