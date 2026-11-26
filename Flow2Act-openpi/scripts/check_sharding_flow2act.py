# ruff: noqa: N801, N802, N999
"""Four-CPU-device integration test of actual FSDP/GA/update/checkpoint primitives.

Use JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4.
The backbone is explicitly reduced; this is not the default four-GPU gate.
"""

import argparse
import dataclasses
from pathlib import Path
import tempfile

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from openpi.flow2act.config_flow2act import Flow2ActConfig
from openpi.flow2act.data_flow2act import Flow2ActDataset
from openpi.flow2act.data_flow2act import model_batch_flow2act
from openpi.flow2act.teacher_flow2act import Flow2ActTeacherConfig
from openpi.flow2act.teacher_flow2act import expected_keys_flow2act
from openpi.flow2act.teacher_flow2act import pack_weights_flow2act
from openpi.flow2act.teacher_flow2act import teacher_targets_flow2act
from openpi.flow2act.training_flow2act import gradient_flow2act
from openpi.flow2act.training_flow2act import initialize_flow2act
from openpi.flow2act.training_flow2act import permissions_flow2act
from openpi.flow2act.training_flow2act import save_checkpoint_flow2act
from openpi.flow2act.training_flow2act import teacher_batch_flow2act
from openpi.flow2act.training_flow2act import update_flow2act
from openpi.models import pi0
from openpi.models import siglip
from openpi.models.pi0_config import Pi0Config
from openpi.training import sharding


def check_teacher_sharding_flow2act():
    cfg = Flow2ActTeacherConfig(width=8, heads=2, hidden=24, depth=4, patch=2, alt_start=1, taps=(0, 1, 2, 3))
    rng = np.random.default_rng(73)
    source = {
        name: (rng.normal(size=shape) * 0.1).astype(np.float32) for name, shape in expected_keys_flow2act(cfg).items()
    }
    weights = pack_weights_flow2act(source, cfg)
    rgb = rng.uniform(size=(8, 2, 256, 256, 3)).astype(np.float32)
    device = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def targets(weights, images):
        return teacher_targets_flow2act(weights, images, cfg, compute_dtype=jnp.float32)

    reference_call = jax.jit(targets, in_shardings=(device, device), out_shardings=device)
    reference = np.asarray(reference_call(jax.device_put(weights, device), jax.device_put(rgb, device)))
    for fsdp in (4, 2, 1):
        mesh = sharding.make_mesh(fsdp)
        data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        output = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, sharding.DATA_AXIS))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        teacher = jax.device_put(weights, replicated)
        call = jax.jit(targets, in_shardings=(replicated, data), out_shardings=output)
        images = jax.device_put(rgb, data)
        sliced = images[:4]
        assert sliced.sharding.is_fully_replicated
        try:
            call(teacher, sliced)
        except ValueError as error:
            if "Sharding passed to pjit does not match" not in str(error):
                raise
        else:
            raise AssertionError("Expected the original eager-slice layout mismatch")
        calls = []

        def checked_call(weights, chunk, *, data=data, calls=calls, call=call):
            assert chunk.sharding.is_equivalent_to(data, chunk.ndim)
            assert chunk.shape == (4, 2, 256, 256, 3)
            calls.append(1)
            return call(weights, chunk)

        for ga in (1, 2):
            parts = []
            micro = len(rgb) // ga
            for index in range(ga):
                piece = jax.device_put(rgb[index * micro : (index + 1) * micro], data)
                target = teacher_batch_flow2act(checked_call, teacher, piece, data)
                assert target.sharding.is_equivalent_to(output, target.ndim)
                parts.append(target)
            combined = np.asarray(jnp.concatenate(parts, axis=1))
            assert combined.shape == (4, 8, 2, 16, 16, 16)
            np.testing.assert_allclose(combined, reference, atol=3e-6, rtol=3e-5)
        assert len(calls) == 4
    print("Teacher slice regression: FSDP4/2/1, GA1/2, layout, sample/view/tap order and numerical parity passed")


def main_flow2act():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-only", action="store_true")
    args = parser.parse_args()
    if len(jax.devices()) != 4 or any(device.platform != "cpu" for device in jax.devices()):
        raise ValueError("This verification requires exactly four CPU devices")
    check_teacher_sharding_flow2act()
    if args.teacher_only:
        return
    root = Path(tempfile.mkdtemp(prefix="fsdp_flow2act_"))
    factory = siglip.Module

    def small(**kwargs):
        return factory(**{**kwargs, "variant": "mu/14"})

    siglip.Module = small
    base = Pi0Config(
        pi05=True,
        discrete_state_input=False,
        action_horizon=30,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        max_token_len=8,
        dtype="float32",
    )
    original = pi0.Pi0(base, nnx.Rngs(0))
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(root / "base", args=ocp.args.PyTreeSave({"params": nnx.state(original).to_pure_dict()}))
    cfg = dataclasses.replace(
        Flow2ActConfig(),
        stage="baseline",
        global_batch_size=8,
        student_taps=(0, 1, 2, 3),
        base_params=str(root / "base"),
        workers=0,
        augment=False,
    )
    mesh = sharding.make_mesh(4)
    data_spec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state, state_spec, graph, tx = initialize_flow2act(cfg, mesh, base_config=base)
    assert any(
        len(value.sharding.device_set) == 4 and not value.sharding.is_fully_replicated
        for value in jax.tree.leaves(state.params)
    )
    dataset = Flow2ActDataset(cfg)
    raw = dataset.batch(np.arange(8), 0)
    dataset.close()
    raw["tokens"], raw["token_mask"] = raw["tokens"][:, :8], raw["token_mask"][:, :8]
    rng = jax.random.key(73)
    _, nk, tk = jax.random.split(rng, 3)
    noise = jax.random.normal(nk, raw["actions"].shape)
    times = jax.random.beta(tk, 1.5, 1, (8,)) * 0.999 + 0.001
    with sharding.set_mesh(mesh):
        grad = jax.jit(lambda p, b: gradient_flow2act(graph, p, rng, b), out_shardings=(state_spec.params, replicated))
        apply = jax.jit(lambda s, g: update_flow2act(cfg, tx, s, g, 8), out_shardings=(state_spec, replicated))
        results = []
        for ga in (1, 2):
            gradients = None
            for index in range(ga):
                start, end = index * (8 // ga), (index + 1) * (8 // ga)
                batch = model_batch_flow2act({k: jax.device_put(v[start:end], data_spec) for k, v in raw.items()})
                batch.update(
                    flow_noise=jax.device_put(noise[start:end], data_spec),
                    flow_time=jax.device_put(times[start:end], data_spec),
                )
                current, _ = grad(state.params, batch)
                current = jax.tree.map(lambda x, divisor=ga: x / divisor, current)
                gradients = current if gradients is None else jax.tree.map(lambda x, y: x + y, gradients, current)
            result, _ = apply(state, gradients)
            results.append(result)
        for first, second in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1]), strict=True):
            np.testing.assert_allclose(first, second, rtol=3e-5, atol=2e-6)
        donated_apply = jax.jit(
            lambda s, g: update_flow2act(cfg, tx, s, g, 8),
            out_shardings=(state_spec, replicated),
            donate_argnums=(0, 1),
        )
        donated_result, _ = donated_apply(state, gradients)
        for first, second in zip(jax.tree.leaves(results[1]), jax.tree.leaves(donated_result), strict=True):
            np.testing.assert_array_equal(first, second)
        norm = Path(cfg.norm_stats, "norm_stats.json").read_text()
        path = save_checkpoint_flow2act(root, results[0], cfg, norm)
        assert (path / "state").is_dir()
        assert (path / "norm_stats.json").read_text() == norm
    permissions_flow2act(root)
    print(f"4 CPU devices: actual native small backbone FSDP, GA1/GA2, EMA, checkpoint save passed; {root}")


if __name__ == "__main__":
    main_flow2act()
