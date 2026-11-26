# ruff: noqa: N801, N802, N803, N999
"""Real-input model contracts with small native backbones for CPU regression."""

import dataclasses

import flax.nnx as nnx
from flax.traverse_util import flatten_dict
from flax.traverse_util import unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.flow2act.client_flow2act import Flow2ActActionChunkClient
from openpi.flow2act.config_flow2act import Flow2ActConfig
from openpi.flow2act.data_flow2act import Flow2ActDataset
from openpi.flow2act.data_flow2act import model_batch_flow2act
from openpi.flow2act.data_flow2act import observation_flow2act
from openpi.flow2act.model_flow2act import Flow2ActPi05
from openpi.flow2act.policy_flow2act import Flow2ActDecisionMemory
from openpi.models import pi0
from openpi.models import siglip
from openpi.models.pi0_config import Pi0Config


@pytest.fixture
def small_backbone_flow2act(monkeypatch):
    original = siglip.Module

    def small(**kwargs):
        return original(**{**kwargs, "variant": "mu/14"})

    monkeypatch.setattr(siglip, "Module", small)
    return Pi0Config(
        pi05=True,
        discrete_state_input=False,
        action_horizon=30,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        max_token_len=8,
        dtype="float32",
    )


def real_batch_flow2act(stage="baseline", memory=1):
    cfg = dataclasses.replace(
        Flow2ActConfig(), stage=stage, memory_length=memory, workers=0, student_taps=(0, 1, 2, 3), augment=False
    )
    dataset = Flow2ActDataset(cfg)
    raw = dataset.batch(np.array([0]), 0)
    raw["tokens"], raw["token_mask"] = raw["tokens"][:, :8], raw["token_mask"][:, :8]
    batch = model_batch_flow2act(raw)
    if stage != "baseline":
        batch["teacher"] = jnp.ones((4, 1, 2, 16, 16, 3072))
    dataset.close()
    return cfg, batch


@pytest.mark.local_data
def test_native_pi05_loss_gradient_sampling_parity_flow2act(small_backbone_flow2act):
    cfg, batch = real_batch_flow2act()
    original = pi0.Pi0(small_backbone_flow2act, nnx.Rngs(123))
    derived = Flow2ActPi05(cfg, nnx.Rngs(123), base_config=small_backbone_flow2act)
    original_graph, weights = nnx.split(original)
    derived_graph, other = nnx.split(derived)
    assert jax.tree.structure(weights) == jax.tree.structure(other)
    for x, y in zip(jax.tree.leaves(weights), jax.tree.leaves(other), strict=True):
        np.testing.assert_array_equal(x, y)
    obs, actions, key = batch["observation"], batch["actions"], jax.random.key(17)

    def loss(params, graph):
        return nnx.merge(graph, params).compute_loss(key, obs, actions, train=False).mean()

    first, grad1 = jax.jit(jax.value_and_grad(lambda p: loss(p, original_graph)))(weights)
    second, grad2 = jax.jit(jax.value_and_grad(lambda p: loss(p, derived_graph)))(other)
    np.testing.assert_array_equal(first, second)
    for x, y in zip(jax.tree.leaves(grad1), jax.tree.leaves(grad2), strict=True):
        np.testing.assert_array_equal(x, y)
    sample1 = jax.jit(lambda w: nnx.merge(original_graph, w).sample_actions(key, obs, num_steps=2))(weights)
    sample2 = jax.jit(lambda w: nnx.merge(derived_graph, w).sample_actions(key, obs, num_steps=2))(other)
    np.testing.assert_array_equal(sample1, sample2)
    _, nk, tk = jax.random.split(key, 3)
    batch.update(
        flow_noise=jax.random.normal(nk, actions.shape), flow_time=jax.random.beta(tk, 1.5, 1, (1,)) * 0.999 + 0.001
    )
    loss_with_draws = jax.jit(lambda p: nnx.merge(derived_graph, p).objective(key, batch)[0])(other)
    np.testing.assert_allclose(loss_with_draws, second, atol=1e-7, rtol=1e-6)


@pytest.mark.local_data
def test_motion_attention_memory_and_gradient_routes_flow2act(small_backbone_flow2act):
    cfg, batch = real_batch_flow2act("motion3d", memory=2)
    module = Flow2ActPi05(cfg, nnx.Rngs(123), base_config=small_backbone_flow2act)
    # Native adaRMS gates are initialized at zero. A pretrained expert has nonzero
    # gates; open them explicitly in this small randomly initialized test expert.
    params = nnx.state(module)
    flat = flatten_dict(params.to_pure_dict())
    for path, value in flat.items():
        if "pre_attention_norm_1" in path and path[-1] == "bias":
            flat[path] = value.at[..., -64:].set(0.1)
    params.replace_by_pure_dict(unflatten_dict(flat))
    nnx.update(module, params)
    obs = batch["observation"]
    prefix, valid, ar, visible, current = module.extended_prefix(obs, batch["history"], batch["history_valid"])
    assert prefix.shape[1] == current + 513 + 1 + 30
    assert not valid[:, current : current + 513].any()  # first frame has no t-10 history
    assert not visible[:, current:-30].any()
    assert visible[:, -30:].all()
    mask = pi0.make_attn_mask(valid, ar)
    assert not mask[:, :current, current:].any()
    assert not mask[:, -30, -29:].any()
    assert mask[:, -1, -30:].all()
    positions = jnp.cumsum(valid, -1) - 1
    assert int(positions[0, current + 513]) == int(positions[0, current - 1]) + 1
    graph, params = nnx.split(module)
    key = jax.random.key(9)
    batch["history"] = obs
    batch["history_valid"] = jnp.ones((1, 1), bool)

    def action_only(p, history_images):
        value = dict(batch)
        value["history"] = value["history"].replace(images=history_images)
        return nnx.merge(graph, p).objective(key, value)[1]["action"]

    param_grad, history_grad = jax.jit(jax.grad(action_only, argnums=(0, 1)))(params, batch["history"].images)
    assert jnp.linalg.norm(param_grad["Flow2Act"]["future"]["queries"].value) > 0
    assert jnp.linalg.norm(param_grad["Flow2Act"]["future"]["memory_proj"]["kernel"].value) > 0
    for value in jax.tree.leaves(history_grad):
        assert jnp.count_nonzero(value) == 0

    def motion_only(p):
        value = nnx.merge(graph, p).objective(key, batch)[1]
        return value["motion2d"] + value["motion3d"]

    motion_grad = jax.jit(jax.grad(motion_only))(params)
    for value in jax.tree.leaves(motion_grad["action_out_proj"]):
        assert jnp.count_nonzero(value) == 0
    assert jnp.isfinite(jax.tree.leaves(param_grad)[0]).all()

    # Training and cached inference must expose exactly the same query KV.
    def compare_cache(p):
        instance = nnx.merge(graph, p)
        noise = jax.random.normal(key, (1, 30, 32))
        prefix, valid, ar, visible, _ = instance.extended_prefix(obs, batch["history"], batch["history_valid"])
        suffix, smask, sar, cond = instance.embed_suffix(obs, noise, jnp.ones(1))
        all_valid = jnp.concatenate((valid, smask), 1)
        joint_mask = pi0.make_attn_mask(all_valid, jnp.concatenate((ar, sar)))
        joint_mask = joint_mask.at[:, prefix.shape[1] :, : prefix.shape[1]].set(
            joint_mask[:, prefix.shape[1] :, : prefix.shape[1]] & visible[:, None]
        )
        (_, out), _ = instance.PaliGemma.llm(
            [prefix, suffix], positions=jnp.cumsum(all_valid, -1) - 1, mask=joint_mask, adarms_cond=[None, cond]
        )
        expected = noise - instance.action_out_proj(out)
        cached = instance.sample_actions(
            key, obs, num_steps=1, noise=noise, history=batch["history"], history_valid=batch["history_valid"]
        )
        return expected, cached

    expected, cached = jax.jit(compare_cache)(params)
    np.testing.assert_allclose(expected, cached, atol=2e-5, rtol=2e-5)


def test_decision_memory_episode_retry_execute_flow2act():
    memory = Flow2ActDecisionMemory(3, 10)
    value = {"state": np.ones(8)}
    assert memory.history("a", "ep1", 0) == [None, None]
    memory.commit("a", "ep1", 0, value)
    memory.commit("a", "ep1", 0, value)
    assert len(memory.records["a"]) == 1
    assert memory.history("a", "ep1", 10)[0] is not None
    assert memory.history("b", "ep1", 10) == [None, None]
    with pytest.raises(ValueError, match="cadence"):
        memory.commit("a", "ep1", 9, value)
    assert memory.history("a", "ep2", 0) == [None, None]
    memory.reset()
    assert memory.records == {}


def test_predict30_execute10_client_flow2act():
    class Flow2ActTransport:
        def __init__(self):
            self.calls = []

        def infer(self, observation):
            self.calls.append(observation)
            return {"actions": np.broadcast_to(np.arange(30)[:, None], (30, 7)), "execute_horizon": 10}

    transport = Flow2ActTransport()
    client = Flow2ActActionChunkClient(transport)
    values = [client.act({}, episode_id="one")[0] for _ in range(23)]
    assert values == [*range(10), *range(10), 0, 1, 2]
    assert [call["decision_step"] for call in transport.calls] == [0, 10, 20]
    assert client.act({}, episode_id="two")[0] == 0
    assert transport.calls[-1]["episode_id"] == "two"
    assert transport.calls[-1]["decision_step"] == 0


def test_default_memory3_forward_backward_sampling_flow2act(small_backbone_flow2act):
    cfg = dataclasses.replace(Flow2ActConfig(), student_taps=(0, 1, 2, 3), workers=0, augment=False)
    assert cfg.stage == "motion3d"
    assert cfg.memory_length == 3
    obs = observation_flow2act(
        jnp.zeros((1, 2, 256, 256, 3), jnp.uint8),
        jnp.zeros((1, 32)),
        jnp.ones((1, 8), jnp.int32),
        jnp.ones((1, 8), bool),
    )
    history = observation_flow2act(
        jnp.full((2, 2, 256, 256, 3), 127, jnp.uint8),
        jnp.ones((2, 32)),
        jnp.ones((2, 8), jnp.int32),
        jnp.ones((2, 8), bool),
    )
    batch = {
        "observation": obs,
        "actions": jnp.zeros((1, 30, 32)),
        "history": history,
        "history_valid": jnp.ones((1, 2), bool),
        "teacher": jnp.ones((4, 1, 2, 16, 16, 3072)),
        "motion2d": jnp.zeros((1, 30, 2, 32, 32)),
        "motion3d": jnp.zeros((1, 30, 3, 32, 32)),
        "mask2d": jnp.ones((1, 30, 32, 32), bool),
        "mask3d": jnp.ones((1, 30, 32, 32), bool),
    }
    module = Flow2ActPi05(cfg, nnx.Rngs(123), base_config=small_backbone_flow2act)
    params = nnx.state(module)
    flat = flatten_dict(params.to_pure_dict())
    for path, value in flat.items():
        if "pre_attention_norm_1" in path and path[-1] == "bias":
            flat[path] = value.at[..., -64:].set(0.1)
    params.replace_by_pure_dict(unflatten_dict(flat))
    nnx.update(module, params)
    prefix, valid, _, visible, current = module.extended_prefix(obs, history, batch["history_valid"])
    assert prefix.shape[1] == current + 2 * 513 + 1 + 30
    assert valid[:, current:].all()
    assert not visible[:, current:-30].any()
    graph, params = nnx.split(module)
    key = jax.random.key(9)

    def loss(parameters, images):
        sample = {**batch, "history": history.replace(images=images)}
        return nnx.merge(graph, parameters).objective(key, sample)

    (total, metrics), (grad, history_grad) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(
        params, history.images
    )
    assert jnp.isfinite(total)
    assert all(jnp.isfinite(value) for value in metrics.values())
    assert all(jnp.isfinite(value).all() for value in jax.tree.leaves(grad))
    assert jnp.linalg.norm(grad["Flow2Act"]["future"]["queries"].value) > 0
    assert jnp.linalg.norm(grad["Flow2Act"]["future"]["memory_proj"]["kernel"].value) > 0
    assert all(jnp.count_nonzero(value) == 0 for value in history_grad.values())
    sample = jax.jit(
        lambda parameters: nnx.merge(graph, parameters).sample_actions(
            key, obs, num_steps=2, history=history, history_valid=batch["history_valid"]
        )
    )(params)
    assert sample.shape == (1, 30, 32)
    assert jnp.isfinite(sample).all()
