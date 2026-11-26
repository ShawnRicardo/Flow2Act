"""Small real Orbax imports preserve EMA values, normalization and the trained memory."""

import dataclasses
import json

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from openpi.flow2act import artifacts_flow2act
from openpi.flow2act import policy_flow2act
from openpi.flow2act.config_flow2act import Flow2ActConfig
from openpi.shared import normalize


class TinyPolicyFlow2Act(nnx.Module):
    def __init__(self, cfg, rngs):
        self.PaliGemma = nnx.Dict(weight=nnx.Param(jnp.arange(6, dtype=jnp.float32).reshape(2, 3)))
        self.Flow2Act = nnx.Dict(future=nnx.Dict(queries=nnx.Param(jnp.ones((30, 3), jnp.float32))))


@pytest.fixture
def archived_checkpoint_flow2act(tmp_path, monkeypatch):
    cfg = Flow2ActConfig(memory_length=1)
    params = nnx.state(TinyPolicyFlow2Act(cfg, nnx.Rngs(0)))
    expected = nnx.eval_shape(lambda: TinyPolicyFlow2Act(cfg, nnx.Rngs(0)))
    monkeypatch.setattr(artifacts_flow2act, "parameter_shape_flow2act", lambda _: nnx.state(expected))
    original = nnx.State(params.raw_mapping)
    original["ArchiveAux"] = original.pop("Flow2Act")
    source = tmp_path / "source/checkpoint-40000"
    source.mkdir(parents=True)
    (source / "config_archive.json").write_text(json.dumps(dataclasses.asdict(cfg)))
    norm = {
        key: normalize.NormStats(mean=np.zeros(size), std=np.ones(size), q01=-np.ones(size), q99=np.ones(size))
        for key, size in (("state", 8), ("actions", 7))
    }
    content = normalize.serialize_json(norm) + "\n"
    (source / "norm_stats.json").write_text(content)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(
            source / "state",
            args=ocp.args.PyTreeSave({"ema": original, "params": original, "step": np.array(40000, np.int32)}),
        )
    return source, params, content


def test_import_native_loader_round_trip_flow2act(tmp_path, archived_checkpoint_flow2act, monkeypatch):
    source, expected, content = archived_checkpoint_flow2act
    destination = tmp_path / "policy/checkpoint-40000"
    cfg, step = artifacts_flow2act.import_checkpoint_flow2act(source, destination)
    assert step == 40000
    assert cfg.memory_length == 1  # Import never promotes an old policy to the new memory=3 default.
    assert cfg.tokenizer_path == Flow2ActConfig().tokenizer_path
    assert (destination / "norm_stats.json").read_text() == content
    assert (source / "config_archive.json").is_file()
    assert not (source / "config_flow2act.json").exists()
    monkeypatch.setattr(policy_flow2act, "Flow2ActPi05", TinyPolicyFlow2Act)
    model, loaded_cfg, _ = policy_flow2act.load_ema_flow2act(destination)
    assert loaded_cfg.memory_length == 1
    for actual, previous in zip(nnx.state(model).flat_state().items(), expected.flat_state().items(), strict=True):
        assert actual[0] == previous[0]
        np.testing.assert_array_equal(actual[1].value, previous[1].value)
    with ocp.PyTreeCheckpointer() as checkpointer:
        assert set(checkpointer.metadata(destination / "state").tree) == {"ema", "step"}
    assert (destination / "config_flow2act.json").stat().st_mode & 0o777 == 0o777
    with pytest.raises(FileExistsError):
        artifacts_flow2act.import_checkpoint_flow2act(source, destination)


def test_import_rejects_step_and_missing_configuration_flow2act(tmp_path, archived_checkpoint_flow2act):
    source, _, _ = archived_checkpoint_flow2act
    with pytest.raises(ValueError, match="saved update"):
        artifacts_flow2act.import_checkpoint_flow2act(source, tmp_path / "checkpoint-30000")
    configuration = source / "config_archive.json"
    values = json.loads(configuration.read_text())
    del values["memory_length"]
    configuration.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="scientific configuration"):
        artifacts_flow2act.import_checkpoint_flow2act(source, tmp_path / "checkpoint-40000")
