"""Ordinary Lightning client state for the complete XR0 method configuration."""
import inspect
import json

CHECKPOINT_KEY_FLOW2ACT = "flow2act"
SCHEMA_FLOW2ACT = "Flow2Act-XR0-checkpoint-v1"


def json_config_flow2act(value):
    return json.loads(json.dumps(value, allow_nan=False))


def resolved_model_config_flow2act(config, constructor):
    result = {name:parameter.default for name,parameter in inspect.signature(constructor).parameters.items()
              if name != "self" and parameter.default is not inspect.Parameter.empty}
    result.update(dict(config))
    return json_config_flow2act(result)


def build_checkpoint_contract_flow2act(*, model_config, method_config, vlm_config,
                                            experiment_config, action_stats,kind="trained_policy",batch_contract=None,
                                            optimizer_group_contract=None):
    if kind not in ("trained_policy", "mechanical_training_smoke"):
        raise ValueError("Unknown checkpoint purpose")
    if not experiment_config:
        raise ValueError("The complete resolved experiment config is required in checkpoint client state")
    if not isinstance(action_stats,dict) or not {"mean","std","gripper_encoding","action_encoding"}<=action_stats.keys():
        raise ValueError("Actual signed-gripper normalization statistics are required in checkpoint client state")
    return json_config_flow2act(dict(schema=SCHEMA_FLOW2ACT,kind=kind,
        model_config=model_config,method_config=method_config,vlm_config=vlm_config,
        experiment_config=experiment_config,action_stats=action_stats,
        batch_contract=batch_contract if batch_contract is not None else {"enabled":False},
        optimizer_group_contract=optimizer_group_contract))


def read_checkpoint_contract_flow2act(payload, allowed_kinds=None):
    value = payload.get(CHECKPOINT_KEY_FLOW2ACT)
    nested = (payload.get("client_state") or {}).get(CHECKPOINT_KEY_FLOW2ACT)
    if value is None:
        value = nested
    elif nested is not None and value != nested:
        raise ValueError("Conflicting checkpoint client-state metadata")
    if not isinstance(value,dict) or value.get("schema") != SCHEMA_FLOW2ACT:
        raise ValueError("Missing Flow2Act-XR0-checkpoint-v1 metadata; import external artifacts with scripts/import_flow2act_artifacts.py before loading")
    for field in ("model_config","method_config","vlm_config","experiment_config","action_stats"):
        if not isinstance(value.get(field),dict):
            raise ValueError(f"Checkpoint metadata is missing {field}")
    if allowed_kinds is not None and value.get("kind") not in allowed_kinds:
        raise ValueError("Checkpoint is not an allowed production policy; mechanical smoke requires an explicit option")
    result = json_config_flow2act(value)
    return result


def _architecture(value):
    if isinstance(value,dict):
        return {key:_architecture(item) for key,item in value.items()
                if key not in {"vlm_config_path","_name_or_path","transformers_version"}
                and not key.startswith("_attn_implementation")}
    if isinstance(value,list):
        return [_architecture(item) for item in value]
    return value


def validate_model_config_flow2act(expected, saved):
    expected,saved=map(_architecture,map(json_config_flow2act,(expected,saved)))
    if expected != saved:
        keys=sorted(key for key in set(expected)|set(saved) if expected.get(key)!=saved.get(key))
        raise ValueError(f"Checkpoint model configuration mismatch: {keys}")


def validate_checkpoint_contract_flow2act(expected, saved):
    if expected["kind"] != saved["kind"]:
        raise ValueError("Checkpoint purpose differs from this run")
    validate_model_config_flow2act(expected["model_config"],saved["model_config"])
    for field in ("method_config","vlm_config"):
        expected_field, saved_field = map(json_config_flow2act, (expected[field], saved[field]))
        if _architecture(expected_field) != _architecture(saved_field):
            raise ValueError(f"Checkpoint {field} mismatch")
    if expected["action_stats"] != saved["action_stats"]:
        raise ValueError("Checkpoint action normalization statistics mismatch")
    if expected.get("batch_contract",{"enabled":False}) != saved.get("batch_contract",{"enabled":False}):
        raise ValueError("Checkpoint exact effective-batch contract mismatch")
    from mibot.utils.optimizer_groups_flow2act import validate_optimizer_groups_flow2act
    validate_optimizer_groups_flow2act(expected.get("optimizer_group_contract"),
                                              saved.get("optimizer_group_contract"))
