"""Validate the complete XR0 Pretrain package used for fresh Flow2Act runs."""
from pathlib import Path
import json


PRETRAIN_EMBODIMENTS_FLOW2ACT = frozenset({
    "agibot_pt_v3", "droid_pt", "midata_mix_ann_pt_v1", "molmoact_pt",
})


def pretrain_base_directory_flow2act(checkpoint_path):
    """Check accompanying Pretrain metadata without depending on the folder name."""
    path = Path(checkpoint_path).expanduser()
    message = (
        "Fresh Flow2Act training requires the complete Xiaomi-Robotics-0-Pretrain "
        "directory with its original preprocessor_config.json."
    )
    if not path.is_dir():
        raise ValueError(message)
    try:
        metadata = json.loads((path / "preprocessor_config.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError(message) from error
    actions = metadata.get("action_config") if isinstance(metadata, dict) else None
    if not isinstance(actions, dict) or set(actions) != PRETRAIN_EMBODIMENTS_FLOW2ACT:
        raise ValueError(message)
    if any(not isinstance(value, dict) or not {"mean", "std"} <= value.keys()
           for value in actions.values()):
        raise ValueError(message)
    return path
