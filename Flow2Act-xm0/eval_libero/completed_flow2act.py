"""Strictly read completed evaluation episodes for an explicit continuation."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
PLUS_PROTOCOL = "libero_plus_first_init_state_zero_settle5_XR0_Flow2Act_v1"
CLASSIFICATION_PATH = (
    Path(os.environ.get("LIBERO_PLUS_ROOT", Path(__file__).resolve().parents[1] / "third_party/LIBERO-plus"))
    / "libero/libero/benchmark/task_classification.json"
)


def completed_episode_records(
    directory: str | Path,
    family: str,
    *,
    seed: int,
    metadata: dict[str, Any] | None = None,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Return unambiguous full-protocol records, including policy failures.

    Technical errors and smoke rollouts are never completion evidence. Source
    files remain untouched; malformed or unfinished JSONL cannot be silently
    interpreted as an episode that should be retried.
    """
    if family not in ("libero", "libero_plus"):
        raise ValueError(f"Unsupported evaluation family: {family!r}")
    if type(seed) is not int or seed < 0:
        raise ValueError("Evaluation seed must be a nonnegative integer")
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary of required policy fields")
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Completed evaluation directory does not exist: {root}")
    catalog = None
    classification_by_name = {}
    if family == "libero_plus":
        catalog = json.loads(CLASSIFICATION_PATH.read_text(encoding="utf-8"))
        if set(catalog) != set(SUITES) or any(not isinstance(catalog[name], list) or not catalog[name] for name in SUITES):
            raise ValueError(f"Invalid four-suite Plus classification: {CLASSIFICATION_PATH}")
        classification_by_name = {
            name: {item["name"]: item for item in catalog[name]} for name in SUITES
        }
    completed: dict[tuple[str, int, int], dict[str, Any]] = {}
    origins: dict[tuple[str, int, int], str] = {}
    for path in sorted((root / family).glob("shard-*/**/episodes.jsonl")):
        for number, raw in enumerate(path.read_bytes().splitlines(keepends=True), 1):
            location = f"{path}:{number}"
            if not raw.strip():
                continue
            if not raw.endswith(b"\n"):
                raise ValueError(f"Unfinished JSONL record at {location}")
            try:
                row = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError(f"Invalid JSONL record at {location}: {error}") from error
            try:
                if not isinstance(row, dict):
                    raise ValueError("Episode record must be a JSON object")
                if type(row.get("smoke_only", False)) is not bool:
                    raise ValueError("smoke_only must be boolean")
                if row.get("smoke_only", False) or row.get("status") != "completed":
                    continue
                if row.get("benchmark") != family:
                    raise ValueError(f"Expected benchmark {family!r}; found {row.get('benchmark')!r}")
                suite = row.get("suite")
                if not isinstance(suite, str) or suite not in SUITES:
                    raise ValueError(f"Unexpected suite: {suite!r}")
                task_id, episode_idx = row.get("task_id"), row.get("episode_idx")
                if type(task_id) is not int or type(episode_idx) is not int:
                    raise ValueError("task_id and episode_idx must be integer indices")
                task_count = 10 if catalog is None else len(catalog[suite])
                episode_count = 50 if family == "libero" else 1
                if not 0 <= task_id < task_count or not 0 <= episode_idx < episode_count:
                    raise ValueError(f"Out-of-range task/episode index: {task_id}/{episode_idx}")
                if type(row.get("seed")) is not int or row["seed"] != seed:
                    raise ValueError(f"Expected seed {seed}; found {row.get('seed')!r}")
                if type(row.get("success")) is not bool:
                    raise ValueError("Completed episodes must have boolean success")
                policy_metadata = row.get("policy_metadata")
                if not isinstance(policy_metadata, dict):
                    raise ValueError("Completed episodes require policy_metadata")
                for key, expected in (metadata or {}).items():
                    actual = policy_metadata.get(key)
                    if key not in policy_metadata or type(actual) is not type(expected) or actual != expected:
                        raise ValueError(f"Policy metadata mismatch for {key}: expected {expected!r}, found {actual!r}")
                if family == "libero_plus":
                    if row.get("protocol") != PLUS_PROTOCOL:
                        raise ValueError(f"Incompatible Plus evaluation protocol: {row.get('protocol')!r}")
                    classification = classification_by_name[suite]
                    task_name = row.get("task_name")
                    if not isinstance(task_name, str) or task_name not in classification:
                        raise ValueError(f"Unknown Plus task name: {task_name!r}")
                    if row.get("perturbation") != classification[task_name]["category"]:
                        raise ValueError(f"Plus task/category mismatch: {task_name!r}")
                identity = (suite, task_id, episode_idx)
                if identity in completed:
                    raise ValueError(f"Duplicate completed episode {identity}: {origins[identity]} and {location}")
                completed[identity] = row
                origins[identity] = location
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid completion evidence at {location}: {error}") from error
    return completed
