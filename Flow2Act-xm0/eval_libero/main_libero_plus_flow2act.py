"""Full LIBERO-Plus rollouts using the trained XR0 decision-memory policy.

One rollout per perturbed task, official initial state zero, five raw zero-action
settling steps, then up to 720 control steps.  Each finished rollout is flushed
to episodes.jsonl; technical exceptions are recorded separately and re-raised.
"""
from __future__ import annotations

import argparse
import collections
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import io
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUS_ROOT = Path(os.environ.get("LIBERO_PLUS_ROOT", REPO_ROOT / "third_party/LIBERO-plus"))
# Isolate the simulator configuration from both vanilla LIBERO and the user's
# default ~/.libero. The worker's PYTHONPATH selects the Plus implementation.
os.environ.setdefault("LIBERO_CONFIG_PATH", str(Path(__file__).parent / "config_LIBEROPlus_flow2act"))
sys.path.insert(0, str(REPO_ROOT))
from deploy.client_flow2act import ClientFlow2Act
from eval_libero.completed_flow2act import completed_episode_records
from flow2act.memory import DecisionMemoryFlow2Act
from eval_libero.libero_plus_env_recovery_flow2act import create_initialized_plus_environment

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
PROTOCOL = "libero_plus_first_init_state_zero_settle5_XR0_Flow2Act_v1"
CLASSIFICATION_PATH = PLUS_ROOT / "libero/libero/benchmark/task_classification.json"


def catalog():
    classification = json.loads(CLASSIFICATION_PATH.read_text())
    return {
        "benchmark": "libero_plus",
        "protocol": PROTOCOL,
        "total": sum(len(classification[name]) for name in SUITES),
        "suites": {
            name: {"total": len(classification[name]), "perturbations": dict(collections.Counter(row["category"] for row in classification[name]))}
            for name in SUITES
        },
    }


def load_suite(name):
    # Plus prints thousands of task-order indices at construction.
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        from libero.libero import benchmark
        suite = benchmark.get_benchmark_dict()[name](task_order_index=0)
    classification = json.loads(CLASSIFICATION_PATH.read_text())[name]
    by_name = {row["name"]: row for row in classification}
    if len(by_name) != len(classification) or len(classification) != suite.get_num_tasks():
        raise ValueError(f"Plus task classification is incomplete for {name}")
    for task_id in range(suite.get_num_tasks()):
        task = suite.get_task(task_id)
        if task.name not in by_name:
            raise ValueError(f"Unclassified Plus task: {task.name}")
    return suite, by_name


def quaternion_axis_angle(quaternion):
    quat = np.asarray(quaternion, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return quat[:3] * (2.0 * math.acos(quat[3]) / denominator)


def model_observation(observation, language):
    # Preserve the existing XR0 LIBERO evaluator's camera/state/action contract.
    base = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1])
    state = np.concatenate((observation["robot0_eef_pos"], quaternion_axis_angle(observation["robot0_eef_quat"]), observation["robot0_gripper_qpos"], np.zeros(24)))
    instruction = str(language).capitalize() + "."
    return {"base": Image.fromarray(base), "wrist_left": Image.fromarray(wrist), "state": state, "language": instruction}


def append_record(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
    path.chmod(0o777)


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.chmod(0o777)
    temporary.replace(path)


def rollout(client, metadata, suite, task_id, args):
    task = suite.get_task(task_id)
    from libero.libero import get_libero_path
    initial_states = suite.get_task_init_states(task_id)
    if len(initial_states) < 1:
        raise ValueError(f"Plus task {task_id} has no official initial state")
    environment, observation = create_initialized_plus_environment(
        bddl_path=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
        init_state=np.asarray(initial_states[0]), seed=args.seed, task_id=task_id, env_resolution=256,
    )
    memory = DecisionMemoryFlow2Act(max_length=metadata.get("memory_length", 1))
    environment_id = f"plus:{args.shard_index}:{os.getpid()}:{args.current_suite}:{task_id}"
    episode_id = f"{environment_id}:seed{args.seed}:trial0"
    steps, decisions = 0, 0
    started = time.monotonic()
    try:
        for _ in range(5):
            observation, _, _, _ = environment.step(np.zeros(7, dtype=np.float64))
        if args.environment_smoke:
            return {"success": None, "episode_steps": 0, "policy_decisions": 0, "elapsed_seconds": time.monotonic() - started, "status": "environment_smoke", "smoke_only": True}
        action_plan = collections.deque()
        success = bool(environment.check_success())
        max_steps = min(args.max_episode_steps, args.max_control_steps) if args.max_control_steps else args.max_episode_steps
        while not success and steps < max_steps:
            if not action_plan:
                observation_inputs = model_observation(observation, task.language)
                decision_observation = dict(observation_inputs)
                decision_observation.update(environment_id=environment_id, episode_id=episode_id, step_index=steps)
                model_inputs = {
                    **observation_inputs,
                    "task_id": "libero_all",
                    "seed": args.seed + 1000 * task_id + SUITES.index(args.current_suite) * 10000000 + 1000000 * decisions,
                    "environment_id": environment_id, "episode_id": episode_id,
                    "decision_id": decisions, "step_index": steps,
                    "execute_horizon": metadata["execute_horizon"], "memory_length": metadata["memory_length"],
                    "history": memory.history(environment_id, episode_id, steps, metadata["memory_length"]),
                }
                actions = np.asarray(client(**model_inputs))[0]
                if actions.shape != (metadata["action_horizon"], 32) or not np.isfinite(actions).all():
                    raise ValueError(f"Invalid XR0 action chunk shape={actions.shape}")
                # XR0 server already returns denormalized LIBERO controls.
                action_plan.extend(actions[: metadata["execute_horizon"], :7])
                memory.commit(environment_id, episode_id, steps, decision_observation)
                decisions += 1
            observation, _, done, _ = environment.step(action_plan.popleft().tolist())
            steps += 1
            success = bool(done or environment.check_success())
        result = {"success": success, "episode_steps": steps, "policy_decisions": decisions, "elapsed_seconds": time.monotonic() - started, "status": "completed"}
        if args.max_control_steps:
            result.update(smoke_only=True, status="smoke", observed_success=success, success=None)
        return result
    finally:
        memory.reset(environment_id, episode_id)
        environment.close()


def evaluate_suite(args, name, client, metadata):
    suite, classification = load_suite(name)
    args.current_suite = name
    task_count = suite.get_num_tasks()
    stop = task_count if not args.task_limit else min(task_count, args.task_start + args.task_limit)
    task_ids = [task_id for task_id in range(args.task_start, stop) if task_id % args.num_shards == args.shard_index]
    assigned = len(task_ids)
    task_ids = [task_id for task_id in task_ids if (name, task_id, 0) not in args.completed_records]
    skipped = assigned - len(task_ids)
    output_dir = args.output_dir / "libero_plus" / f"shard-{args.shard_index}" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o777)
    records_path = output_dir / "episodes.jsonl"
    if records_path.exists() and records_path.stat().st_size:
        raise FileExistsError(f"Refusing to append to an earlier evaluation: {records_path}")
    completed, successes = 0, 0
    categories = {}
    def progress():
        write_json(output_dir / "progress.json", {
            "benchmark": "libero_plus", "protocol": PROTOCOL, "suite": name, "seed": args.seed,
            "num_shards": args.num_shards, "shard_index": args.shard_index,
            "requested": len(task_ids), "suite_task_count": task_count,
            "skipped_completed": skipped,
            "completed": completed, "successes": successes,
            "success_rate": successes / completed if completed else None,
            "perturbations": categories, "smoke_only": bool(args.max_control_steps or args.environment_smoke),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
    progress()
    logging.info("[LIBERO-Plus] suite=%s shard=%s/%s requested=%s/%s skipped_completed=%s", name, args.shard_index, args.num_shards, len(task_ids), task_count, skipped)
    for task_id in task_ids:
        task = suite.get_task(task_id)
        row = classification[task.name]
        record = {
            "benchmark": "libero_plus", "protocol": PROTOCOL, "suite": name,
            "task_id": task_id, "task_name": task.name, "episode_idx": 0,
            "perturbation": row["category"], "difficulty_level": row["difficulty_level"],
            "seed": args.seed, "official_init_state_index": 0,
            "settle_zero_steps": 5, "max_episode_steps": args.max_episode_steps,
            "policy_metadata": metadata, "language": task.language,
        }
        try:
            record.update(rollout(client, metadata, suite, task_id, args))
        except BaseException as error:
            record.update(status="error", success=None, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
            append_record(records_path, record)
            logging.exception("Technical error in Plus suite=%s task_id=%s", name, task_id)
            raise
        append_record(records_path, record)
        if record["status"] == "completed":
            completed += 1
            successes += int(record["success"])
            item = categories.setdefault(record["perturbation"], {"completed": 0, "successes": 0})
            item["completed"] += 1
            item["successes"] += int(record["success"])
            item["success_rate"] = item["successes"] / item["completed"]
        progress()
        logging.info("[LIBERO-Plus] suite=%s shard=%s completed=%s/%s current_SR=%s task=%s perturbation=%s success=%s", name, args.shard_index, completed, len(task_ids), f"{successes/completed:.2%}" if completed else "smoke", task_id, record["perturbation"], record["success"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all",) + SUITES, default="all")
    parser.add_argument("--suites", help="Comma-separated selection; original SUITES order still determines seeds")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--completed-dir", type=Path, help="Prior output whose completed episodes are skipped")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--max-control-steps", type=int, default=0)
    parser.add_argument("--environment-smoke", action="store_true")
    parser.add_argument("--catalog-only", action="store_true")
    args = parser.parse_args()
    if args.catalog_only:
        print(json.dumps(catalog(), indent=2, sort_keys=True))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required for rollouts")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    if min(args.seed, args.task_start, args.task_limit, args.max_control_steps) < 0 or args.max_episode_steps <= 0:
        parser.error("invalid rollout range or horizon")
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.completed_dir and args.completed_dir.expanduser().resolve() == args.output_dir:
        parser.error("Continuation requires a fresh output directory distinct from completed-dir")
    if args.completed_dir and args.environment_smoke:
        parser.error("Continuation requires current policy metadata and cannot use environment-smoke")
    logging.info("Simulator MuJoCo version: %s", __import__("mujoco").__version__)
    np.random.seed(args.seed)
    client = None if args.environment_smoke else ClientFlow2Act(args.host, args.port)
    try:
        metadata = {} if client is None else client.policy_metadata()
        if not (args.max_control_steps or args.environment_smoke) and metadata["checkpoint_kind"] != "trained_policy":
            raise ValueError("Full evaluation requires a trained policy checkpoint")
        logging.info("Policy metadata: %s", metadata)
        args.completed_records = completed_episode_records(
            args.completed_dir, "libero_plus", seed=args.seed, metadata=metadata,
        ) if args.completed_dir else {}
        selected = args.suites.split(",") if args.suites else (SUITES if args.suite == "all" else (args.suite,))
        if not selected or len(set(selected)) != len(selected) or any(name not in SUITES for name in selected):
            parser.error("--suites must contain distinct supported suites")
        for name in (name for name in SUITES if name in selected):
            evaluate_suite(args, name, client, metadata)
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    main()
