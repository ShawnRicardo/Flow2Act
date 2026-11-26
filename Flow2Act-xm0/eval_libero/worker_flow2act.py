"""Episode-sharded LIBERO evaluation with one durable JSONL record per rollout."""
from __future__ import annotations

import argparse
import collections
import json
import logging
import os
from pathlib import Path
import sys
import time

import imageio
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("LIBERO_CONFIG_PATH", str(Path(__file__).resolve().parent / "config_flow2act"))
from deploy.client_flow2act import ClientFlow2Act
from eval_libero.completed_flow2act import completed_episode_records
from eval_libero.observation import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    _get_libero_env,
    _quat2axisangle,
    benchmark,
    convert_to_uint8,
)
from flow2act.memory import DecisionMemoryFlow2Act


SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def assigned_episode_jobs(args, benchmark_dict):
    """Shard the fixed 40-task / 50-initial-state protocol without duplication."""
    ordinal = 0
    task_count = 0
    for suite_name in args.suites.split(","):
        suite = benchmark_dict[suite_name]()
        for task_id in range(suite.n_tasks):
            if args.task_limit and task_count >= args.task_limit:
                return
            task_count += 1
            trials = min(args.num_trials_per_task, args.episode_limit or args.num_trials_per_task)
            for episode_idx in range(trials):
                if ordinal % args.num_shards == args.shard_index:
                    yield suite_name, suite, task_id, episode_idx
                ordinal += 1


def run_episode(args, client, env, init_state, task_description, suite_name, task_id, episode_idx):
    """Preserve the existing XR0 closed-loop preprocessing and decision FIFO."""
    env.seed(args.seed)
    env.reset()
    obs = env.set_init_state(init_state)
    environment_id = f"libero:{args.shard_index}:{os.getpid()}:{suite_name}:{task_id}"
    episode_id = f"{suite_name}:{task_id}:{episode_idx}:{args.seed}"
    memory = DecisionMemoryFlow2Act(max_length=args.memory_length)
    action_plan = collections.deque()
    control_step = 0
    decisions = 0
    done = False
    replay_images = []
    max_steps = SUITE_MAX_STEPS[suite_name]
    if args.max_control_steps:
        max_steps = min(max_steps, args.max_control_steps)
    try:
        for _ in range(args.num_steps_wait):
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
        while control_step < max_steps:
            # Both cameras have the same 180-degree correction as the native evaluator.
            img = convert_to_uint8(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            wrist_img = convert_to_uint8(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))
            if args.save_videos:
                replay_images.append(img)
            if not action_plan:
                state = np.concatenate([
                    obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"], np.zeros(24),
                ])
                inputs = {
                    "task_id": "libero_all", "state": state,
                    "base": Image.fromarray(img), "wrist_left": Image.fromarray(wrist_img),
                    "language": str(task_description).capitalize() + ".",
                    "seed": args.seed + 1000 * task_id + episode_idx + 1000000 * decisions,
                    "environment_id": environment_id, "episode_id": episode_id,
                    "decision_id": decisions, "step_index": control_step,
                    "execute_horizon": args.replan_steps, "memory_length": args.memory_length,
                    "history": memory.history(environment_id, episode_id, control_step, args.memory_length),
                }
                chunk = np.asarray(client(**inputs))[0]
                if chunk.shape != (args.server_action_horizon, 32) or not np.isfinite(chunk).all():
                    raise ValueError(f"Invalid action chunk: {chunk.shape}")
                action_plan.extend(chunk[:args.replan_steps, :7])
                observation = {key: inputs[key] for key in ("base", "wrist_left", "state", "language")}
                observation.update(environment_id=environment_id, episode_id=episode_id, step_index=control_step)
                memory.commit(environment_id, episode_id, control_step, observation)
                decisions += 1
            obs, _, done, _ = env.step(action_plan.popleft().tolist())
            control_step += 1
            if done:
                break
    finally:
        memory.reset(environment_id, episode_id)
    video = None
    if args.save_videos and replay_images:
        video_path = args.worker_dir / "videos" / suite_name / f"task{task_id}_episode{episode_idx}_{'success' if done else 'failure'}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video_path, replay_images, fps=10)
        video = str(video_path)
    return {
        "success": bool(done), "control_steps": control_step,
        "decisions": decisions, "video": video,
    }


def evaluate(args):
    logging.info("Simulator MuJoCo version: %s", __import__("mujoco").__version__)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if not 1 <= args.num_trials_per_task <= 50:
        raise ValueError("The official initial-state protocol provides 50 trials per task")
    if min(args.max_control_steps, args.task_limit, args.episode_limit) < 0:
        raise ValueError("Smoke limits must be nonnegative")
    if any(name not in SUITE_MAX_STEPS for name in args.suites.split(",")):
        raise ValueError("Only the four trained LIBERO evaluation suites are supported")
    if args.completed_dir and Path(args.completed_dir).expanduser().resolve() == Path(args.output_dir).expanduser().resolve():
        raise ValueError("Continuation requires a fresh output directory distinct from completed-dir")
    args.worker_dir = Path(args.output_dir) / "libero" / f"shard-{args.shard_index}"
    args.worker_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.worker_dir / "episodes.jsonl"
    if results_path.exists() and results_path.stat().st_size:
        raise FileExistsError(f"Choose a fresh evaluation output directory: {results_path}")
    np.random.seed(args.seed)
    client = ClientFlow2Act(args.host, args.port)
    env = None
    try:
        metadata = client.policy_metadata(replan_steps=args.replan_steps, memory_length=args.memory_length)
        if metadata["checkpoint_kind"] != "trained_policy" and not args.max_control_steps:
            raise ValueError("Untrained policies require an explicit max-control-steps smoke cap")
        args.memory_length = metadata["memory_length"]
        args.replan_steps = metadata["execute_horizon"]
        args.server_action_horizon = metadata["action_horizon"]
        logging.info("Policy metadata: %s", metadata)
        completed_ids = completed_episode_records(
            args.completed_dir, "libero", seed=args.seed, metadata=metadata,
        ) if args.completed_dir else {}
        skipped = 0
        current_task = None
        with results_path.open("a", buffering=1) as result_file:
            for suite_name, suite, task_id, episode_idx in assigned_episode_jobs(args, benchmark.get_benchmark_dict()):
                if (suite_name, task_id, episode_idx) in completed_ids:
                    skipped += 1
                    continue
                row = {
                    "benchmark": "libero", "suite": suite_name, "task_id": task_id,
                    "episode_idx": episode_idx, "perturbation": "clean", "seed": args.seed,
                    "shard_index": args.shard_index, "status": "completed", "error": None,
                    "smoke_only": bool(args.max_control_steps),
                    "policy_metadata": metadata,
                }
                started = time.monotonic()
                try:
                    if current_task != (suite_name, task_id):
                        if env is not None:
                            env.close()
                        task = suite.get_task(task_id)
                        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
                        initial_states = suite.get_task_init_states(task_id)
                        current_task = (suite_name, task_id)
                    row.update(run_episode(args, client, env, initial_states[episode_idx], task_description,
                                           suite_name, task_id, episode_idx))
                except Exception as exc:
                    row.update(status="error", success=None, error=f"{type(exc).__name__}: {exc}")
                    logging.exception("Technical rollout error: %s", row)
                    raise
                finally:
                    row["elapsed_seconds"] = round(time.monotonic() - started, 3)
                    result_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    result_file.flush()
                logging.info("EPISODE %s", json.dumps(row, ensure_ascii=False))
        logging.info("CONTINUATION family=libero shard=%s/%s skipped_completed=%s", args.shard_index, args.num_shards, skipped)
    finally:
        if env is not None:
            env.close()
        client.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--completed-dir", type=Path, help="Prior output whose completed episodes are skipped")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--suites", default=",".join(SUITE_MAX_STEPS))
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--memory-length", type=int)
    parser.add_argument("--replan-steps", type=int)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--max-control-steps", type=int, default=0)
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--episode-limit", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate(parse_args())
