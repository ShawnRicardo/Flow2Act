# ruff: noqa: N802, N999
"""Full-FOV H30/execute10 JAX rollouts in isolated LIBERO simulator runtimes."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr
from contextlib import redirect_stdout
import io
import json
import logging
import math
from pathlib import Path
import time

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from examples.libero.libero_plus_env_recovery import create_initialized_plus_environment
from openpi.flow2act.client_flow2act import Flow2ActActionChunkClient

ROOT = Path(__file__).resolve().parents[2]
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
MAX_STEPS = {"libero_10": 520, "libero_goal": 300, "libero_object": 280, "libero_spatial": 220}


def model_observation_flow2act(observation, prompt):
    quat = np.array(observation["robot0_eef_quat"], np.float64, copy=True)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] ** 2)
    rotation = np.zeros(3) if math.isclose(denominator, 0.0) else quat[:3] * (2 * math.acos(quat[3]) / denominator)
    return {
        "observation/image": np.ascontiguousarray(observation["agentview_image"][::-1, ::-1]),
        "observation/wrist_image": np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1]),
        "observation/state": np.concatenate(
            (observation["robot0_eef_pos"], rotation, observation["robot0_gripper_qpos"])
        ).astype(np.float32),
        "prompt": str(prompt),
    }


def rollout_flow2act(args, transport, metadata, environment, observation, prompt, suite_index, task_id, episode):
    client = Flow2ActActionChunkClient(
        transport, action_horizon=metadata["action_horizon"], execute_horizon=metadata["execute_horizon"]
    )
    for _ in range(5 if args.family == "libero_plus" else 10):
        settling = np.zeros(7) if args.family == "libero_plus" else np.array([0.0] * 6 + [-1.0])
        observation, _, _, _ = environment.step(settling)
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, suite_index, task_id, episode]))
    identity = f"{args.family}:{SUITES[suite_index]}:{task_id}:{episode}:seed{args.seed}"
    if args.smoke_steps:
        identity = "smoke:" + identity
    success = bool(environment.check_success()) if args.family == "libero_plus" else False
    maximum = 720 if args.family == "libero_plus" else MAX_STEPS[SUITES[suite_index]]
    if args.smoke_steps:
        maximum = min(maximum, args.smoke_steps)
    steps = 0
    while not success and steps < maximum:
        request = model_observation_flow2act(observation, prompt)
        if steps % metadata["execute_horizon"] == 0:
            request["noise"] = rng.standard_normal((metadata["action_horizon"], 32), dtype=np.float32)
        action = np.array(client.act(request, episode_id=identity), copy=True)
        # The checkpoint preserves native LeRobot binary grippers, not signed Panda controls.
        action[6] = -1.0 if action[6] > 0.5 else 1.0
        observation, _, done, _ = environment.step(action.tolist())
        steps += 1
        success = bool(done or (args.family == "libero_plus" and environment.check_success()))
    return bool(success), steps


def main_flow2act():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("libero", "libero_plus"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--episode-limit", type=int, default=0)
    parser.add_argument("--classification", type=Path)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard")
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    np.random.seed(args.seed)
    transport = WebsocketClientPolicy("127.0.0.1", args.port, ping_interval=None, ping_timeout=None)
    metadata = transport.get_server_metadata()
    if (
        metadata.get("model") != "pi05_JAX_flow2act"
        or metadata.get("weights") != "ema"
        or not metadata.get("accepts_flow_noise")
    ):
        raise ValueError(f"Unexpected policy metadata: {metadata}")
    logging.info("POLICY_METADATA %s", json.dumps(metadata))
    classification = {}
    if args.family == "libero_plus":
        source = (
            args.classification or ROOT / "third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"
        )
        classification = json.loads(source.read_text())
    ordinal = completed = 0
    for suite_index, name in enumerate(SUITES):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            suite = benchmark.get_benchmark_dict()[name](task_order_index=0)
        directory = args.output_dir / args.family / f"shard-{args.shard_index}" / name
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o777)
        records = directory / "episodes.jsonl"
        if records.exists():
            raise FileExistsError(records)
        categories = {r["name"]: r["category"] for r in classification.get(name, [])}
        with records.open("x", buffering=1) as stream:
            records.chmod(0o777)
            for task_id in range(suite.n_tasks):
                trials = range(1 if args.family == "libero_plus" else 50)
                assigned = []
                for episode in trials:
                    if ordinal % args.num_shards == args.shard_index:
                        assigned.append(episode)
                    ordinal += 1
                if not assigned:
                    continue
                task = suite.get_task(task_id)
                states = suite.get_task_init_states(task_id)
                bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
                environment = None
                try:
                    if args.family == "libero":
                        environment = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
                        environment.seed(args.seed)
                    for episode in assigned:
                        started = time.monotonic()
                        row = {
                            "benchmark": args.family,
                            "suite": name,
                            "task_id": task_id,
                            "task_name": str(task.name),
                            "episode_idx": episode,
                            "perturbation": categories[task.name] if args.family == "libero_plus" else "clean",
                            "seed": args.seed,
                            "shard_index": args.shard_index,
                            "status": "completed",
                            "smoke_only": bool(args.smoke_steps),
                        }
                        try:
                            if args.family == "libero_plus":
                                environment, observation = create_initialized_plus_environment(
                                    bddl_path=bddl,
                                    init_state=np.asarray(states[0]),
                                    seed=args.seed,
                                    task_id=task_id,
                                    env_resolution=256,
                                )
                            else:
                                environment.reset()
                                observation = environment.set_init_state(states[episode])
                            success, steps = rollout_flow2act(
                                args,
                                transport,
                                metadata,
                                environment,
                                observation,
                                task.language,
                                suite_index,
                                task_id,
                                episode,
                            )
                            row.update(success=success, control_steps=steps)
                        except Exception as error:
                            row.update(status="error", success=None, error=f"{type(error).__name__}: {error}")
                            logging.exception("Technical rollout error")
                            raise
                        finally:
                            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
                            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                            stream.flush()
                            if args.family == "libero_plus" and environment is not None:
                                environment.close()
                                environment = None
                        logging.info("EPISODE %s", json.dumps(row, ensure_ascii=False))
                        completed += 1
                        if args.episode_limit and completed >= args.episode_limit:
                            return
                finally:
                    if environment is not None:
                        environment.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main_flow2act()
