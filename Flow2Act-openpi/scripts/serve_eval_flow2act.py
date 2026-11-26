# ruff: noqa: N802, N999
"""EMA-only JAX evaluation server with a completed inference warmup."""

import argparse
import logging
from pathlib import Path
import time

import jax
import numpy as np

from openpi.flow2act.policy_flow2act import Flow2ActPolicy
from openpi.flow2act.policy_flow2act import load_ema_flow2act
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


def main_flow2act():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps", type=int, default=10)
    args = parser.parse_args()
    started = time.monotonic()
    module, cfg, stats = load_ema_flow2act(args.checkpoint)
    logging.info("EMA_RESTORED seconds=%.2f device=%s", time.monotonic() - started, jax.devices())
    policy = Flow2ActPolicy(module, cfg, stats, num_steps=args.num_steps)
    policy.rng = jax.random.key(args.seed)
    warmup = {
        "observation/image": np.zeros((256, 256, 3), np.uint8),
        "observation/wrist_image": np.zeros((256, 256, 3), np.uint8),
        "observation/state": np.zeros(8, np.float32),
        "prompt": "pick up the object",
        "episode_id": "warmup",
        "env_id": "warmup",
        "decision_step": 0,
        "execute_horizon": cfg.execute_horizon,
        "noise": np.random.default_rng(args.seed).standard_normal((cfg.action_horizon, 32), dtype=np.float32),
    }
    result = policy.infer(warmup)
    if result["actions"].shape != (cfg.action_horizon, 7):
        raise ValueError("Invalid warmup actions")
    policy.reset()
    policy.rng = jax.random.key(args.seed)
    metadata = {
        **policy.metadata,
        "checkpoint_global_step": int(args.checkpoint.name.split("-")[1]),
        "weights": "ema",
        "num_denoise_steps": args.num_steps,
        "evaluation_seed": args.seed,
        "accepts_flow_noise": True,
        "action_contract": "lerobot_binary_gripper_v1",
    }
    logging.info("SERVER_WARMUP_OK total_seconds=%.2f metadata=%s", time.monotonic() - started, metadata)
    WebsocketPolicyServer(
        policy, host="127.0.0.1", port=args.port, metadata=metadata, ping_interval=None, ping_timeout=None
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main_flow2act()
