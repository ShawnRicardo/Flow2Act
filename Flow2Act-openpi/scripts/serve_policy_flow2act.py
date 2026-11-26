# ruff: noqa: N801, N802, N999
"""Serve an explicit JAX Flow2Act checkpoint; EMA is the deployment default."""

import argparse

from openpi.flow2act.policy_flow2act import Flow2ActPolicy
from openpi.flow2act.policy_flow2act import load_ema_flow2act
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


def main_flow2act():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()
    module, cfg, stats = load_ema_flow2act(args.checkpoint, use_ema=not args.no_ema)
    policy = Flow2ActPolicy(module, cfg, stats, num_steps=args.num_steps)
    WebsocketPolicyServer(policy, host="0.0.0.0", port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main_flow2act()
