# ruff: noqa: N801, N802, N999
"""Execute a fixed decision chunk from every H30 prediction; isolate episodes."""

from collections import deque

import numpy as np


class Flow2ActActionChunkClient:
    def __init__(self, transport, *, action_horizon=30, execute_horizon=10):
        if not 1 <= execute_horizon <= action_horizon:
            raise ValueError("Invalid execution horizon")
        self.transport = transport
        self.action_horizon, self.execute_horizon = action_horizon, execute_horizon
        self.episodes, self.steps, self.plans = {}, {}, {}

    def reset(self, env_id="0"):
        self.episodes.pop(str(env_id), None)
        self.steps.pop(str(env_id), None)
        self.plans.pop(str(env_id), None)

    def act(self, observation, *, episode_id, env_id="0"):
        env = str(env_id)
        episode = str(episode_id)
        if self.episodes.get(env) != episode:
            self.reset(env)
            self.episodes[env] = episode
            self.steps[env] = 0
            self.plans[env] = deque()
        if not self.plans[env]:
            result = self.transport.infer(
                {
                    **observation,
                    "env_id": env,
                    "episode_id": episode,
                    "decision_step": self.steps[env],
                    "execute_horizon": self.execute_horizon,
                }
            )
            actions = np.asarray(result["actions"])
            if actions.shape != (self.action_horizon, 7) or result["execute_horizon"] != self.execute_horizon:
                raise ValueError("Server prediction/execution protocol differs from client")
            if not np.isfinite(actions).all():
                raise FloatingPointError("Non-finite client action plan")
            self.plans[env].extend(actions[: self.execute_horizon])
        self.steps[env] += 1
        return self.plans[env].popleft()
