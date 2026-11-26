"""Client-owned decision history; control-loop intermediate frames are excluded."""
from collections import deque
from copy import deepcopy


class DecisionMemoryFlow2Act:
    def __init__(self, max_length=8):
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
            raise ValueError("max_length must be positive")
        self.max_length = max_length
        self._episodes = {}

    def history(self, env_id, episode_id, current_step, length):
        if isinstance(length, bool) or not isinstance(length, int) or not 1 <= length <= self.max_length:
            raise ValueError("length includes current and must fit max_length")
        if current_step < 0:
            raise ValueError("current_step must be nonnegative")
        if length == 1:
            return []
        records = self._episodes.get((env_id, episode_id), ())
        # Most recent first. Same-step retries must not become their own history.
        return deepcopy([r for r in reversed(records) if r["decision_step"] < current_step][:length - 1])

    def commit(self, env_id, episode_id, step, observation):
        if step < 0:
            raise ValueError("decision step must be nonnegative")
        key = (env_id, episode_id)
        records = self._episodes.setdefault(key, deque(maxlen=self.max_length))
        if records and step < records[-1]["decision_step"]:
            raise ValueError("Decision time went backwards; reset the episode first")
        if records and step == records[-1]["decision_step"]:
            return False
        records.append({"decision_step": int(step), "observation": deepcopy(observation)})
        return True

    def reset(self, env_id, episode_id=None):
        keys = [k for k in self._episodes if k[0] == env_id and (episode_id is None or k[1] == episode_id)]
        for key in keys:
            del self._episodes[key]


def validate_execution_flow2act(action_horizon, execute_horizon, trained_execute_horizon):
    if not 1 <= execute_horizon <= action_horizon:
        raise ValueError("Require 1 <= execute_horizon <= action_horizon")
    if execute_horizon != trained_execute_horizon:
        raise ValueError("Client execution differs from trained decision-memory cadence")
