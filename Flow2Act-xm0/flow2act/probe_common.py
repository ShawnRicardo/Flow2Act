"""Episode split helpers shared by the independent depth reconstruction route."""
from collections import defaultdict
import random


def stratified_episode_split_flow2act(episodes, *, seed=42, val_fraction=0.1):
    """Keep every episode in exactly one split, stratified by suite/task."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must lie strictly between zero and one")
    groups = defaultdict(list)
    for index, episode in enumerate(episodes):
        task = episode.get("task_index", -1)
        if task == -1:
            task = episode.get("task")
        if task is None or task == "":
            raise ValueError("Episode task identity is required for stratification")
        groups[(str(episode["suite"]), str(task))].append(index)
    rng = random.Random(seed)
    train, val = [], []
    for key in sorted(groups):
        indices = sorted(groups[key], key=lambda i: int(episodes[i]["episode_index"]))
        if len(indices) < 2:
            raise ValueError(f"Need at least two episodes for suite/task {key}")
        rng.shuffle(indices)
        count = max(1, min(len(indices)-1, round(len(indices)*val_fraction)))
        val.extend(indices[:count])
        train.extend(indices[count:])
    return sorted(train), sorted(val)


def episode_frame_indices_flow2act(episode_indices, sample_ends):
    result = []
    for index in episode_indices:
        start = int(sample_ends[index-1]) if index else 0
        result.extend(range(start, int(sample_ends[index])))
    return result
