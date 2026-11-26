"""Read-only cache/data alignment check; never generate, copy or modify motion labels."""

import argparse
from pathlib import Path


def check_motion_cache_flow2act(cache_path, data_root, *, action_horizon=30, future_horizon=30, motion_3d=True):
    from openpi.flow2act.data_flow2act import Flow2ActMotionCache
    from openpi.flow2act.data_flow2act import Flow2ActWindows

    if min(action_horizon, future_horizon) < 1:
        raise ValueError("Action and future horizons must be positive")
    windows = Flow2ActWindows(data_root, action_horizon)
    if not windows.episodes:
        raise ValueError("Dataset contains no complete action windows")
    cache = Flow2ActMotionCache(cache_path, future_horizon, three_dimensional=motion_3d)
    for suite, _, _, episode in windows.episodes:
        cache.validate_episode(suite, episode["episode_index"], episode["length"])
    return {
        "metadata": str(cache.metadata_path),
        "episodes": len(windows.episodes),
        "complete_windows": len(windows),
        "cache_horizon": cache.metadata["horizon"],
        "future_horizon": future_horizon,
        "motion_3d": motion_3d,
    }


def main_flow2act():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/libero_lerobot"))
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--future-horizon", type=int, default=30)
    parser.add_argument("--motion-3d", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    result = check_motion_cache_flow2act(
        args.cache,
        args.data_root,
        action_horizon=args.action_horizon,
        future_horizon=args.future_horizon,
        motion_3d=args.motion_3d,
    )
    print(
        f"Cache ready: {result['episodes']} episodes, {result['complete_windows']} complete action windows; "
        f"cache H{result['cache_horizon']} covers future H{result['future_horizon']}; 3D={result['motion_3d']}"
    )
    print(f"Metadata: {result['metadata']}")
    print("Read-only metadata/array-header checks passed. No labels generated or changed; no Teacher loaded.")


if __name__ == "__main__":
    main_flow2act()
