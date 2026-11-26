# ruff: noqa: N801, N802, N999
"""Prepare native-format q01/q99 over exactly the complete H30 LIBERO4Mix windows."""

import argparse
import json
import os
from pathlib import Path

from openpi.flow2act.data_flow2act import Flow2ActWindows
from openpi.flow2act.stats_flow2act import prepare_stats_flow2act


def main_flow2act():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/libero_lerobot")
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--output-dir", default="assets/libero4mix_H30_flow2act")
    args = parser.parse_args()
    os.umask(0)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty stats directory: {output}")
    windows = Flow2ActWindows(args.data_root, args.action_horizon)
    prepare_stats_flow2act(windows, output)
    print(json.dumps(windows.sampling_spec(), indent=2))


if __name__ == "__main__":
    main_flow2act()
