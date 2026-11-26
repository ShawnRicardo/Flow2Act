"""Import a local JAX EMA checkpoint for inference; retain its trained memory setting."""

import argparse
import os
from pathlib import Path

from openpi.flow2act.artifacts_flow2act import import_checkpoint_flow2act


def main_flow2act():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    args = parser.parse_args()
    os.umask(0)
    cfg, step = import_checkpoint_flow2act(
        args.source, args.output, config_path=args.config, tokenizer_path=args.tokenizer_path
    )
    print(f"Imported EMA update={step}, stage={cfg.stage}, memory_length={cfg.memory_length}: {args.output}")
    print(
        "Tensor values and norm stats are unchanged. This EMA-only checkpoint supports inference, not training resume."
    )


if __name__ == "__main__":
    main_flow2act()
