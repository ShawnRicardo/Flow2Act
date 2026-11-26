# ruff: noqa: N802, N999
"""Restore the saved run configuration and enter the existing JAX training Shell."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def resume_environment_flow2act(checkpoint, environ):
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.parent.name != "checkpoints" or not re.fullmatch(r"checkpoint-[1-9][0-9]*", checkpoint.name):
        raise ValueError("Resume requires an explicit checkpoints/checkpoint-N path")
    config = json.loads((checkpoint / "config_flow2act.json").read_text())
    norm_content = (checkpoint / "norm_stats.json").read_text()
    run = checkpoint.parent.parent
    if json.loads(norm_content) != json.loads((run / "norm_stats.json").read_text()):
        raise ValueError("Saved checkpoint and run normalization statistics differ")
    metadata = json.loads((checkpoint / "state/_CHECKPOINT_METADATA").read_text())
    if not metadata.get("commit_timestamp_nsecs"):
        raise ValueError(f"Checkpoint is not committed: {checkpoint}")
    environment = dict(environ)
    for key, value in config.items():
        if key in ("output_dir", "resume_checkpoint"):
            continue
        # Only performance/logging controls may override the saved configuration.
        if key in ("workers", "log_interval", "stop_after_updates") and key.upper() in environ:
            continue
        if isinstance(value, bool):
            text = "1" if value else "0"
        elif isinstance(value, list):
            text = ",".join(map(str, value))
        else:
            text = str(value)
        environment[key.upper()] = text
    environment["OUTPUT_DIR"] = str(run)
    environment["RESUME_CHECKPOINT"] = str(checkpoint)
    return config, environment


def preserve_incomplete_flow2act(path, checkpoint, *, dry_run):
    checkpoint = Path(checkpoint).resolve()
    path = Path(path).absolute()
    if path.is_symlink():
        raise ValueError("Incomplete checkpoint must not be a symlink")
    if not path.exists():
        return
    match = re.fullmatch(r"checkpoint-([1-9][0-9]*)_flow2act\.tmp", path.name)
    if (
        path.parent.resolve() != checkpoint.parent
        or not match
        or int(match[1]) <= int(checkpoint.name.split("-")[1])
        or not path.is_dir()
    ):
        raise ValueError("Only an explicit incomplete checkpoint after the resume step in the same run is accepted")
    destination = path.with_suffix(".incomplete")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite preserved checkpoint: {destination}")
    print(f"Preserve incomplete checkpoint: {path} -> {destination}", flush=True)
    if not dry_run:
        # Same-parent rename preserves all data without copying or deleting it.
        subprocess.run(["mv", "-T", "--no-clobber", "--", str(path), str(destination)], check=True)
        if path.exists():
            raise FileExistsError(f"Incomplete checkpoint was not moved: {path}")
        destination.chmod(0o777)


def main_flow2act():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-checkpoint", required=True, type=Path)
    parser.add_argument("--incomplete-checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config, environment = resume_environment_flow2act(args.resume_checkpoint, os.environ)
    step = int(Path(environment["RESUME_CHECKPOINT"]).name.split("-")[1])
    print(f"RESUME_CHECKPOINT={environment['RESUME_CHECKPOINT']}", flush=True)
    print(f"OUTPUT_DIR={environment['OUTPUT_DIR']}", flush=True)
    print(
        f"Resume optimizer update {step + 1}/{config['max_steps']}; stage={config['stage']} "
        f"GPUs={config['num_gpus']} FSDP={config['fsdp_devices']} "
        f"GBS={config['global_batch_size']} GA={config['ga']}",
        flush=True,
    )
    if args.incomplete_checkpoint:
        preserve_incomplete_flow2act(args.incomplete_checkpoint, args.resume_checkpoint, dry_run=args.dry_run)
    if args.dry_run:
        print("Dry run: no files changed and no training started.", flush=True)
        return
    os.chdir(ROOT)
    entry = ROOT / "train_scripts/train_pi05_libero_flow2act.sh"
    os.execve("/bin/bash", ["bash", str(entry)], environment)


if __name__ == "__main__":
    main_flow2act()
