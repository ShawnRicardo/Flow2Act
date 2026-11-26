# ruff: noqa: N802, N999
"""Resource-path and explicit-resume contracts for the publication entries."""

import dataclasses
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from openpi.flow2act.config_flow2act import Flow2ActConfig
from scripts.eval_libero_family_flow2act import benchmark_config_flow2act
from scripts.resume_flow2act import resume_environment_flow2act


def test_benchmark_paths_with_spaces_flow2act(tmp_path):
    simulator = tmp_path / "benchmark with spaces"
    output = tmp_path / "evaluation" / "libero_plus"
    result = benchmark_config_flow2act(output, simulator)
    config = json.loads((result / "config.yaml").read_text())
    base = simulator / "libero/libero"
    assert config["benchmark_root"] == str(base)
    assert config["bddl_files"] == str(base / "bddl_files")
    assert config["init_states"] == str(base / "init_files")
    assert all(Path(value).is_absolute() for value in config.values())
    assert not (simulator / "config.yaml").exists()


def test_resume_saved_batch_and_moved_output_flow2act(tmp_path):
    run = tmp_path / "published run"
    checkpoint = run / "checkpoints/checkpoint-30000"
    (checkpoint / "state").mkdir(parents=True)
    cfg = dataclasses.asdict(Flow2ActConfig(ga=2, output_dir="previous/location"))
    (checkpoint / "config_flow2act.json").write_text(json.dumps(cfg))
    for path in (run / "norm_stats.json", checkpoint / "norm_stats.json"):
        path.write_text('{"norm_stats": {}}')
    (checkpoint / "state/_CHECKPOINT_METADATA").write_text('{"commit_timestamp_nsecs": 1}')
    _, env = resume_environment_flow2act(checkpoint, {"GA": "1", "WORKERS": "2"})
    assert env["GA"] == "2"
    assert env["GLOBAL_BATCH_SIZE"] == "256"
    assert env["WORKERS"] == "2"
    assert Path(env["OUTPUT_DIR"]) == run
    assert Path(env["RESUME_CHECKPOINT"]) == checkpoint


def test_shell_defaults_match_python_flow2act(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "train_scripts").mkdir()
    entry = tmp_path / "train_scripts/train_pi05_libero_flow2act.sh"
    shutil.copyfile(root / "train_scripts/train_pi05_libero_flow2act.sh", entry)
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(f"#!{sys.executable}\nimport json, sys\nprint('CAPTURED=' + json.dumps(sys.argv[1:]))\n")
    python.chmod(0o755)
    scientific_keys = {field.name.upper() for field in dataclasses.fields(Flow2ActConfig)}
    environment = {key: value for key, value in os.environ.items() if key not in scientific_keys}
    result = subprocess.run(["bash", str(entry)], env=environment, text=True, capture_output=True, check=True)
    line = next(line for line in result.stdout.splitlines() if line.startswith("CAPTURED="))
    args = json.loads(line.removeprefix("CAPTURED="))
    options = dict(zip(args[1::2], args[2::2], strict=False))
    cfg = Flow2ActConfig()
    assert options["--stage"] == cfg.stage == "motion3d"
    assert int(options["--memory-length"]) == cfg.memory_length == 3
    assert int(options["--global-batch-size"]) == cfg.global_batch_size
    assert int(options["--ga"]) == cfg.ga
    assert "motion3d_mem3_" in options["--output-dir"]
    assert options["--norm-stats"] == cfg.norm_stats
