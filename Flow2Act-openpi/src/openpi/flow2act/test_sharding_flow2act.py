# ruff: noqa: N802, N999
"""Run the production Teacher dispatcher on four isolated CPU devices."""

import os
from pathlib import Path
import subprocess
import sys


def test_teacher_batch_sharding_flow2act():
    entry = Path(__file__).resolve().parents[3] / "scripts/check_sharding_flow2act.py"
    result = subprocess.run(
        [sys.executable, str(entry), "--teacher-only"],
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
