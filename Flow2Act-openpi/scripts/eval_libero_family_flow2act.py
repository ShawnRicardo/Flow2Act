# ruff: noqa: N802, N999
"""Launch eight JAX EMA servers and full LIBERO/LIBERO-Plus evaluation."""

import argparse
from collections import Counter
import contextlib
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


def log_flow2act(message):
    print(f"{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')} {message}", flush=True)


def write_json_flow2act(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.chmod(0o777)
    temporary.replace(path)
    path.chmod(0o777)


def benchmark_config_flow2act(directory, simulator):
    """Create a local benchmark config without requiring a machine-specific checkout layout."""
    directory = Path(directory)
    directory.mkdir(parents=True)
    directory.chmod(0o777)
    base = Path(simulator).resolve() / "libero/libero"
    # JSON is valid YAML and safely quotes checkout paths containing spaces.
    write_json_flow2act(
        directory / "config.yaml",
        {
            "assets": str(base / "assets"),
            "bddl_files": str(base / "bddl_files"),
            "benchmark_root": str(base),
            "datasets": str(base.parent / "datasets"),
            "init_states": str(base / "init_files"),
        },
    )
    return directory


def progress_flow2act(output, catalog):
    totals = {"libero": dict.fromkeys(SUITES, 500), "libero_plus": {name: len(catalog[name]) for name in SUITES}}
    records = {family: [] for family in totals}
    identities = set()
    for family, family_records in records.items():
        for path in (output / family).glob("shard-*/*/episodes.jsonl"):
            for line in path.read_bytes().splitlines(keepends=True):
                if not line.endswith(b"\n"):
                    continue
                row = json.loads(line)
                if row.get("smoke_only"):
                    continue
                identity = (family, row["suite"], row["task_id"], row["episode_idx"])
                if identity in identities:
                    raise ValueError(f"Duplicate evaluation episode: {identity}")
                identities.add(identity)
                family_records.append(row)

    def metric(rows, expected):
        completed = [r for r in rows if r["status"] == "completed"]
        successes = sum(r["success"] for r in completed)
        return {
            "completed": len(completed),
            "expected": expected,
            "successes": successes,
            "technical_errors": sum(r["status"] == "error" for r in rows),
            "success_rate": successes / len(completed) if completed else None,
        }

    snapshot = {"seed": 7, "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "benchmarks": {}}
    lines = []
    for family, rows in records.items():
        result = metric(rows, sum(totals[family].values()))
        result["suites"] = {
            suite: metric([r for r in rows if r["suite"] == suite], total) for suite, total in totals[family].items()
        }
        if family == "libero_plus":
            categories = Counter(r["category"] for suite in SUITES for r in catalog[suite])
            result["perturbations"] = {
                name: metric([r for r in rows if r["perturbation"] == name], total)
                for name, total in sorted(categories.items())
            }
        snapshot["benchmarks"][family] = result
        rate = "—" if result["success_rate"] is None else f"{100 * result['success_rate']:.2f}%"
        lines.append(
            f"{family.upper()}  completed {result['completed']}/{result['expected']}  success {rate}  technical_errors={result['technical_errors']}"
        )
        for name, row in {**result["suites"], **result.get("perturbations", {})}.items():
            rate = "—" if row["success_rate"] is None else f"{100 * row['success_rate']:.2f}%"
            lines.append(
                f"  {name:25s} {row['completed']:5d}/{row['expected']:<5d} success {rate:>8s} ({row['successes']} successes) errors={row['technical_errors']}"
            )
    snapshot["complete"] = all(
        r["completed"] == r["expected"] and r["technical_errors"] == 0 for r in snapshot["benchmarks"].values()
    )
    write_json_flow2act(output / "summary.json", snapshot)
    text = f"Updated {snapshot['updated_at']} | seed=7 | complete={snapshot['complete']}\n" + "\n".join(lines)
    (output / "progress.txt").write_text(text + "\n")
    (output / "progress.txt").chmod(0o777)
    print(text + "\n", flush=True)
    return snapshot


def run_flow2act(args):
    simulators = {
        "libero": args.libero_root.expanduser().resolve(),
        "libero_plus": args.plus_root.expanduser().resolve(),
    }
    args.libero_python = args.libero_python.expanduser().resolve()
    args.plus_python = args.plus_python.expanduser().resolve()
    for simulator in simulators.values():
        if not (simulator / "libero/libero/benchmark").is_dir():
            raise FileNotFoundError(f"Missing LIBERO benchmark checkout: {simulator}")
    catalog_path = (
        args.classification or simulators["libero_plus"] / "libero/libero/benchmark/task_classification.json"
    ).resolve()
    catalog = json.loads(catalog_path.read_text())
    output = args.output_dir.resolve()
    if output.exists() and any(p.name != "log.log" for p in output.iterdir()):
        raise FileExistsError(f"Refusing nonempty evaluation directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    output.chmod(0o777)
    logs = output / "logs"
    logs.mkdir()
    logs.chmod(0o777)
    benchmark_configs = {
        family: benchmark_config_flow2act(output / "benchmark_configs_flow2act" / family, simulator)
        for family, simulator in simulators.items()
    }
    checkpoint = args.checkpoint.resolve()
    config = json.loads((checkpoint / "config_flow2act.json").read_text())
    commit = json.loads((checkpoint / "state/_CHECKPOINT_METADATA").read_text())
    if not commit.get("commit_timestamp_nsecs"):
        raise ValueError("Evaluation requires a committed checkpoint")
    if sum(len(catalog[name]) for name in SUITES) != 10030:
        raise ValueError("Unexpected LIBERO-Plus catalog")
    gpus = list(range(8))
    for gpu in gpus:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.base_port + gpu))
    configuration = {
        "checkpoint": str(checkpoint),
        "checkpoint_global_step": int(checkpoint.name.split("-")[1]),
        "policy": "pi05 JAX Flow2Act EMA",
        "weights": "ema",
        "seed": 7,
        "gpu_ids": gpus,
        "generation_steps": 10,
        "action_horizon": config["action_horizon"],
        "execute_horizon": config["execute_horizon"],
        "future_horizon": config["future_horizon"],
        "memory_length": config["memory_length"],
        "suites": SUITES,
        "vanilla_episodes": 2000,
        "plus_episodes": 10030,
        "vanilla_trials_per_task": 50,
        "vanilla_settle_steps": 10,
        "vanilla_max_steps": {"libero_10": 520, "libero_goal": 300, "libero_object": 280, "libero_spatial": 220},
        "plus_initial_state": 0,
        "plus_zero_action_settle_steps": 5,
        "plus_max_steps": 720,
        "seed_protocol": "NumPy SeedSequence(seed,suite_index,task_id,episode); explicit FP32 flow noise per decision",
        "render_backend": "EGL",
        "benchmark_roots": {family: str(path) for family, path in simulators.items()},
        "classification": str(catalog_path),
        "mujoco_versions": {
            family: subprocess.check_output(
                [str(executable), "-c", "from importlib.metadata import version; print(version('mujoco'))"],
                text=True,
            ).strip()
            for family, executable in (("libero", args.libero_python), ("libero_plus", args.plus_python))
        },
        "save_videos": False,
        "image_protocol": "full 256x256 FOV; fixed simulator-to-dataset 180-degree camera convention",
        "action_protocol": "native delta; LeRobot binary gripper >0.5 -> -1 else +1",
    }
    write_json_flow2act(output / "evaluation_config.json", configuration)
    base = dict(os.environ)
    base.update(
        {
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "NO_PROXY": "127.0.0.1,localhost,::1",
        }
    )
    cpus = sorted(os.sched_getaffinity(0))
    handles, servers, workers, processes = [], {}, {}, []

    def spawn(command, environment, logfile, gpu):
        stream = logfile.open("x")
        logfile.chmod(0o777)
        handles.append(stream)
        affinity = cpus[gpu * 8 : (gpu + 1) * 8] or cpus[:8]
        command = ["taskset", "-c", ",".join(map(str, affinity)), *command]
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append(process)
        return process

    def start_worker(family, gpu, *, smoke=False):
        executable = args.plus_python if family == "libero_plus" else args.libero_python
        environment = dict(base)
        simulator = simulators[family]
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "MUJOCO_EGL_DEVICE_ID": str(gpu),
                "LIBERO_PLUS_SUPPRESS_GYM_WARNINGS": "1",
                "LIBERO_CONFIG_PATH": str(benchmark_configs[family]),
                "PYTHONPATH": os.pathsep.join(
                    map(str, (ROOT, ROOT / "src", ROOT / "packages/openpi-client/src", simulator))
                ),
                "LD_LIBRARY_PATH": str(executable.parent.parent / "lib") + os.pathsep + base.get("LD_LIBRARY_PATH", ""),
            }
        )
        environment.pop("DISPLAY", None)
        environment.pop("XAUTHORITY", None)
        destination = output / "smoke" if smoke else output
        command = [
            str(executable),
            str(ROOT / "examples/libero/eval_worker_flow2act.py"),
            "--family",
            family,
            "--port",
            str(args.base_port + gpu),
            "--output-dir",
            str(destination),
            "--num-shards",
            str(1 if smoke else 4 if family == "libero" else 8),
            "--shard-index",
            str(0 if smoke else gpu),
            "--seed",
            "7",
            "--classification",
            str(catalog_path),
        ]
        if smoke:
            command += ["--smoke-steps", "20", "--episode-limit", "1"]
        logfile = logs / f"{'smoke-' if smoke else ''}{family}-gpu{gpu}.log"
        process = spawn(command, environment, logfile, gpu)
        log_flow2act(f"WORKER_START family={family} GPU={gpu} smoke={smoke} pid={process.pid} log={logfile}")
        return process

    try:
        log_flow2act(f"EVALUATION_START checkpoint={checkpoint} GPUs={gpus} LIBERO=2000 LIBERO_PLUS=10030")
        for gpu in gpus:
            environment = dict(base)
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "JAX_PLATFORMS": "cuda",
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                    "PYTHONPATH": f"{ROOT}/src:{ROOT}/packages/openpi-client/src",
                }
            )
            command = [
                sys.executable,
                str(ROOT / "scripts/serve_eval_flow2act.py"),
                "--checkpoint",
                str(checkpoint),
                "--port",
                str(args.base_port + gpu),
                "--seed",
                "7",
            ]
            servers[gpu] = spawn(command, environment, logs / f"server-gpu{gpu}.log", gpu)
            log_flow2act(f"SERVER_START GPU={gpu} port={args.base_port + gpu} pid={servers[gpu].pid}")
        pending = set(gpus)
        deadline = time.monotonic() + 1200
        while pending:
            for gpu in list(pending):
                if servers[gpu].poll() is not None:
                    raise RuntimeError(
                        f"JAX server GPU {gpu} exited {servers[gpu].returncode}; inspect server-gpu{gpu}.log"
                    )
                try:
                    with urlopen(f"http://127.0.0.1:{args.base_port + gpu}/healthz", timeout=1) as response:
                        if response.status == 200:
                            pending.remove(gpu)
                            log_flow2act(f"SERVER_READY GPU={gpu}")
                except OSError:
                    pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"JAX server warmup timeout: {sorted(pending)}")
            time.sleep(2)
        for family in ("libero", "libero_plus"):
            process = start_worker(family, 0, smoke=True)
            if process.wait(timeout=600):
                raise RuntimeError(f"{family} real rollout smoke failed; inspect smoke-{family}-gpu0.log")
            log_flow2act(f"REAL_ROLLOUT_SMOKE_OK family={family} GPU=0 steps_cap=20 excluded_from_full_results")
        for gpu in gpus:
            family = "libero" if gpu < 4 else "libero_plus"
            workers[gpu] = (family, start_worker(family, gpu))
        last_summary = 0
        failed = []
        while workers:
            for gpu, (family, process) in list(workers.items()):
                if servers[gpu].poll() is not None:
                    raise RuntimeError(f"JAX server GPU {gpu} exited")
                if process.poll() is None:
                    continue
                log_flow2act(f"WORKER_EXIT family={family} GPU={gpu} returncode={process.returncode}")
                if process.returncode:
                    failed.append((family, gpu, process.returncode))
                if family == "libero":
                    workers[gpu] = ("libero_plus", start_worker("libero_plus", gpu))
                else:
                    del workers[gpu]
            if time.monotonic() - last_summary >= 30:
                progress_flow2act(output, catalog)
                last_summary = time.monotonic()
            time.sleep(2)
        snapshot = progress_flow2act(output, catalog)
        if failed or not snapshot["complete"]:
            raise RuntimeError(f"Evaluation incomplete; failed workers={failed}; see summary.json and worker logs")
        log_flow2act("EVALUATION_COMPLETE: LIBERO=2000/2000 LIBERO_PLUS=10030/10030; summary.json saved")
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for stream in handles:
            stream.close()


if __name__ == "__main__":
    os.umask(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-port", type=int, default=26800)
    parser.add_argument("--libero-python", type=Path, required=True)
    parser.add_argument("--plus-python", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, default=ROOT / "third_party/libero")
    parser.add_argument("--plus-root", type=Path, default=ROOT / "third_party/LIBERO-plus")
    parser.add_argument("--classification", type=Path, help="Defaults to the LIBERO-Plus task classification catalog")
    run_flow2act(parser.parse_args())
