"""Evaluate Flow2Act checkpoints on selected LIBERO benchmarks and suites."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import signal
import shutil
import socket
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def family_gpu_assignment(gpu_ids, benchmarks):
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)) or any(gpu < 0 for gpu in gpu_ids):
        raise ValueError("GPU IDs must be distinct nonnegative physical IDs")
    if len(benchmarks) == 1:
        return {benchmarks[0]: gpu_ids}
    if len(gpu_ids) == 1:
        return {"libero": gpu_ids}
    middle = max(1, len(gpu_ids) // 2)
    return {"libero": gpu_ids[:middle], "libero_plus": gpu_ids[middle:]}


def cpu_offload_assignment(value, gpu_ids):
    result = {gpu: 0 for gpu in gpu_ids}
    assigned = set()
    for item in value.split(",") if value.strip() else ():
        try:
            gpu_text, count_text = item.split(":")
            gpu, count = int(gpu_text), int(count_text)
        except ValueError as error:
            raise ValueError("CPU offload map must use GPU:layer_count entries, e.g. 0:12,2:12") from error
        if gpu not in result or gpu in assigned or count < 0:
            raise ValueError(f"Invalid or duplicate CPU offload assignment: {item!r}")
        result[gpu] = count
        assigned.add(gpu)
    return result


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}", flush=True)


def continuation_records(args, configuration):
    """Explicit continuation imports only completed, protocol-compatible episodes."""
    from eval_libero.completed_flow2act import completed_episode_records
    from eval_libero.progress_flow2act import build_snapshot
    if not args.continue_from:
        return None, {}
    source = json.loads((args.continue_from / "evaluation_config.json").read_text())
    if Path(source["checkpoint"]).expanduser().resolve() != args.checkpoint:
        raise ValueError("Continuation must use exactly the source checkpoint")
    for field in ("seed", "generation_steps", "seed_protocol", "mujoco_versions", "render_backend",
                  "vanilla_trials_per_task", "plus_trials_per_task", "plus_max_control_steps", "save_videos", "suites", "benchmarks"):
        if source.get(field) != configuration[field]:
            raise ValueError(f"Continuation protocol mismatch: {field}")
    if source.get("policy_metadata") != configuration["policy_metadata"]:
        raise ValueError("Continuation policy metadata mismatch")
    snapshot = build_snapshot(args.continue_from, args.classification, args.seed,
                              benchmarks=args.benchmarks, suites=args.suites)
    for field in ("data_errors", "duplicate_episodes", "ignored_partial_lines", "ignored_smoke_records"):
        if snapshot[field]:
            raise ValueError(f"Cannot continue a source with {field}: {snapshot[field]}")
    imported = {}
    for family in args.benchmarks:
        rows = completed_episode_records(args.continue_from, family, seed=args.seed,
                                         metadata=configuration["policy_metadata"])
        imported[family] = {identity: row for identity, row in rows.items() if identity[0] in args.suites}
        if len(imported[family]) != snapshot["benchmarks"][family]["completed"]:
            raise ValueError(f"Completed-record count differs for {family}")
    return snapshot, imported


def receive_exact(connection, count):
    data = b""
    while len(data) < count:
        block = connection.recv(count - len(data))
        if not block:
            raise ConnectionError("Policy metadata response was truncated")
        data += block
    return data


def metadata(port):
    with socket.create_connection(("127.0.0.1", port), timeout=2) as connection:
        body = pickle.dumps({"request_type": "metadata"}, protocol=4)
        connection.sendall(struct.pack(">I", len(body)) + body)
        count = struct.unpack(">I", receive_exact(connection, 4))[0]
        return pickle.loads(receive_exact(connection, count))


def report_progress(args, *, final=False):
    command = [args.server_python, str(ROOT / "eval_libero/progress_flow2act.py"),
               "--output-dir", str(args.output_dir), "--seed", str(args.seed),
               "--benchmarks", ",".join(args.benchmarks), "--suites", ",".join(args.suites),
               "--classification", str(args.classification)]
    if final:
        command.append("--final")
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with (args.output_dir / "progress.log").open("a", buffering=1) as stream:
        stream.write(result.stdout)
    print(result.stdout, end="", flush=True)
    if result.returncode:
        raise RuntimeError(f"Evaluation summary failed with exit code {result.returncode}")


def tail(path, count=50):
    if path.exists():
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 12000))
            return "\n".join(stream.read().decode(errors="replace").splitlines()[-count:])
    return "Log file has not been created"


def simulator_process(args, base_env, logs, family, gpu):
    """Keep simulator packages and OpenGL libraries out of the policy process."""
    executable = args.libero_python if family == "libero" else args.plus_python
    simulator = args.libero_root if family == "libero" else args.libero_plus_root
    config = logs / f"config-{family}"
    config.mkdir(exist_ok=True)
    package = simulator / "libero/libero"
    paths = {"benchmark_root": package, "bddl_files": package / "bddl_files",
             "init_states": package / "init_files", "assets": package / "assets",
             "datasets": simulator / "datasets"}
    # JSON strings are valid YAML scalars, including paths containing spaces.
    (config / "config.yaml").write_text("".join(f"{key}: {json.dumps(str(value))}\n" for key, value in paths.items()))
    environment = {**base_env, "CUDA_VISIBLE_DEVICES": str(gpu),
                   "PYTHONPATH": f"{ROOT}:{simulator}", "MUJOCO_GL": "egl",
                   "PYOPENGL_PLATFORM": "egl", "LP_NUM_THREADS": "2",
                   "LIBERO_CONFIG_PATH": str(config), "LIBERO_ROOT": str(args.libero_root),
                   "LIBERO_PLUS_ROOT": str(args.libero_plus_root), "MUJOCO_EGL_DEVICE_ID": str(gpu),
                   "LIBERO_PLUS_SUPPRESS_GYM_WARNINGS": "1"}
    environment["LD_LIBRARY_PATH"] = str(Path(executable).parent.parent / "lib") + os.pathsep + base_env.get("LD_LIBRARY_PATH", "")
    environment.pop("DISPLAY", None)
    environment.pop("XAUTHORITY", None)
    return environment, [executable]


def _check_render_pair(args, base_env, logs, family, gpu):
    """Check one original rendering path, allowing explicit cold-import time."""
    environment, prefix = simulator_process(args, base_env, logs, family, gpu)
    logfile = logs / f"render-check-{family}-gpu{gpu}.log"
    command = [*prefix, str(ROOT / "eval_libero/smoke_render_flow2act.py"),
               "--family", family, "--seed", str(args.seed),
               "--output-dir", str(logs / "render-check" / f"gpu{gpu}")]
    log(f"RENDER_CHECK_START family={family} GPU={gpu} backend={args.render_backend} initialization_timeout={args.render_check_timeout}s")
    try:
        with logfile.open("w") as stream:
            process = subprocess.Popen(command, env=environment, cwd=logs, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = process.wait(timeout=args.render_check_timeout)
            except BaseException:
                # Only this newly created simulator process group is owned here.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{family} GPU {gpu} rendering initialization timed out after {args.render_check_timeout}s; "
                           f"see {logfile}\n" + tail(logfile)) from error
    if returncode:
        diagnostics = tail(logfile)
        raise RuntimeError(f"{family} GPU {gpu} rendering check exited {returncode}; "
                           f"backend={args.render_backend}; see {logfile}\n{diagnostics}")
    if "RENDER_SMOKE_OK " not in logfile.read_text():
        raise RuntimeError(f"Rendering check exited without verifying both cameras: {logfile}")
    report = json.loads((logs / "render-check" / f"gpu{gpu}" / f"{family}_render_smoke.json").read_text())
    expected = getattr(args, "mujoco_versions", {}).get(family)
    if expected is not None and report["mujoco_version"] != expected:
        raise ValueError(f"{family} preflight imported a different MuJoCo version: {report['mujoco_version']} != {expected}")
    log(f"RENDER_CHECK_OK family={family} GPU={gpu} mujoco={report['mujoco_version']} backend={args.render_backend} log={logfile}")


def check_rendering(args, base_env, logs, family_gpus):
    """Run at most two pair checks; cancel queued checks on the first error."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        checks = [pool.submit(_check_render_pair, args, base_env, logs, family, gpu)
                  for family, gpu_ids in family_gpus.items() for gpu in gpu_ids]
        try:
            for check in as_completed(checks):
                check.result()
        except BaseException:
            for check in checks:
                check.cancel()
            raise


def run(args):
    if args.continue_from and args.continue_from == args.output_dir:
        raise ValueError("Continuation requires a fresh output directory distinct from its source")
    if args.output_dir.exists():
        # A launcher log may be opened by the shell before Python starts.
        if any(path.name not in {"log.log", "tmux.log"} for path in args.output_dir.iterdir()):
            raise FileExistsError(f"Choose a fresh output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = args.output_dir / "logs"
    logs.mkdir()
    gpu_ids = [int(value) for value in args.gpu_ids.split(",")]
    initial_family_gpus = family_gpu_assignment(gpu_ids, args.benchmarks)
    family_gpus = {family: (gpu_ids if family == "libero_plus" else initial_family_gpus[family]) for family in args.benchmarks}
    family_num_shards = {family: len(devices) for family, devices in family_gpus.items()}
    offload_layers = cpu_offload_assignment(args.cpu_offload_layer_map, gpu_ids)
    for executable in [args.server_python] + [args.libero_python if family == "libero" else args.plus_python for family in args.benchmarks]:
        if not Path(executable).is_file():
            raise FileNotFoundError(executable)
    base_env = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "LIBERO_ROOT": str(args.libero_root), "LIBERO_PLUS_ROOT": str(args.libero_plus_root)}
    if args.render_check_only:
        check_rendering(args, base_env, logs, family_gpus)
        log("RENDER_CHECKS_COMPLETE: both cameras verified for each assigned simulator/GPU")
        return
    checkpoint = args.checkpoint
    if checkpoint.is_dir():
        candidates = (checkpoint / "checkpoint/mp_rank_00_model_states.pt",
                      checkpoint / "mp_rank_00_model_states.pt")
        checkpoint = next((path for path in candidates if path.is_file()), candidates[0])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    import torch
    sys.path.insert(0, str(ROOT / "xr0"))
    from mibot.utils.checkpoint_contract_flow2act import read_checkpoint_contract_flow2act
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    contract = read_checkpoint_contract_flow2act(payload, allowed_kinds={"trained_policy"})
    method = contract["method_config"]
    expected_metadata = {
        "architecture_version": contract["schema"], "checkpoint_kind": contract["kind"],
        "checkpoint_global_step": int(payload.get("global_step", -1)),
        "stage": method["stage"], "memory_length": method["memory_length"],
        "future_horizon": method["future_horizon"], "execute_horizon": method["execute_horizon"],
        "action_horizon": contract["model_config"]["action_shape"][0],
    }
    if expected_metadata["checkpoint_global_step"] <= 0:
        raise ValueError("Evaluation requires a checkpoint with completed training updates")
    del payload
    # Reserve no existing service ports and never interfere with other users' jobs.
    for index in range(len(gpu_ids)):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.base_port + index))
    args.mujoco_versions = {}
    for family, devices in family_gpus.items():
        environment, prefix = simulator_process(args, base_env, logs, family, devices[0])
        args.mujoco_versions[family] = subprocess.check_output(
            [*prefix, "-c", "import mujoco; print(mujoco.__version__)"], env=environment, text=True).strip()
    configuration = {
        "checkpoint": str(args.checkpoint), "model_dir": str(args.model_dir),
        "render_backend": "EGL/headless" if args.render_backend == "egl" else "GLX/Xvfb",
        "render_preflight_checked": False, "policy": "trained Flow2Act XR0",
        "render_check_timeout": args.render_check_timeout,
        "policy_metadata": expected_metadata,
        "generation_steps": contract["model_config"]["num_steps"],
        "seed_protocol": "Flow2Act evaluator: per-task/episode/decision deterministic seeds; not the official observation-hash policy seed",
        "mujoco_versions": args.mujoco_versions,
        "seed": args.seed, "gpu_ids": gpu_ids, "vanilla_episodes": len(args.suites) * 500 if "libero" in args.benchmarks else 0,
        "plus_episodes": sum(len(json.loads(args.classification.read_text())[name]) for name in args.suites) if "libero_plus" in args.benchmarks else 0, "vanilla_trials_per_task": 50,
        "plus_trials_per_task": 1, "plus_max_control_steps": 720,
        "suites": args.suites, "benchmarks": args.benchmarks, "save_videos": False,
        "families_concurrent": len(args.benchmarks) > 1 and len(gpu_ids) > 1, "family_gpu_ids": family_gpus,
        "initial_family_gpu_ids": initial_family_gpus, "family_num_shards": family_num_shards,
        "reuse_completed_libero_gpus_for_plus": len(args.benchmarks) > 1,
        "cpu_offload_layers_by_gpu": offload_layers,
    }
    source_snapshot, imported = continuation_records(args, configuration)
    configuration.update(
        continue_from=str(args.continue_from),
        imported_completed_episodes={family: len(rows) for family, rows in imported.items()},
        source_technical_errors=source_snapshot["technical_errors"] if source_snapshot else 0,
    )
    for family, rows in imported.items():
        destination = args.output_dir / family / "shard-imported" / "episodes.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as stream:
            for identity in sorted(rows):
                stream.write(json.dumps(rows[identity], ensure_ascii=False) + "\n")
    (args.output_dir / "evaluation_config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    log(f"CHECKPOINT={args.checkpoint} metadata={json.dumps(expected_metadata)}")
    log(f"OUTPUT={args.output_dir} GPUs={gpu_ids} seed={args.seed} benchmarks={args.benchmarks} suites={args.suites}")
    log(f"FAMILIES_CONCURRENT initial_GPUs={json.dumps(initial_family_gpus)} final_GPUs={json.dumps(family_gpus)} CPU_OFFLOAD_LAYERS={json.dumps(offload_layers)}")
    log(f"CONTINUE_FROM={args.continue_from} imported={configuration['imported_completed_episodes']} source_technical_errors={configuration['source_technical_errors']}")
    report_progress(args)
    check_rendering(args, base_env, logs, family_gpus)
    configuration["render_preflight_checked"] = True
    (args.output_dir / "evaluation_config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    servers, workers, handles = [], [], []

    def start(command, environment, logfile):
        handle = logfile.open("a", buffering=1)
        handles.append(handle)
        process = subprocess.Popen(command, env=environment, cwd=logs,
                                   stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        return process

    def stop_owned(processes):
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 15
        for process in processes:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    def check_servers():
        for index, process in enumerate(servers):
            if process.poll() is not None:
                raise RuntimeError(f"Policy server GPU={gpu_ids[index]} port={args.base_port + index} exited {process.returncode}\n"
                                   + tail(logs / f"server-gpu{gpu_ids[index]}.log"))

    try:
        for index, gpu in enumerate(gpu_ids):
            environment = {**base_env, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(ROOT)}
            command = [args.server_python, str(ROOT / "deploy/server_offload_flow2act.py"),
                       "--model-dir", str(args.model_dir), "--checkpoint", str(args.checkpoint),
                       "--host", "127.0.0.1", "--port", str(args.base_port + index), "--device", "cuda:0",
                       "--cpu-offload-layers", str(offload_layers[gpu]),
                       "--processor-path", str(args.processor_path)]
            process = start(command, environment, logs / f"server-gpu{gpu}.log")
            servers.append(process)
            log(f"SERVER_START GPU={gpu} pid={process.pid} port={args.base_port + index} cpu_offload_layers={offload_layers[gpu]}")
        ready = set()
        started = time.monotonic()
        while len(ready) < len(servers):
            check_servers()
            for index in range(len(servers)):
                if index in ready:
                    continue
                try:
                    value = metadata(args.base_port + index)
                except (OSError, ConnectionError):
                    continue
                if any(value.get(key) != expected for key, expected in expected_metadata.items()):
                    raise ValueError(f"Server metadata differs from the requested Flow2Act checkpoint: {value}")
                ready.add(index)
                log(f"SERVER_READY GPU={gpu_ids[index]} metadata={json.dumps(value)}")
            if time.monotonic() - started > args.server_start_timeout:
                raise TimeoutError("Timed out while loading evaluation policies")
            if len(ready) < len(servers):
                time.sleep(2)
        report_progress(args)
        worker_assignments = []
        entries = {"libero": "worker_flow2act.py",
                   "libero_plus": "main_libero_plus_flow2act.py"}

        def start_worker(family, gpu, shard_index):
            server_index = gpu_ids.index(gpu)
            environment, prefix = simulator_process(args, base_env, logs, family, gpu)
            command = [*prefix, str(ROOT / "eval_libero" / entries[family]),
                       "--output-dir", str(args.output_dir), "--host", "127.0.0.1",
                       "--port", str(args.base_port + server_index), "--num-shards", str(family_num_shards[family]),
                       "--shard-index", str(shard_index), "--seed", str(args.seed),
                       "--suites", ",".join(args.suites)]
            if args.continue_from:
                command += ["--completed-dir", str(args.continue_from)]
            workers.append(start(command, environment, logs / f"{family}-gpu{gpu}.log"))
            worker_assignments.append((family, gpu, shard_index))
            log(f"WORKER_START family={family} GPU={gpu} shard={shard_index}/{family_num_shards[family]} pid={workers[-1].pid}")

        for family, assigned_gpus in initial_family_gpus.items():
            log(f"PHASE_START {family} GPUs={assigned_gpus}")
            for gpu in assigned_gpus:
                # Plus always uses its final all-GPU sharding, including before
                # the lower-index GPUs become available after their LIBERO jobs.
                start_worker(family, gpu, family_gpus[family].index(gpu))
        finished, completed_families = set(), set()
        next_report = time.monotonic() + args.progress_interval
        while len(finished) < len(workers):
            check_servers()
            for index, process in enumerate(tuple(workers)):
                if index in finished or process.poll() is None:
                    continue
                family, gpu, shard_index = worker_assignments[index]
                if process.returncode:
                    raise RuntimeError(f"{family} GPU={gpu} shard={shard_index} exited {process.returncode}\n"
                                       + tail(logs / f"{family}-gpu{gpu}.log"))
                finished.add(index)
                log(f"WORKER_COMPLETE family={family} GPU={gpu} shard={shard_index}")
                if family == "libero" and "libero_plus" in args.benchmarks:
                    start_worker("libero_plus", gpu, gpu_ids.index(gpu))
            for family in family_gpus:
                family_indices = [index for index, assignment in enumerate(worker_assignments)
                                  if assignment[0] == family]
                if (family not in completed_families and len(family_indices) == family_num_shards[family]
                        and all(index in finished for index in family_indices)):
                    completed_families.add(family)
                    report_progress(args)
                    log(f"PHASE_COMPLETE {family}")
            if time.monotonic() >= next_report:
                report_progress(args)
                next_report = time.monotonic() + args.progress_interval
            if len(finished) < len(workers):
                time.sleep(2)
        report_progress(args, final=True)
        log("EVALUATION_COMPLETE: selected benchmark results saved to summary.json")
    except BaseException:
        # Preserve all completed episode records and make the last progress visible.
        try:
            report_progress(args)
        except Exception as summary_error:
            log(f"Could not refresh progress after failure: {summary_error}")
        raise
    finally:
        stop_owned(workers)
        stop_owned(servers)
        for handle in handles:
            handle.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processor-path", type=Path, default=ROOT / "checkpoints/Xiaomi-Robotics-0-Pretrain")
    parser.add_argument("--continue-from", type=Path, help="Explicit source evaluation; destination must be fresh")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmarks", default="libero,libero_plus")
    parser.add_argument("--suites", default=",".join(SUITES))
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--libero-root", type=Path, default=Path(os.environ.get("LIBERO_ROOT", ROOT / "third_party/LIBERO")))
    parser.add_argument("--libero-plus-root", type=Path, default=Path(os.environ.get("LIBERO_PLUS_ROOT", ROOT / "third_party/LIBERO-plus")))
    parser.add_argument("--render-backend", choices=("egl",), default="egl")
    parser.add_argument("--render-check-timeout", type=int, default=600)
    parser.add_argument("--render-check-only", action="store_true")
    parser.add_argument("--cpu-offload-layer-map", default="")
    parser.add_argument("--base-port", type=int, default=28050)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress-interval", type=int, default=30)
    parser.add_argument("--server-start-timeout", type=int, default=1200)
    parser.add_argument("--server-python", default=sys.executable)
    parser.add_argument("--libero-python", default=str(ROOT / ".venvs/libero/bin/python"))
    parser.add_argument("--plus-python", default=str(ROOT / ".venvs/libero_plus/bin/python"))
    args = parser.parse_args()
    args.benchmarks = args.benchmarks.split(",")
    args.suites = args.suites.split(",")
    if not args.benchmarks or len(set(args.benchmarks)) != len(args.benchmarks) or any(x not in ("libero", "libero_plus") for x in args.benchmarks):
        parser.error("Invalid --benchmarks selection")
    if not args.suites or len(set(args.suites)) != len(args.suites) or any(x not in SUITES for x in args.suites):
        parser.error("Invalid --suites selection")
    if min(args.render_check_timeout, args.progress_interval, args.server_start_timeout) <= 0 or args.seed < 0:
        parser.error("Timeouts and progress interval must be positive; seed must be nonnegative")
    for name in ("model_dir", "checkpoint", "processor_path", "output_dir", "libero_root", "libero_plus_root", "continue_from"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    for name in ("server_python", "libero_python", "plus_python"):
        value = getattr(args, name)
        setattr(args, name, shutil.which(value) or str(Path(value).expanduser().resolve()))
    args.classification = args.libero_plus_root / "libero/libero/benchmark/task_classification.json"
    os.environ["LIBERO_PLUS_ROOT"] = str(args.libero_plus_root)
    return args


if __name__ == "__main__":
    os.umask(0)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    run(parse_args())
