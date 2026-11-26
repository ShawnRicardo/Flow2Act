"""Check real LIBERO camera rendering before loading a policy for evaluation.

The caller selects the simulator installation and OpenGL backend through its
environment. This check does not produce benchmark rollout records.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import random
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("libero", "libero_plus"), required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def run(args):
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository))
    config_directory = (
        "config_flow2act"
        if args.family == "libero"
        else "config_LIBEROPlus_flow2act"
    )
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(repository / "eval_libero" / config_directory))
    context = {
        "family": args.family,
        "seed": args.seed,
        "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
        "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "MUJOCO_EGL_DEVICE_ID": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        "DISPLAY": os.environ.get("DISPLAY"),
    }
    print("RENDER_SMOKE_START " + json.dumps(context, sort_keys=True), flush=True)

    import mujoco
    import numpy as np
    from OpenGL import GL
    from libero.libero import benchmark, get_libero_path

    random.seed(args.seed)
    np.random.seed(args.seed)
    # LIBERO-Plus prints the order of thousands of tasks during construction.
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        suite = benchmark.get_benchmark_dict()["libero_spatial"](task_order_index=0)
    task = suite.get_task(0)
    states = suite.get_task_init_states(0)
    if len(states) == 0:
        raise ValueError("libero_spatial task 0 has no official initial state")
    bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    environment = None
    try:
        if args.family == "libero_plus":
            from eval_libero.libero_plus_env_recovery_flow2act import (
                create_initialized_plus_environment,
            )

            environment, observation = create_initialized_plus_environment(
                bddl_path=bddl_path,
                init_state=np.array(states[0], copy=True),
                seed=args.seed,
                task_id=0,
                env_resolution=256,
            )
            dummy_action = np.zeros(7, dtype=np.float64)
        else:
            from libero.libero.envs import OffScreenRenderEnv

            environment = OffScreenRenderEnv(
                bddl_file_name=bddl_path,
                camera_heights=256,
                camera_widths=256,
            )
            environment.seed(args.seed)
            # ControlEnv.reset has an unbounded exception-catching loop. Allow
            # renderer failures in this preflight to propagate immediately.
            environment.env.reset()
            observation = environment.set_init_state(np.array(states[0], copy=True))
            dummy_action = [0.0] * 6 + [-1.0]
        for _ in range(2):
            observation, _, _, _ = environment.step(dummy_action)

        def gl_string(name):
            value = GL.glGetString(name)
            if value is None:
                raise RuntimeError("No active OpenGL context after offscreen rendering")
            return value.decode("utf-8", errors="replace")

        report = {
            **context,
            "mujoco_version": mujoco.__version__,
            "suite": "libero_spatial",
            "task_id": 0,
            "task_name": task.name,
            "official_init_state_index": 0,
            "dummy_steps": 2,
            "GL_VENDOR": gl_string(GL.GL_VENDOR),
            "GL_RENDERER": gl_string(GL.GL_RENDERER),
            "GL_VERSION": gl_string(GL.GL_VERSION),
            "images": {},
        }
        images = {}
        for camera, key in (
            ("main", "agentview_image"),
            ("wrist", "robot0_eye_in_hand_image"),
        ):
            image = np.asarray(observation[key])
            if image.shape != (256, 256, 3) or image.dtype != np.uint8:
                raise ValueError(f"{camera}: expected uint8 [256,256,3], got {image.shape} {image.dtype}")
            if not np.isfinite(image).all() or not np.any(image != image[0, 0]):
                raise ValueError(f"{camera}: rendered image is nonfinite or spatially constant")
            report["images"][camera] = {
                "observation_key": key,
                "shape": list(image.shape),
                "dtype": str(image.dtype),
                "min": int(image.min()),
                "max": int(image.max()),
                "std": float(image.std()),
            }
            # Use the same 180-degree correction as the evaluation observation.
            images[camera] = np.ascontiguousarray(image[::-1, ::-1])
        if args.output_dir is not None:
            from PIL import Image

            args.output_dir.mkdir(parents=True, exist_ok=True)
            for camera, image in images.items():
                Image.fromarray(image).save(args.output_dir / f"{args.family}_{camera}_rgb.png")
            (args.output_dir / f"{args.family}_render_smoke.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print("RENDER_SMOKE_OK " + json.dumps(report, sort_keys=True), flush=True)
    finally:
        if environment is not None:
            environment.close()


if __name__ == "__main__":
    run(parse_args())
