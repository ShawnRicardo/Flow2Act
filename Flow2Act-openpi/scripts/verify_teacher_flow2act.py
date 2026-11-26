# ruff: noqa: N801, N802, N999
"""Numerical verification ONLY: optional separate-process Torch reference.

The training and deployment entries never import reference project source.
Run each engine into its own exact output directory, then compare features.npy.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import types

import numpy as np


def main_flow2act():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("jax", "reference"), required=True)
    parser.add_argument("--reference-root")
    parser.add_argument("--reference-module", help="Module name relative to the reference root")
    parser.add_argument("--reference-class", help="Reference feature-backbone class name")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--compute-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument(
        "--teacher-path",
        default="checkpoints/da3_flow2act/da3_giant_flow2act.safetensors",
    )
    parser.add_argument("--rgb", required=True, help="NPY [1,2,256,256,3], FP32 RGB [0,1]")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compare")
    args = parser.parse_args()
    if args.engine == "reference" and not all((args.reference_root, args.reference_module, args.reference_class)):
        parser.error("Reference verification requires --reference-root, --reference-module, and --reference-class")
    os.umask(0)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    output.chmod(0o777)
    rgb = np.load(args.rgb, allow_pickle=False)
    if args.engine == "jax":
        import jax
        import jax.numpy as jnp

        from openpi.flow2act.teacher_flow2act import load_teacher_flow2act
        from openpi.flow2act.teacher_flow2act import teacher_features_flow2act

        expected_platform = "gpu" if args.device == "cuda" else "cpu"
        if jax.devices()[0].platform != expected_platform:
            raise ValueError("Verification --device differs from JAX_PLATFORMS")
        weights = jax.device_put(load_teacher_flow2act(args.teacher_path))
        dtype = jnp.float16 if args.compute_dtype == "float16" else jnp.float32
        run = jax.jit(lambda w, x: teacher_features_flow2act(w, x, compute_dtype=dtype))
        features = np.asarray(run(weights, jnp.asarray(rgb)))
    else:
        from safetensors.torch import load_file
        import torch

        directory = Path(args.reference_root).resolve()
        package = types.ModuleType("reference_da3_flow2act")
        package.__path__ = [str(directory)]
        sys.modules[package.__name__] = package
        source = importlib.import_module(package.__name__ + "." + args.reference_module)
        backbone = getattr(source, args.reference_class)
        with torch.device("meta"):
            module = backbone(
                "vitg", [19, 27, 33, 39], alt_start=13, qknorm_start=13, rope_start=13
            )
        module.load_state_dict(load_file(args.teacher_path), strict=True, assign=True)
        if args.device == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.18)
        module = module.to(args.device)
        module.eval()
        image = torch.from_numpy(rgb).permute(0, 1, 4, 2, 3)
        image = torch.nn.functional.interpolate(
            image.reshape(-1, 3, 256, 256), (252, 252), mode="bilinear", align_corners=False, antialias=True
        )
        image = (image - torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]) / torch.tensor(
            [0.229, 0.224, 0.225]
        )[None, :, None, None]
        with torch.no_grad(), torch.autocast(args.device, dtype=torch.float16, enabled=args.compute_dtype == "float16"):
            values, _ = module(image.to(args.device).reshape(1, 2, 3, 252, 252), ref_view_strategy="first")
        features = np.stack([pair[0].float().cpu().numpy().reshape(1, 2, 18, 18, 3072) for pair in values])
    path = output / "features.npy"
    np.save(path, features)
    path.chmod(0o777)
    if args.compare:
        reference = np.load(args.compare, allow_pickle=False)
        delta = features - reference
        report = {
            "max_abs": float(np.max(np.abs(delta))),
            "rms": float(np.sqrt(np.mean(delta**2))),
            "relative_rms": float(np.linalg.norm(delta) / np.linalg.norm(reference)),
            "shape": list(features.shape),
            "device": args.device,
            "compute_dtype": args.compute_dtype,
        }
        (output / "comparison_flow2act.json").write_text(json.dumps(report, indent=2) + "\n")
        (output / "comparison_flow2act.json").chmod(0o777)
        print(report)
        tolerance = 1e-2 if args.compute_dtype == "float16" else 1e-3
        if report["relative_rms"] > tolerance:
            raise AssertionError("DA3 reference parity failed")


if __name__ == "__main__":
    main_flow2act()
