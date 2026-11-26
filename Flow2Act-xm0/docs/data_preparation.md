# Data, normalization statistics, and frozen teacher preparation

Run preparation once before launching training. These commands do not belong in
each distributed rank. Run all examples from this repository's root. Keep
datasets, weights, caches and results on your large storage volume; the local
`data/`, `checkpoints/` and `results/` directories may be symlinks to that volume.
No weights, datasets or prior experimental outputs are included in this release.

## LIBERO four-suite input

`DATA_ROOT` must contain the four **LeRobot v2.1** datasets:

```text
data/libero_lerobot/
├── libero_10_no_noops_1.0.0_lerobot/
├── libero_goal_no_noops_1.0.0_lerobot/
├── libero_object_no_noops_1.0.0_lerobot/
└── libero_spatial_no_noops_1.0.0_lerobot/
```

Each dataset contains `meta/info.json`, `meta/episodes.jsonl`, task metadata,
per-episode parquet actions / state and synchronized RGB videos. The loader uses
`info.json`'s train split and path templates. The two camera keys are
`observation.images.image` (main) and `observation.images.wrist_image` (wrist).
Videos must preserve the original full field of view, frame order and episode
boundaries. The Motion exporter requires native 256×256 main-camera frames; it
does not change FPS or synthesize intermediate frames.

This release reads this existing format directly. It does not bundle a raw
LIBERO HDF5-to-LeRobot converter or silently treat raw HDF5 as compatible input.
Use the same prepared dataset when comparing stages.

## Horizon-specific action statistics

Use the policy training environment. Statistics must match `ACTION_LENGTH`; the
original H30 file cannot be truncated to represent the H10 window distribution.
The script uses complete action windows within episodes and converts gripper
values from `{0,1}` to signed `1−2g` before computing statistics.

```bash
python xr0/tools/compute_libero_stats_flow2act.py \
  --data-root ./data/libero_lerobot --action-length 10 \
  --output ./checkpoints/Flow2Act/libero4mix_H10_norm_stats_flow2act.json

python xr0/tools/compute_libero_stats_flow2act.py \
  --data-root ./data/libero_lerobot --action-length 30 \
  --output ./checkpoints/Flow2Act/libero4mix_H30_norm_stats_flow2act.json
```

Pass the matching file through `NORM_STATS`. `FUTURE_HORIZON` controls Motion
queries / labels independently of the action prediction length.

## Dedicated preprocessing environment

The online frozen DA3 teacher used by policy training is included in the policy
environment. Full Track4World inference is only required to **generate** Motion
caches and can use its own environment:

```bash
conda create -n flow2act-preprocess python=3.11 -y
conda activate flow2act-preprocess
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements/preprocess.txt
```

The vendored PyTorch RoPE fallback is sufficient; optional CUDA RoPE / 3D Gaussian
rendering extensions are not required for native-UV cache generation. A warning
about missing `gsplat` concerns an unused 3DGS renderer.

## Teacher weights

The static teacher and independent depth reconstruction both use DA3 components
from **Track4World's checkpoint**, not a substituted standalone DA3 checkpoint.
Prepare the files below under `checkpoints/Track4World` or point
`TRACK4WORLD_CHECKPOINTS_ROOT` at an external directory with this layout:

```text
checkpoints/Track4World/
├── track4world_da3.pth
├── DA3NESTED-GIANT-LARGE-1.1/
│   ├── config.json
│   └── model.safetensors
└── torchvision/
    └── convnext_tiny-983f1562.pth
```

Public source commands, using the preprocessing environment:

```bash
python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download
from urllib.request import urlretrieve
root = Path('checkpoints/Track4World')
root.mkdir(parents=True, exist_ok=True)
hf_hub_download('TencentARC/Track4World', 'track4world_da3.pth', local_dir=root)
for name in ('config.json', 'model.safetensors'):
    hf_hub_download('depth-anything/DA3NESTED-GIANT-LARGE-1.1', name,
                    local_dir=root / 'DA3NESTED-GIANT-LARGE-1.1')
convnext = root / 'torchvision/convnext_tiny-983f1562.pth'
convnext.parent.mkdir(parents=True, exist_ok=True)
if not convnext.exists():
    urlretrieve('https://download.pytorch.org/models/convnext_tiny-983f1562.pth', convnext)
PY

python preprocess/extract_da3_weights_flow2act.py \
  --checkpoint ./checkpoints/Track4World/track4world_da3.pth \
  --output-dir ./checkpoints/da3_flow2act
```

The extraction script copies only the frozen AnyView giant backbone and native
DualDPT tensors, validates their keys, counts, FP32 types and finite values, and
refuses to replace different existing files. Its outputs are:

```text
checkpoints/da3_flow2act/
├── da3_anyview_giant_track4world_flow2act.safetensors
└── da3_dualdpt_track4world_flow2act.safetensors
```

Use `TEACHER_WEIGHTS_DIR=./checkpoints/da3_flow2act` in policy / probe launches.
Only those extracted files are needed by static training and depth reconstruction.
The full Nested and ConvNeXt weights are needed by the offline Motion exporter:
Track4World replaces the Nested model's AnyView branch from its own checkpoint,
while the Nested checkpoint supplies its metric branch.

The extraction may require substantial CPU RAM. It reads the source checkpoint
with mmap and processes each artifact separately; it does not launch a GPU.
See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) before redistributing any
derived model files.

## Native-UV Motion cache

First run a CPU-only decoder contract check:

```bash
python preprocess/export_motion_flow2act.py --self-test
```

Then export all four training suites, using a free GPU and the preprocessing
environment. `OUTPUT_ROOT` must be a **new** directory on the large volume.
The following default is under `checkpoints/`, never inside the shared raw data:

```bash
DATA_ROOT=./data/libero_lerobot \
OUTPUT_ROOT=./checkpoints/cache/libero4mix_motion3d_flow2act \
TRACK4WORLD_CHECKPOINTS_ROOT="$PWD/checkpoints/Track4World" \
CUDA_VISIBLE_DEVICES=0 MOTION_3D_ENABLED=true \
bash preprocess/export_motion_flow2act.sh
```

Use `MOTION_3D_ENABLED=false` for a 2D-only cache. A full 3D cache is reusable by
`motion2d`: the reader then opens only the 2D files. The exporter owns one episode
at a time and refuses to overwrite any existing episode directory. For a
diagnostic before a full run, append `--smoke --limit-episodes 1`; smoke output is
not a training cache and must use a different `OUTPUT_ROOT`.

The scientific contract is fixed: canonical contexts contain at most 128 real
frames, each context owns 78 current anchors, and the native iterative decoder
uses S16 / stride8 / four iterations with overlap warm-start. The stored horizon
is 50 future retained-frame indices. Training can read the first 10 or 30 without
regenerating trajectories. No target crosses an episode boundary.

For an episode of length `T`, the cache stores:

| Array | Shape | Meaning |
| --- | --- | --- |
| `displacement_normalized_flow2act.npy` | `[T,50,2,32,32]` | Cumulative main-camera pixel displacement from anchor `t` to `t+h`, channels `(dx/W,dy/H)`; not adjacent-frame flow. |
| `native_visconf_flow2act.npy` | `[T,50,2,32,32]` | Native sigmoid visibility and confidence. |
| `in_bounds_flow2act.npy` | `[T,50,32,32]` | Finite endpoint / full-image bounds before float16 storage. |
| `horizon_valid_flow2act.npy` | `[T,50]` | Whether the target is a real future frame in the same episode. |
| `delta_xyz_normalized_flow2act.npy` | `[T,50,3,32,32]` | Optional `(ΔX,ΔY,ΔZ)` in the **current anchor camera** frame divided by anchor median depth. |
| `valid3d_flow2act.npy` | `[T,50,32,32]` | Optional depth / calibration / geometry validity. |
| `anchor_scale_flow2act.npy` | `[T]` | Optional anchor median-depth scale used during lifting. |

The 3D target is constructed from the same native 2D trajectories, original DA3
depth and full intrinsics / poses; it is not the native Track4World XYZ flow.
Geometry context NPZ files are retained as diagnostics. These are pseudo-labels,
not ground-truth metric motion. The loader also masks visibility / confidence
below 0.5, episode tails and invalid geometry.

The root `metadata_flow2act.json` declares the `Flow2Act-motion-v1` cache
contract; its per-episode filenames match the table above. Existing development
caches must be imported into a new directory with
`python scripts/import_flow2act_artifacts.py motion-cache --source /path/to/existing-cache --output /path/to/new-flow2act-cache`.
The importer preserves array values and supervision semantics without rerunning
the Teacher or modifying the source cache. Extracted DA3 files can similarly be
imported with the `da3-weights` subcommand. See the root README for full commands.

To use the cache in the stage launcher:

```bash
STAGE=motion3d MEMORY_LENGTH=1 \
MOTION_CACHE_ROOT=./checkpoints/cache/libero4mix_motion3d_flow2act \
TEACHER_WEIGHTS_DIR=./checkpoints/da3_flow2act \
bash train_scripts/train_flow2act_libero.sh
```

`baseline` needs neither DA3 nor a Motion cache; `static` needs only the extracted
DA3 backbone; `motion2d` needs static weights and 2D cache files; `motion3d` also
needs the 3D arrays. DepthOnly needs the extracted backbone and DualDPT, but no
Motion cache or action statistics.
