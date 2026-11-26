# Local resources

Run from the repository root. These resources are excluded by `.gitignore` and
are not bundled in the source release.

```text
data/
  libero_lerobot/
    libero_10_no_noops_1.0.0_lerobot/
    libero_goal_no_noops_1.0.0_lerobot/
    libero_object_no_noops_1.0.0_lerobot/
    libero_spatial_no_noops_1.0.0_lerobot/
  libero4mix_motion3d_flow2act/
checkpoints/
  Track4World/
    track4world_da3.pth
    DA3NESTED-GIANT-LARGE-1.1/{config.json,model.safetensors}
    torchvision/convnext_tiny-983f1562.pth
  openpi-assets/checkpoints/pi05_base/params/
  big_vision/paligemma_tokenizer.model
  da3_flow2act/da3_giant_flow2act.safetensors
assets/
  libero4mix_H30_flow2act/norm_stats.json  # Created automatically by training
  visualization_flow2act/plotly-2.35.2.min.js
```

Every training path is overridable through the Shell. Flow2Act code, metadata,
array filenames, output directories, and checkpoint sidecars use the names below.

## Base policy and tokenizer

Obtain the official JAX π0.5 base parameter tree described by
[OpenPI](https://github.com/Physical-Intelligence/openpi), including all Orbax
parameter files, not the PyTorch safetensors conversion. The native sources refer
to `gs://openpi-assets/checkpoints/pi05_base/params`.
The PaliGemma tokenizer source is `gs://big_vision/paligemma_tokenizer.model`.
Download during resource preparation and place the files at the paths above.

Training validates native parameter keys, shapes, dtypes, and finite values.
It does not replace missing parameters with random initialization.

## LIBERO4Mix

The loader expects LeRobot **v2.1**, 20 fps, per-episode parquet data and videos,
and `meta/{info.json,episodes.jsonl,tasks.jsonl}`.
The two image fields are `observation.images.image` and
`observation.images.wrist_image`; state/action dimensions are 8/7.

For the development dataset, complete H30 window counts are:

| Suite | Windows |
| --- | ---: |
| libero_10 | 90,478 |
| libero_goal | 39,630 |
| libero_object | 53,818 |
| libero_spatial | 40,442 |
| Total | 224,368 |

Do not repeat the last action at episode boundaries. Statistics must be computed
over these complete windows: states at anchors, actions over every window label.
Training automatically computes native-format `norm_stats.json` when it is
missing, using the same q01/q99 calculation as `scripts/prepare_stats_flow2act.py`.
It reads parquet state/action columns on CPU, without decoding videos or loading
the Teacher. The `sampling` description is checked against the actual training
index before an existing file is reused. Invalid or mismatched statistics are
rejected, never overwritten automatically. Concurrent launches sharing a stats
directory serialize preparation with a directory lock; no lock file is created.

The Shell defaults to `assets/libero4mix_H${ACTION_HORIZON}_flow2act`; `NORM_STATS`
overrides the directory. Select a fresh directory when changing the data.

The standalone tool remains available for optional preparation before allocating
training GPUs:

```bash
JAX_PLATFORMS=cpu .venv/bin/python scripts/prepare_stats_flow2act.py \
  --data-root data/libero_lerobot --action-horizon 30 \
  --output-dir assets/libero4mix_H30_flow2act
```

## Frozen DA3 Teacher

Reuse existing extracted FP32 safetensors through `TEACHER_PATH`; no extraction is
needed if they match the JAX Giant feature Teacher contract. The basename is not
part of that contract. The exact `pretrained.*` key layout is checked by
`expected_keys_flow2act` in `teacher_flow2act.py`.
It is **not** interchangeable with an arbitrary DA3 model.safetensors or a
Track4World .pth file. Key, shape, dtype, and finite checks are strict.

Architecture: width 1536, depth 40, heads 24, hidden 4096, patch 14, source positional
grid37×37; the current 252×252 Teacher input produces an 18×18 grid.
Teacher taps are [19,27,33,39]. Teacher weights remain FP32; GPU compute uses FP16.

## Preprocessing resources

This section applies only when the required extracted Teacher or motion cache is
missing. Existing training resources should be reused; see the next section.

The included `third_party/Track4World/` runtime is used only by standalone cache
export. Set up `flow2act-preprocess` using `requirements/preprocess_flow2act.txt`
as shown in the README. The raw local model dependencies are:

- `checkpoints/Track4World/track4world_da3.pth` from the Track4World release.
- `checkpoints/Track4World/DA3NESTED-GIANT-LARGE-1.1/{config.json,model.safetensors}`.
- `checkpoints/Track4World/torchvision/convnext_tiny-983f1562.pth`.

The README links the original sources. Extraction selects exactly the FP32 Giant
backbone keys and checks the same shape contract as the JAX Teacher:

```bash
conda run --no-capture-output -n flow2act-preprocess python preprocess/extract_da3_weights_flow2act.py \
  --checkpoint checkpoints/Track4World/track4world_da3.pth \
  --output-dir checkpoints/da3_flow2act
conda run --no-capture-output -n flow2act-preprocess bash preprocess/export_motion_flow2act.sh
```

The Shell exposes `DATA_ROOT`, `OUTPUT_ROOT`, `TRACK4WORLD_ROOT`,
`TRACK4WORLD_CHECKPOINTS_ROOT`, `CHECKPOINT`, `PYTHON` and `CUDA_VISIBLE_DEVICES`.
It defaults to the four suites in 10→goal→object→spatial order, 3D enabled and
`data/libero4mix_motion3d_flow2act`. Neither export nor training downloads weights.
A lightweight decoder/context test does not need model weights or CUDA:

```bash
conda run -n flow2act-preprocess python preprocess/export_motion_flow2act.py --self-test
```

The exporter uses real canonical contexts of up to 128 frames, 78 anchor ownership,
50 future targets, native S16/stride8 decoding with four updates and overlap
warm-start, and full-resolution UV upsampling before grid32 sampling. Three-dimensional
labels lift those UV tracks with original DA3 depth/full intrinsics/poses into
the anchor camera and divide by its valid median depth. This preserves the
reference exporter algorithm; raw Track4World XYZ predictions are not labels.
A required horizon-valid array is published last for each episode, then root
metadata after all selected episodes finish. Interrupted episodes are refused
rather than silently treated as complete. Use a new exact output directory for
an intentional re-export; no existing episode is overwritten.

The JAX verifier can compare against supplied reference features. Its optional
`--engine reference` adapter requires explicit `--reference-root`,
`--reference-module`, and `--reference-class` arguments and is not a training dependency.

## Reuse an existing motion cache

Set `MOTION_CACHE` to the existing cache directory. This may be the same physical
directory used by another policy/framework. The loader reads its unique
`metadata*.json`, derives the array suffix from that filename, and opens the
corresponding semantic array names using read-only mmap. The source directory,
metadata, array names, permissions and tensor values are never changed.
No symlink view, copy, Teacher execution or motion recomputation is required.

For archived schema names, all native-UV provenance fields must match the shared
contract: context128/ownership78, native S16/stride8/four iterations, original
DA3 geometry, current-camera depth normalization and `raw_xyz_flow_used=false`.
Coordinates, camera, image/grid sizes, horizon and 3D availability remain strict.
Conflicting headers, incompatible contracts, missing required arrays or incorrect
shapes/dtypes are rejected; the loader never substitutes different labels.

Validate the complete training range without scanning the motion tensor contents:

```bash
JAX_PLATFORMS=cpu .venv/bin/python scripts/check_motion_cache_flow2act.py \
  --cache /path/to/existing/motion-cache --data-root data/libero_lerobot \
  --action-horizon 30 --future-horizon 30
```

The command checks required array headers for every episode with complete action
windows and compares their frame counts with dataset metadata. It writes no output
files, loads no Teacher and does not modify the cache. `--no-motion-3d` checks the
2D-only branch without opening 3D arrays.

A cache with 50 future targets and 3D labels supports the default H30 training.
Changing framework, memory length, batch, GA or LR does not require re-export.
History for memory=3 comes from the dataset's RGB/state at t−10 and t−20.
Labels must be regenerated or extended only when their actual dataset alignment,
Teacher/target definitions, geometry or required horizon/3D coverage changes.

## New motion cache format

New exports write `metadata_flow2act.json` specifying
`schema=Flow2Act-motion-v1`, `coordinate=displacement_xy_div_wh`,
`image_hw=[256,256]`, `grid_hw=[32,32]`,
`source_camera=observation.images.image`, sufficient `horizon` (at least 30),
and `has_motion_3d=true` for motion3d.

Each `<suite>/episode_<index:06d>/` contains:

| Array file prefix (append `_flow2act.npy`) | Shape | Dtype |
| --- | --- | --- |
| displacement_normalized | [T,H,2,32,32] | float16/float32 |
| native_visconf | [T,H,2,32,32] | float16/float32 |
| in_bounds | [T,H,32,32] | bool |
| horizon_valid | [T,H] | bool |
| delta_xyz_normalized | [T,H,3,32,32] | float16/float32 |
| valid3d | [T,H,32,32] | bool |

T is episode length and H is the cache horizon (50 for the included exporter).
Targets are cumulative displacement from the anchor to t+1…t+H; default training
selects the first 30 targets. They are not adjacent-frame velocities.
3D displacement uses the current camera coordinate system and depth normalization.
Masks, moving/static balancing, and coordinate scaling must match the loader/loss.

Baseline does not open the Teacher or cache. Static does not open the motion
cache. Motion2d does not open the two 3D arrays. The existing cache is consumed
directly; a cache from another implementation requires an explicit format and
coordinate check before use. The old lowercase `flow2act-motion-v1` schema is
also accepted; new exports use the shared canonical `Flow2Act-motion-v1` spelling.

## Artifact naming and parameter trees

Checkpoints use `config_flow2act.json`, metrics use `metrics_flow2act.jsonl`,
and the auxiliary parameter root is `Flow2Act` in live parameters, EMA, and
optimizer moments. Native OpenPI parameter paths and tensor values are unchanged.
Archived policy parameter roots require the explicit checkpoint importer described
in [the checkpoint guide](CHECKPOINTS_flow2act.md). Existing motion caches use the
read-only naming adapter above, without re-exporting labels. Source resources are
never modified automatically.
