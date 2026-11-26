# Run Flow2Act on LIBERO (Simulation)

> This example trains **Flow2Act** on **LIBERO** and evaluates it on **LIBERO** and **LIBERO-Plus**.<br>
> We use **OpenPI π0.5 (JAX)** as the base model.

Use `Flow2Act-openpi/` as the root directory for all steps below.

```bash
cd ./Flow2Act-openpi  # If your terminal is currently in the parent directory
```

## 1. Environmental Setup

### 1.1 Policy Environment

Use Linux with NVIDIA GPUs and Python 3.11:

```bash
sudo apt-get install -y ffmpeg
pip install uv
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src"
```

Policy training, online DA3 feature extraction, and inference run in JAX.

### 1.2 Preprocessing Environment

Motion cache export uses a separate PyTorch environment for Track4World. If you
already have compatible extracted DA3 weights and a motion cache, skip this
environment and follow section 2.3.

```bash
conda create -n flow2act-preprocess python=3.11 -y
conda run -n flow2act-preprocess pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
conda run -n flow2act-preprocess pip install -r requirements/preprocess_flow2act.txt
```

## 2. Data and Checkpoint Preparation

### 2.1 LIBERO Dataset and Pretrained Weights

Prepare the four LIBERO datasets in LeRobot v2.1 format from
[IPEC-COMMUNITY's LIBERO collection](https://huggingface.co/collections/IPEC-COMMUNITY/libero-benchmark-dataset).
Existing dataset directories can be reused through `DATA_ROOT`.
Prepare the official **JAX Orbax π0.5 base parameters** and PaliGemma tokenizer from
[OpenPI](https://github.com/Physical-Intelligence/openpi).

To prepare the DA3 weights and motion cache, download [Track4World](https://huggingface.co/TencentARC/Track4World/tree/main)
(`track4world_da3.pth`), [DA3 Nested](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1/tree/main)
(`config.json` and `model.safetensors`), and
[ConvNeXt Tiny](https://download.pytorch.org/models/convnext_tiny-983f1562.pth).
Arrange these local resources as follows:

```text
data/libero_lerobot/
  libero_10_no_noops_1.0.0_lerobot/
  libero_goal_no_noops_1.0.0_lerobot/
  libero_object_no_noops_1.0.0_lerobot/
  libero_spatial_no_noops_1.0.0_lerobot/
checkpoints/
  openpi-assets/checkpoints/pi05_base/params/
  big_vision/paligemma_tokenizer.model
  Track4World/
    track4world_da3.pth
    DA3NESTED-GIANT-LARGE-1.1/{config.json,model.safetensors}
    torchvision/convnext_tiny-983f1562.pth
```

### 2.2 Weight Extraction and Motion Cache Export

Extract the DA3 weights and export the motion cache:

```bash
conda run --no-capture-output -n flow2act-preprocess python preprocess/extract_da3_weights_flow2act.py \
  --checkpoint checkpoints/Track4World/track4world_da3.pth \
  --output-dir checkpoints/da3_flow2act
conda run --no-capture-output -n flow2act-preprocess bash preprocess/export_motion_flow2act.sh
```

This creates `checkpoints/da3_flow2act/da3_giant_flow2act.safetensors` and
`data/libero4mix_motion3d_flow2act/`, including 2D/3D targets and validity masks.
See the [resource guide](docs/RESOURCES_flow2act.md) for resource formats and preparation details.

### 2.3 Reuse Existing Resources (Optional)

To use existing compatible resources, set their paths before training:

```bash
export DATA_ROOT="/path/to/existing/libero_lerobot"
export MOTION_CACHE="/path/to/existing/motion-cache"
export TEACHER_PATH="/path/to/existing/da3-giant.safetensors"
JAX_PLATFORMS=cpu .venv/bin/python scripts/check_motion_cache_flow2act.py \
  --cache "$MOTION_CACHE" --data-root "$DATA_ROOT"
```

The cache must cover the training dataset and provide at least 30 future steps
with 3D labels for the default configuration. See the resource guide for compatibility requirements.

## 3. Training

```bash
GLOBAL_BATCH_SIZE=256 GA=1 \
NUM_GPUS=4 FSDP_DEVICES=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_STEPS=30000 SAVE_INTERVAL=5000 \
bash train_scripts/train_pi05_libero_flow2act.sh
```

To resume, specify a complete training checkpoint. The saved configuration,
optimizer state, and normalization statistics are restored:

```bash
RESUME_CHECKPOINT="outputs_flow2act/pi05/libero4mix/<run>/checkpoints/checkpoint-20000" \
bash train_scripts/resume_pi05_libero_flow2act.sh
```

## 4. Inference and Evaluation

Use a checkpoint trained with this project directly for inference and evaluation.
Inference loads the EMA policy, normalization statistics, and local tokenizer;
DA3 weights and motion caches are only needed during training. To import an
existing local JAX policy, follow the [checkpoint guide](docs/CHECKPOINTS_flow2act.md).
Imported EMA-only packages support inference; resume requires a full training checkpoint.

Install the separate LIBERO and LIBERO-Plus environments and assets using the
[evaluation guide](docs/EVALUATION_flow2act.md), then evaluate all four suites on
eight GPUs: **2,000 LIBERO + 10,030 LIBERO-Plus episodes**.

```bash
CHECKPOINT="outputs_flow2act/pi05/libero4mix/<run>/checkpoints/checkpoint-30000"
EVAL_OUTPUT="results/eval/Flow2Act/pi05/checkpoint-30000-libero-family-seed7-8gpu"
SIM_ENV_ROOT="$(conda info --base)/envs"
mkdir -p "$EVAL_OUTPUT"
set -o pipefail
.venv/bin/python -u scripts/eval_libero_family_flow2act.py \
  --checkpoint "$CHECKPOINT" --output-dir "$EVAL_OUTPUT" \
  --libero-python "$SIM_ENV_ROOT/flow2act-libero/bin/python" \
  --plus-python "$SIM_ENV_ROOT/flow2act-plus/bin/python" \
  --libero-root third_party/libero --plus-root third_party/LIBERO-plus \
  2>&1 | tee -a "$EVAL_OUTPUT/log.log"
```

See the [evaluation guide](docs/EVALUATION_flow2act.md) for standalone policy serving
and result files, [experiment details](docs/FLOW2ACT.md) for model and training
settings, and [third-party notices](THIRD_PARTY_NOTICES.md) for attribution.
