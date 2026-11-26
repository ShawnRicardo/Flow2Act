# LIBERO and LIBERO-Plus evaluation

The release evaluates **Flow2Act checkpoints**, including `baseline`, `static`, `motion2d`, and `motion3d`, with the `Flow2Act-XR0-checkpoint-v1` contract. State/action normalization, query layout, and decision-memory semantics are preserved. Official XR0 weights are not Flow2Act checkpoints. Before using a policy from the development version, explicitly import it with `scripts/import_flow2act_artifacts.py checkpoint --source /path/to/existing-policy.ckpt --output /path/to/policy_flow2act.ckpt`; the source remains untouched.

## Separate environments

Keep three Python environments: the policy environment described in the root README, a LIBERO simulator, and a LIBERO-Plus simulator. Both benchmarks expose a Python package named `libero`; installing both in one environment changes which benchmark is imported.

From this repository root, prepare the benchmark source trees (or set `LIBERO_ROOT` and `LIBERO_PLUS_ROOT` to existing installations):

```bash
mkdir -p third_party
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git clone https://github.com/sylvestf/LIBERO-plus.git third_party/LIBERO-plus

conda create -n flow2act-libero python=3.10 -y
conda run -n flow2act-libero python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
conda run -n flow2act-libero python -m pip install -r requirements/libero.txt
conda run -n flow2act-libero python -m pip install --no-deps -e third_party/LIBERO

conda create -n flow2act-plus python=3.10 -y
conda run -n flow2act-plus python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
conda run -n flow2act-plus python -m pip install -r requirements/libero_plus.txt
conda run -n flow2act-plus python -m pip install --no-deps -e third_party/LIBERO-plus
```

The requirements preserve **MuJoCo 3.3.6 for LIBERO** and **MuJoCo 2.3.7 for LIBERO-Plus**, with robosuite 1.4.0. In the original development environment, vanilla LIBERO used a private `PYTHONPATH` overlay containing MuJoCo 3.3.6 over an environment whose installed package metadata reported 2.3.7. This release replaces that overlay with a direct version pin. The launcher records the versions actually imported by the simulator interpreters; renderer reports also contain the actual version and OpenGL device.

These are different simulator stacks. Do not replace one with the other when comparing results, and do not describe the Flow2Act evaluator as an exact reproduction of the official XR0 evaluation seeding scheme. This publication copy has CPU/CLI tests; a fresh installation still needs the supplied rendering preflight and a short rollout on its target GPU.

LIBERO-Plus needs its separate [benchmark assets](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main). Follow the benchmark README and extract `assets.zip` into `third_party/LIBERO-plus/libero/libero/`. Keep benchmark source, BDDL files, official initial states, the classification JSON, and assets from the same benchmark version. The release does not redistribute those assets. Its additional perturbation dependencies are included in `requirements/libero_plus.txt`; the `Wand` package needs the system ImageMagick/MagickWand library.

Hugging Face, PyTorch, and compilation caches use their standard library defaults; existing environment overrides are inherited unchanged.

Rendering uses NVIDIA EGL, without a display server. The machine/container must provide its NVIDIA driver, EGL/GLVND libraries, and access to the selected GPUs. Simulator configuration is generated under each output directory; it does not depend on `~/.libero` or on a sibling directory outside this repository.

## Full evaluation

Activate the policy environment. Set absolute interpreter paths for the two simulators and select the checkpoint. `PROCESSOR_PATH` is the downloaded XR0 Pretrain processor/config directory; it is explicitly supplied so a checkpoint's old machine-specific path is not used.

```bash
CHECKPOINT=/path/to/step-step=0030000_flow2act.ckpt \
PROCESSOR_PATH=/path/to/Xiaomi-Robotics-0-Pretrain \
LIBERO_PYTHON=/path/to/envs/flow2act-libero/bin/python \
PLUS_PYTHON=/path/to/envs/flow2act-plus/bin/python \
GPU_IDS=0,1,2,3,4,5,6,7 \
BENCHMARKS=libero,libero_plus \
SUITES=libero_spatial,libero_object,libero_goal,libero_10 \
SEED=7 \
OUTPUT_DIR=/path/to/results/eval/flow2act-motion3d \
bash scripts/evaluate_flow2act.sh
```

For four GPUs, use `GPU_IDS=0,1,2,3`. For one benchmark/suite, for example use `BENCHMARKS=libero_plus SUITES=libero_10`. A single GPU is supported; when both benchmarks are selected it runs LIBERO before Plus. With multiple GPUs, the launcher initially splits the devices between benchmarks and transfers completed LIBERO devices to Plus. Episode seeds do not depend on GPU or shard count. Optional `CPU_OFFLOAD_LAYER_MAP=0:12,1:12` moves selected VLM blocks' weights to CPU between forwards; its default is no offload.

`CHECKPOINT` accepts a normal checkpoint file or the existing DeepSpeed checkpoint directory containing `checkpoint/mp_rank_00_model_states.pt`. The saved model contract determines action horizon, execute horizon, future-query horizon, and memory length. The launcher never infers these from the experiment folder name. The static teacher and motion cache are not needed during policy inference.

A fresh run refuses to overwrite an existing evaluation. The default output name includes the checkpoint, benchmark selection, suite selection, and seed; set `OUTPUT_DIR` for explicit organization. Use a directory on your large results volume, rather than committing generated files.

## Flow2Act evaluation protocol

| Setting | LIBERO | LIBERO-Plus |
|---|---|---|
| Suites | Spatial / Object / Goal / LIBERO-10 | Same four suite names |
| Trials | 50 official initial states per task | First official initial state, one trial per perturbed task |
| Full count | 2,000 episodes | 10,030 episodes (LIBERO-10: 2,519) |
| Settling | 10 steps, `[0,0,0,0,0,0,-1]` | 5 steps, raw zero action |
| Control limit | Spatial 220, Object 280, Goal 300, LIBERO-10 520 | 720 |
| Images | Main + wrist, 256×256, both rotated 180° | Same |
| Policy generation | Checkpoint's saved flow-integration count | Same |
| Replanning / memory | Saved execute horizon and current-included decision FIFO | Same |

The policy client uses deterministic task/episode/decision seeds. Plus retains the original suite-dependent seed offset even when only one suite is selected. The worker feeds denormalized signed LIBERO actions without a second normalization. Historical records are real decision observations; control-loop frames between replans are not inserted into memory.

The Plus initialization helper retains its bounded placement recovery, then installs the official initial state. Technical simulator/RPC errors are recorded and stop the launcher; they are excluded from policy failure counts. Policy failures (`success=false`) are completed episodes and are never automatically retried.

## Logs, progress, and explicit continuation

```bash
cat /path/to/results/eval/flow2act-motion3d/progress.txt
tail -f /path/to/results/eval/flow2act-motion3d/log.log
```

Outputs include `progress.json`, human-readable `progress.txt`, per-GPU policy/rollout stdout and stderr in `logs/`, and flushed per-episode `episodes.jsonl`. Plus reports every perturbation category within each suite and across the selected suites. The macro average weights the selected suites equally; the default selection is all four. A final `summary.json` is written only after all requested episodes finish without technical or data errors.

To continue a stopped evaluation, keep the checkpoint, suite/benchmark selection, seed, and simulator versions unchanged. Explicitly point to the old output and choose a **new** destination; GPU count may change:

```bash
CHECKPOINT=/path/to/step-step=0030000_flow2act.ckpt \
PROCESSOR_PATH=/path/to/Xiaomi-Robotics-0-Pretrain \
LIBERO_PYTHON=/path/to/envs/flow2act-libero/bin/python \
PLUS_PYTHON=/path/to/envs/flow2act-plus/bin/python \
CONTINUE_FROM=/path/to/results/eval/flow2act-motion3d \
OUTPUT_DIR=/path/to/results/eval/flow2act-motion3d-continued \
GPU_IDS=0,1,2,3 SEED=7 \
bash scripts/evaluate_flow2act.sh
```

The launcher verifies the source checkpoint/contract/protocol and imports completed episodes, including normal policy failures. New workers skip those identities, resharing only unfinished episodes. Source outputs remain untouched. It rejects malformed, duplicate, partial, incompatible, or smoke completion evidence. Continuation is explicit; there is no automatic retry or background recovery mechanism. The portable launcher expects its own `evaluation_config.json` format, rather than silently merging historical runs from incompatible launchers.

A renderer-only check loads no policy and produces no benchmark scores:

```bash
# Supply CHECKPOINT and interpreter variables as above; choose a fresh OUTPUT_DIR.
bash scripts/evaluate_flow2act.sh --render-check-only
```

Run portable bookkeeping tests without CUDA or either benchmark installed:

```bash
python -m unittest eval_libero.test_release_eval -v
```
