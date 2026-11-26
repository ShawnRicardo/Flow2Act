# LIBERO and LIBERO-Plus evaluation

The launcher runs eight local JAX EMA servers on GPUs 0–7 and distributes the
benchmark across them. Evaluation uses seed 7, 10 denoising steps, prediction
horizon 30, execution horizon 10, and the checkpoint's memory setting.

## External simulator environments

Keep simulator installations separate from the Python 3.11 JAX environment.
Use Python 3.10 environments for LIBERO and LIBERO-Plus, each importing only its
own benchmark checkout. The simulator dependencies are MuJoCo 2.3.7,
robosuite 1.4.0, Gym 0.25.2, and NumPy 1.24.4, as specified in
`requirements/eval_flow2act.txt`.

Install the simulator sources and compatible runtime environments separately
from the JAX environment using the pinned benchmark revisions below:

```bash
sudo apt-get install -y unzip libegl1 libgl1 libglvnd0 libmagickwand-dev
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/libero
git -C third_party/libero checkout f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
conda create -n flow2act-libero python=3.10 -y
conda run -n flow2act-libero pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
conda run -n flow2act-libero pip install -r requirements/eval_flow2act.txt
conda run -n flow2act-libero pip install -e third_party/libero --no-deps
conda run -n flow2act-libero pip install -e packages/openpi-client

git clone https://github.com/sylvestf/LIBERO-plus.git third_party/LIBERO-plus
git -C third_party/LIBERO-plus checkout 4976dc30028e805ff8094b55501d532c48fec182
conda create -n flow2act-plus python=3.10 -y
conda run -n flow2act-plus pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
conda run -n flow2act-plus pip install -r requirements/eval_flow2act.txt
conda run -n flow2act-plus pip install -e third_party/LIBERO-plus --no-deps
conda run -n flow2act-plus pip install -e packages/openpi-client
```

Prepare each benchmark's scene assets, BDDL definitions and initial states using
its upstream instructions. LIBERO-Plus scene assets are distributed separately
at [Sylvest/LIBERO-plus](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main).
The runner requires the 10,030-entry Plus task catalog
`libero/libero/benchmark/task_classification.json`; it rejects a different size.
No simulator or model downloads occur when evaluation starts.

## Launch and logs

```bash
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src"
CHECKPOINT="checkpoints/Flow2Act/checkpoint-30000"
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

Use a fresh output directory. The preexisting `log.log` is allowed, but existing
evaluation records are refused. Pass resource paths and simulator interpreters
explicitly as shown above.
`--classification` overrides the Plus catalog path; `--base-port` defaults to 26800.

The output contains:

- `log.log`: launch status and regular total/suite/perturbation summaries.
- `logs/`: separate JAX server, smoke, and evaluation worker logs.
- `benchmark_configs_flow2act/`: generated per-run benchmark YAML files.
- `libero/`, `libero_plus/`: per-shard, per-suite episode JSONL.
- `evaluation_config.json`, `summary.json`, `progress.txt`: configuration and results.

The launcher runs 20-step real-environment smokes before starting the full
evaluation; smoke episodes do not contribute to reported success rates.
LIBERO covers 40 tasks × 50 initial states = 2,000 episodes. LIBERO-Plus covers 10,030
task/perturbation episodes, one per catalog task. All 40 LIBERO tasks use the suite
order 10→goal→object→spatial. Plus episodes are partitioned across 8 shards; the
first 4 servers run regular LIBERO first, then switch to their Plus shards.

Images preserve full FOV with the fixed simulator-to-dataset 180-degree camera
convention. Training gripper labels remain native; at the simulator boundary,
binary gripper > 0.5 maps to −1, otherwise +1. LIBERO uses 10 settling steps; Plus
uses its prescribed initial state and 5 zero-action steps. Explicit per-episode
flow noise makes the policy RNG independent of shard execution order.
Initialization failures use the existing bounded placement-recovery helper.

## Standalone serving and validation

For an already prepared native checkpoint or imported deployment package:

```bash
.venv/bin/python scripts/serve_policy_flow2act.py \
  --checkpoint checkpoints/Flow2Act/checkpoint-30000 --port 8000
```

The full benchmark command starts its own dedicated evaluation servers; the
standalone server above is for other clients. Servers use the checkpoint's saved
memory setting and isolate histories by environment and episode. New training
defaults to memory=3; importing an older memory=1 policy preserves memory=1.

A source-only CPU suite is available with:

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run --frozen pytest -q
```

The CPU suite covers cache I/O, geometry, checkpoint import, and reduced-model
forward/backward/sampling. Tests requiring prepared datasets use `-m local_data`.
For a short GPU training check, set `STOP_AFTER_UPDATES=20` with a fresh `OUTPUT_DIR`.
