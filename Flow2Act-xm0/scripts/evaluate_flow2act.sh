#!/usr/bin/env bash
{
set -Eeuo pipefail
umask 000
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_PYTHON="${POLICY_PYTHON:-python}"
SIMULATOR_ENVS_DIR="$(dirname -- "${CONDA_PREFIX:-${REPO_ROOT}/.venvs/flow2act}")"
LIBERO_PYTHON="${LIBERO_PYTHON:-${SIMULATOR_ENVS_DIR}/flow2act-libero/bin/python}"
PLUS_PYTHON="${PLUS_PYTHON:-${SIMULATOR_ENVS_DIR}/flow2act-plus/bin/python}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results}"
CHECKPOINT="${CHECKPOINT:-${CHECKPOINT_ROOT}/Flow2Act/policy_flow2act.ckpt}"
MODEL_DIR="${MODEL_DIR:-$(dirname -- "${CHECKPOINT}")}"
PROCESSOR_PATH="${PROCESSOR_PATH:-${CHECKPOINT_ROOT}/Xiaomi-Robotics-0-Pretrain}"
LIBERO_ROOT="${LIBERO_ROOT:-${REPO_ROOT}/third_party/LIBERO}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-${REPO_ROOT}/third_party/LIBERO-plus}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
BENCHMARKS="${BENCHMARKS:-libero,libero_plus}"
SUITES="${SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"
SEED="${SEED:-7}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/eval/Flow2Act/$(basename -- "${MODEL_DIR}")-$(basename -- "${CHECKPOINT}")-${BENCHMARKS//,/-}-${SUITES//,/-}-seed${SEED}}"
CONTINUE_FROM="${CONTINUE_FROM:-}"
CPU_OFFLOAD_LAYER_MAP="${CPU_OFFLOAD_LAYER_MAP:-}"
BASE_PORT="${BASE_PORT:-28050}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"
RENDER_CHECK_TIMEOUT="${RENDER_CHECK_TIMEOUT:-600}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-1200}"
mkdir -p -- "${OUTPUT_DIR}"
cd -- "${REPO_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
EXTRA_ARGS=()
if [[ -n "${CONTINUE_FROM}" ]]; then EXTRA_ARGS+=(--continue-from "${CONTINUE_FROM}"); fi
set +e
"${POLICY_PYTHON}" -u eval_libero/launch_flow2act.py \
  --model-dir "${MODEL_DIR}" --checkpoint "${CHECKPOINT}" --processor-path "${PROCESSOR_PATH}" \
  --server-python "${POLICY_PYTHON}" --libero-python "${LIBERO_PYTHON}" --plus-python "${PLUS_PYTHON}" \
  --libero-root "${LIBERO_ROOT}" --libero-plus-root "${LIBERO_PLUS_ROOT}" \
  --gpu-ids "${GPU_IDS}" --benchmarks "${BENCHMARKS}" --suites "${SUITES}" --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}" --base-port "${BASE_PORT}" --progress-interval "${PROGRESS_INTERVAL}" \
  --cpu-offload-layer-map "${CPU_OFFLOAD_LAYER_MAP}" --render-check-timeout "${RENDER_CHECK_TIMEOUT}" \
  --server-start-timeout "${SERVER_START_TIMEOUT}" "${EXTRA_ARGS[@]}" "$@" 2>&1 | tee -a "${OUTPUT_DIR}/log.log"
STATUS=("${PIPESTATUS[@]}")
if (( STATUS[0] != 0 )); then exit "${STATUS[0]}"; fi
exit "${STATUS[1]}"
}
