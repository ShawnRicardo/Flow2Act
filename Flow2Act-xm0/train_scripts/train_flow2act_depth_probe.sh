#!/usr/bin/env bash
{
set -Eeuo pipefail
umask 000
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PATH="${ENV_PATH:-${CONDA_PREFIX:-}}"
if [[ -n "${ENV_PATH}" ]]; then export PATH="${ENV_PATH}/bin:${PATH}"; fi
export PYTHONPATH="${REPO_ROOT}/xr0:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:?Set POLICY_CHECKPOINT to a trained Flow2Act checkpoint}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/libero_lerobot}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints}"
PROCESSOR_PATH="${PROCESSOR_PATH:-${CHECKPOINT_ROOT}/Xiaomi-Robotics-0-Pretrain}"
TEACHER_WEIGHTS_DIR="${TEACHER_WEIGHTS_DIR:-${CHECKPOINT_ROOT}/da3_flow2act}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/probe/Flow2Act/DepthOnly}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-5000}"
LR="${LR:-1e-4}"
MIN_LR="${MIN_LR:-1e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
CLIP_GRAD_NORM="${CLIP_GRAD_NORM:-1.0}"
SI_LOG_WEIGHT="${SI_LOG_WEIGHT:-1.0}"
GRADIENT_WEIGHT="${GRADIENT_WEIGHT:-0.5}"
LOG_DEPTH_WEIGHT="${LOG_DEPTH_WEIGHT:-0.1}"
SEED="${SEED:-42}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-500}"
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-10}"
NNODES="${NNODES:-${MLP_WORKER_NUM:-1}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${RESOURCE_GPU:-8}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 "$((NPROC_PER_NODE - 1))")}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29501}}"
if [[ "${NNODES}" == 1 ]]; then
 NODE_RANK="${NODE_RANK:-0}"
 MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
cd "${REPO_ROOT}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
for name in NNODES NPROC_PER_NODE GLOBAL_BATCH_SIZE GRAD_ACCUM; do
 [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "${name} must be a positive integer" >&2; exit 2; }
done
[[ "${NODE_RANK}" =~ ^[0-9]+$ && -n "${MASTER_ADDR}" ]] || { echo "Multi-node probe requires NODE_RANK and MASTER_ADDR" >&2; exit 2; }
(( NODE_RANK < NNODES )) || { echo "Invalid node rank" >&2; exit 2; }
TOTAL_GPUS=$((NNODES*NPROC_PER_NODE))
(( GLOBAL_BATCH_SIZE % (TOTAL_GPUS*GRAD_ACCUM) == 0 )) || { echo "GBS must be divisible by total GPUs * GA" >&2; exit 2; }
BATCH_PER_GPU=$((GLOBAL_BATCH_SIZE/(TOTAL_GPUS*GRAD_ACCUM)))
export NNODES NPROC_PER_NODE NODE_RANK MASTER_ADDR MASTER_PORT
TRAIN_ARGS=(
 --policy-checkpoint "${POLICY_CHECKPOINT}" --data-root "${DATA_ROOT}" --processor-path "${PROCESSOR_PATH}"
 --teacher-weights-dir "${TEACHER_WEIGHTS_DIR}" --output-root "${OUTPUT_ROOT}" --output-dir "${OUTPUT_DIR}"
 --global-batch-size "${GLOBAL_BATCH_SIZE}" --grad-accum "${GRAD_ACCUM}" --max-steps "${MAX_STEPS}"
 --lr "${LR}" --min-lr "${MIN_LR}" --warmup-steps "${WARMUP_STEPS}" --weight-decay "${WEIGHT_DECAY}"
 --clip-grad-norm "${CLIP_GRAD_NORM}" --si-log-weight "${SI_LOG_WEIGHT}" --gradient-weight "${GRADIENT_WEIGHT}" --log-depth-weight "${LOG_DEPTH_WEIGHT}"
 --seed "${SEED}" --val-fraction "${VAL_FRACTION}" --val-every "${VAL_EVERY_STEPS}" --val-max-batches "${VAL_MAX_BATCHES}"
 --save-every "${SAVE_INTERVAL}" --num-workers "${NUM_WORKERS}" --log-every "${LOG_EVERY_STEPS}"
)
ENTRY="${REPO_ROOT}/xr0/tools/train_flow2act_depth_probe.py"
# Resolve the real policy step from its checkpoint contract before creating logs.
# The preparation pass only reads metadata and checks the destination.
OUTPUT_DIR="$(python "${ENTRY}" --prepare-output "${TRAIN_ARGS[@]}" "$@")"
CONSOLE_LOG="${OUTPUT_DIR}/console_node${NODE_RANK}_DepthOnly_flow2act.log"
# Hold the output lock on the console file.
exec {OUTPUT_LOG_FD}>>"${CONSOLE_LOG}"
flock -n "${OUTPUT_LOG_FD}" || { echo "Output is already owned by a running node: ${OUTPUT_DIR}" >&2; exit 2; }
LOG_FILES=("${CONSOLE_LOG}")
if (( NODE_RANK == 0 )); then LOG_FILES+=("${OUTPUT_DIR}/log.log"); fi
set +e
(
set -e
echo "Depth-only adapter: frozen policy=${POLICY_CHECKPOINT}; frozen DA3 backbone/head; output=${OUTPUT_DIR}"
echo "GBS=${GLOBAL_BATCH_SIZE} = total_GPUs=${TOTAL_GPUS} * batch/GPU=${BATCH_PER_GPU} * GA=${GRAD_ACCUM}; lr=${LR}, steps=${MAX_STEPS}"
echo "Loss = ${SI_LOG_WEIGHT} * SI-log + ${GRADIENT_WEIGHT} * multiscale gradient + ${LOG_DEPTH_WEIGHT} * log-depth"
exec torchrun --nnodes="${NNODES}" --nproc_per_node="${NPROC_PER_NODE}" --node_rank="${NODE_RANK}" \
 --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" --tee 3 \
 --log-dir "${OUTPUT_DIR}/torchrun_node${NODE_RANK}_DepthOnly_flow2act" \
 "${ENTRY}" "${TRAIN_ARGS[@]}" "$@" --output-dir "${OUTPUT_DIR}"
) 2>&1 | tee -a "${LOG_FILES[@]}"
STATUS=("${PIPESTATUS[@]}")
if (( STATUS[0] != 0 )); then exit "${STATUS[0]}"; fi
exit "${STATUS[1]}"
}
