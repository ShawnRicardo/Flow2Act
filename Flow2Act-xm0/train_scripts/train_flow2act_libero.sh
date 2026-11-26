#!/usr/bin/env bash
set -Eeuo pipefail
umask 000
# Modified for Flow2Act: portable environment and output defaults.
# Parse the whole block before execution so later file edits cannot alter shutdown.
{
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PATH="${ENV_PATH:-${CONDA_PREFIX:-}}"
if [[ -n "${ENV_PATH}" ]]; then export PATH="${ENV_PATH}/bin:${PATH}"; fi
STAGE="${STAGE:-motion3d}"
MEMORY_LENGTH="${MEMORY_LENGTH:-1}"
ACTION_LENGTH="${ACTION_LENGTH:-30}"
FUTURE_HORIZON="${FUTURE_HORIZON:-30}"
EXECUTE_HORIZON="${EXECUTE_HORIZON:-10}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-1e-4}"
MIN_LR="${MIN_LR:-5e-7}"
WARMUP_LR="${WARMUP_LR:-5e-7}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
MAX_STEPS="${MAX_STEPS:-30000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
TRAINING_REPEAT="${TRAINING_REPEAT:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
SEED="${SEED:-42}"
AUGMENT="${AUGMENT:-true}"
GEO_LOSS_WEIGHT="${GEO_LOSS_WEIGHT:-0.02}"
MOTION_2D_LOSS_WEIGHT="${MOTION_2D_LOSS_WEIGHT:-1.0}"
MOTION_3D_LOSS_WEIGHT="${MOTION_3D_LOSS_WEIGHT:-0.1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/libero_lerobot}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${CHECKPOINT_ROOT}/Xiaomi-Robotics-0-Pretrain}"
PROCESSOR_PATH="${PROCESSOR_PATH:-${PRETRAINED_CHECKPOINT}}"
VLM_CONFIG_PATH="${VLM_CONFIG_PATH:-${PROCESSOR_PATH}/config.json}"
NORM_STATS="${NORM_STATS:-${CHECKPOINT_ROOT}/Flow2Act/libero4mix_H${ACTION_LENGTH}_norm_stats_flow2act.json}"
MOTION_CACHE_ROOT="${MOTION_CACHE_ROOT:-${CHECKPOINT_ROOT}/cache/libero4mix_motion3d_flow2act}"
TEACHER_WEIGHTS_DIR="${TEACHER_WEIGHTS_DIR:-${CHECKPOINT_ROOT}/da3_flow2act}"
TEACHER_COMPUTE_DEVICE="${TEACHER_COMPUTE_DEVICE:-cuda}"
TEACHER_CPU_OFFLOAD="${TEACHER_CPU_OFFLOAD:-false}"
TEACHER_MMAP_WEIGHTS="${TEACHER_MMAP_WEIGHTS:-false}"
OFFLOAD_OPTIMIZER="${OFFLOAD_OPTIMIZER:-false}"
MAX_OPTIMIZER_GROUP_ELEMENTS="${MAX_OPTIMIZER_GROUP_ELEMENTS:-0}"
NNODES="${NNODES:-${MLP_WORKER_NUM:-1}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${RESOURCE_GPU:-8}}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"
if [[ "${NNODES}" == 1 ]]; then NODE_RANK="${NODE_RANK:-0}"; MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"; fi
[[ "${STAGE}" == baseline || "${STAGE}" == static || "${STAGE}" == motion2d || "${STAGE}" == motion3d ]] || { echo "STAGE must be baseline/static/motion2d/motion3d" >&2; exit 2; }
for name in MEMORY_LENGTH GLOBAL_BATCH_SIZE GRAD_ACCUM NNODES NPROC_PER_NODE ACTION_LENGTH FUTURE_HORIZON EXECUTE_HORIZON MAX_STEPS SAVE_INTERVAL TRAINING_REPEAT; do
 [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "${name} must be a positive integer" >&2; exit 2; }
done
if [[ "${STAGE}" == baseline || "${STAGE}" == static ]]; then
 (( MEMORY_LENGTH == 1 )) || { echo "baseline/static require MEMORY_LENGTH=1" >&2; exit 2; }
fi
[[ "${NODE_RANK}" =~ ^[0-9]+$ && -n "${MASTER_ADDR}" ]] || { echo "Multi-node launch requires NODE_RANK and MASTER_ADDR" >&2; exit 2; }
(( NODE_RANK < NNODES )) || { echo "Invalid node topology" >&2; exit 2; }
TOTAL_GPUS=$((NNODES*NPROC_PER_NODE))
(( GLOBAL_BATCH_SIZE % (TOTAL_GPUS*GRAD_ACCUM) == 0 )) || { echo "GBS must be divisible by total GPUs * GA" >&2; exit 2; }
BATCH_SIZE=$((GLOBAL_BATCH_SIZE/(TOTAL_GPUS*GRAD_ACCUM)))
# Keep the experiment key even when a human tag is supplied.
EXPERIMENT_KEY="${STAGE}-L${MEMORY_LENGTH}-ha${ACTION_LENGTH}-hf${FUTURE_HORIZON}-m${EXECUTE_HORIZON}-${TOTAL_GPUS}gpu-gbs${GLOBAL_BATCH_SIZE}-ga${GRAD_ACCUM}-lr${LR}-min${MIN_LR}-wlr${WARMUP_LR}-wu${WARMUP_STEPS}-steps${MAX_STEPS}-s${SEED}-fr${TRAINING_REPEAT}-n${NNODES}x${NPROC_PER_NODE}-w${NUM_WORKERS}-aug${AUGMENT}"
if [[ "${STAGE}" != baseline ]]; then EXPERIMENT_KEY+="-geo${GEO_LOSS_WEIGHT}"; fi
if [[ "${STAGE}" == motion2d || "${STAGE}" == motion3d ]]; then EXPERIMENT_KEY+="-m2d${MOTION_2D_LOSS_WEIGHT}"; fi
if [[ "${STAGE}" == motion3d ]]; then EXPERIMENT_KEY+="-m3d${MOTION_3D_LOSS_WEIGHT}"; fi
EXP_NAME="${EXPERIMENT_KEY}${EXP_NAME:+-${EXP_NAME}}"
[[ "${EXP_NAME}" != */* && ${#EXP_NAME} -le 255 ]] || { echo "Experiment name must be a single path component of at most 255 characters" >&2; exit 2; }
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULTS_ROOT:-${REPO_ROOT}/results}/train/Flow2Act/libero4mix}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${EXP_NAME}}"
[[ "${OUTPUT_DIR}" == /* ]] || OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
# These values determine the experiment directory. Override their environment
# variables instead of passing a conflicting Hydra value after the defaults.
for argument in "$@"; do
 key="${argument%%=*}"; key="${key#+}"; key="${key#+}"; key="${key#\~}"
 case "${key}" in
  trainer.output_dir|trainer.exp_name|hydra.run.dir|hydra.sweep.*|--multirun|-m|\
  trainer.accumulate_grad_batches|trainer.num_nodes|trainer.max_steps|trainer.seed|trainer.scheduler.params.*|\
  data.params.train_datasets.batch_size|data.params.train_datasets.action_length|data.params.train_datasets.augment|\
  model.params.expected_world_size|model.params.visualization.*|model.params.model.stage|\
  model.params.model.memory_length|model.params.model.action_shape|model.params.model.future_horizon|\
  model.params.model.execute_horizon|model.params.model.training_repeat|model.params.model.*loss_weight)
   echo "Use the corresponding environment variable for ${key}; Hydra overrides would invalidate the output directory or training settings" >&2
   exit 2 ;;
 esac
done
# Main policy training never creates feature/depth/motion visualization files.
VIZ_EVERY_STEPS=0
LOG_FILE="${OUTPUT_DIR}/log.log"
CONSOLE_LOG="${OUTPUT_DIR}/console_node${NODE_RANK:-0}_flow2act.log"
# Node zero owns the shared output. Lock its existing console file, rather than
# creating marker files; torchrun rendezvous keeps other nodes from training
# until node zero has accepted the directory. Other nodes may log bootstrap.
if (( NODE_RANK == 0 )); then
 check_output_directory() {
  python - "${OUTPUT_DIR}" "${NNODES}" "${ACTION_LENGTH}" "${REPO_ROOT}" "${NORM_STATS}" <<'PY'
import json, re, sys
from pathlib import Path
directory, nodes, action, repo, stats_path = sys.argv[1:]
stats_path = Path(stats_path)
if not stats_path.is_absolute():
    stats_path = Path(repo) / stats_path
if stats_path.exists():
    stats = json.loads(stats_path.read_text())
    for key in ('mean', 'std'):
        values = stats.get(key)
        if not isinstance(values, list) or len(values) != int(action) or any(
            not isinstance(row, list) or len(row) != 32 for row in values
        ):
            sys.exit(f'Action statistics mismatch: {stats_path} must contain {key} shaped ({action}, 32). Use statistics computed for ACTION_LENGTH={action}; do not reuse another horizon.')
directory = Path(directory)
for item in directory.iterdir() if directory.is_dir() else ():
    if item.name == 'log.log' and item.is_file() and item.stat().st_size == 0:
        continue
    match = re.fullmatch(r'(?:console_node|torchrun_node)([1-9][0-9]*)_flow2act(?:\.log)?', item.name)
    if match and 0 < int(match[1]) < int(nodes):
        continue
    sys.exit(f'Output directory is not empty: {directory}. Use a different OUTPUT_DIR/EXP_NAME.')
PY
 }
 # Validate before opening a file, then again under the lock to close races.
 if ! check_output_directory; then exit 2; fi
 mkdir -p "${OUTPUT_DIR}"
 exec {OUTPUT_LOG_FD}>>"${LOG_FILE}"
 flock -n "${OUTPUT_LOG_FD}" || { echo "Output directory is already owned by a running job: ${OUTPUT_DIR}" >&2; exit 2; }
 if ! check_output_directory; then exit 2; fi
fi
mkdir -p "${OUTPUT_DIR}"
LOG_FILES=("${CONSOLE_LOG}")
if (( NODE_RANK == 0 )); then LOG_FILES+=("${LOG_FILE}"); fi
touch "${LOG_FILES[@]}"
# Ensure shared output permissions; umask alone creates ordinary files as 666.
set_output_permissions() {
 find -H "${OUTPUT_DIR}" \( -type f -o -type d \) -exec chmod 00777 -- {} +
}
finish_output_permissions() {
 local status=$?
 trap - EXIT
 set +e
 set_output_permissions 2>&1 | tee -a "${LOG_FILES[@]}"
 local permission_status=("${PIPESTATUS[@]}")
 if (( status != 0 )); then exit "${status}"; fi
 if (( permission_status[0] != 0 )); then exit "${permission_status[0]}"; fi
 exit "${permission_status[1]}"
}
trap finish_output_permissions EXIT
set_output_permissions
set +e
(
set -e
# Hydra path values need literal quotes, including checkpoint names containing '='.
hydra_string() {
 python -c 'import sys; from hydra.core.override_parser.types import QuotedString, Quote; print(QuotedString(sys.argv[1], Quote.double).with_quotes())' "$1"
}

for name in DATA_ROOT CHECKPOINT_ROOT PRETRAINED_CHECKPOINT PROCESSOR_PATH VLM_CONFIG_PATH NORM_STATS MOTION_CACHE_ROOT TEACHER_WEIGHTS_DIR; do
 [[ "${!name}" == /* ]] || printf -v "${name}" '%s/%s' "${REPO_ROOT}" "${!name}"
done
# Other nodes wait in torchrun rendezvous until node zero finishes preparation.
if (( NODE_RANK == 0 )) && [[ ! -f "${NORM_STATS}" ]]; then
 echo "Computing missing action statistics for ACTION_LENGTH=${ACTION_LENGTH}: ${NORM_STATS}"
 python "${REPO_ROOT}/xr0/tools/compute_libero_stats_flow2act.py" \
  --data-root "${DATA_ROOT}" --action-length "${ACTION_LENGTH}" --output "${NORM_STATS}" --if-missing
fi
export NNODES NPROC_PER_NODE NODE_RANK MASTER_ADDR MASTER_PORT
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 PYTHONFAULTHANDLER=1
export TORCHRUN_LOG_DIR="${OUTPUT_DIR}/torchrun_node${NODE_RANK}_flow2act"
OPTIMIZER_OVERRIDES=()
if [[ "${OFFLOAD_OPTIMIZER}" == true ]]; then
 OPTIMIZER_OVERRIDES+=("+trainer.strategy.params.offload_optimizer=true" "trainer.optimizer.type=deepspeed.ops.adam.DeepSpeedCPUAdam")
elif [[ "${OFFLOAD_OPTIMIZER}" != false ]]; then echo "OFFLOAD_OPTIMIZER must be true or false" >&2; exit 2; fi
echo "Started $(date -u '+%Y-%m-%d %H:%M:%S UTC'); stage=${STAGE}; memory including current=${MEMORY_LENGTH}; HA=${ACTION_LENGTH}; HF=${FUTURE_HORIZON}; M=${EXECUTE_HORIZON}"
echo "GBS=${GLOBAL_BATCH_SIZE}; total_GPUs=${TOTAL_GPUS}; GA=${GRAD_ACCUM}; batch/GPU=${BATCH_SIZE}; flow_repeat=${TRAINING_REPEAT}; steps=${MAX_STEPS}; LR=${LR}"
echo "pretrained=${PRETRAINED_CHECKPOINT}; data=${DATA_ROOT}; stats=${NORM_STATS}; output=${OUTPUT_DIR}"
echo "Motion cache is read only by motion stages: ${MOTION_CACHE_ROOT}; baseline does not construct DA3"
exec bash "${REPO_ROOT}/xr0/scripts/train_flow2act.sh" \
 "data.params.train_datasets.root=$(hydra_string "${DATA_ROOT}")" "data.params.train_datasets.stats_path=$(hydra_string "${NORM_STATS}")" \
 "data.params.train_datasets.motion_cache_root=$(hydra_string "${MOTION_CACHE_ROOT}")" "data.params.train_datasets.batch_size=${BATCH_SIZE}" \
 "data.params.train_datasets.action_length=${ACTION_LENGTH}" "data.params.train_datasets.augment=${AUGMENT}" \
 "data.params.processor_path=$(hydra_string "${PROCESSOR_PATH}")" "data.params.num_workers=${NUM_WORKERS}" "data.params.prefetch_factor=${PREFETCH_FACTOR}" \
 "model.params.pretrained=$(hydra_string "${PRETRAINED_CHECKPOINT}")" "model.params.teacher_weights_dir=$(hydra_string "${TEACHER_WEIGHTS_DIR}")" \
 "model.params.teacher_compute_device=${TEACHER_COMPUTE_DEVICE}" "model.params.teacher_cpu_offload=${TEACHER_CPU_OFFLOAD}" "model.params.teacher_mmap_weights=${TEACHER_MMAP_WEIGHTS}" \
 "model.params.max_optimizer_group_elements=${MAX_OPTIMIZER_GROUP_ELEMENTS}" "model.params.expected_world_size=${TOTAL_GPUS}" \
 "model.params.visualization.every_n_steps=0" "model.params.visualization.output_dir=null" \
 "model.params.model.vlm_config_path=$(hydra_string "${VLM_CONFIG_PATH}")" "model.params.model.stage=${STAGE}" \
 "model.params.model.action_shape=[${ACTION_LENGTH},32]" "model.params.model.future_horizon=${FUTURE_HORIZON}" \
 "model.params.model.memory_length=${MEMORY_LENGTH}" "model.params.model.execute_horizon=${EXECUTE_HORIZON}" \
 "model.params.model.training_repeat=${TRAINING_REPEAT}" "model.params.model.geo_loss_weight=${GEO_LOSS_WEIGHT}" \
 "model.params.model.motion_2d_loss_weight=${MOTION_2D_LOSS_WEIGHT}" "model.params.model.motion_3d_loss_weight=${MOTION_3D_LOSS_WEIGHT}" \
 "trainer.output_dir=$(hydra_string "${OUTPUT_DIR}")" "trainer.exp_name=$(hydra_string "${EXP_NAME}")" \
 "trainer.scheduler.params.max_lr=${LR}" "trainer.scheduler.params.min_lr=${MIN_LR}" "trainer.scheduler.params.warmup_lr_start=${WARMUP_LR}" \
 "trainer.scheduler.params.num_warmup_steps=${WARMUP_STEPS}" "trainer.max_steps=${MAX_STEPS}" "trainer.save_interval=${SAVE_INTERVAL}" \
 "trainer.accumulate_grad_batches=${GRAD_ACCUM}" "trainer.seed=${SEED}" "trainer.num_nodes=${NNODES}" \
 "hydra.run.dir=$(hydra_string "${OUTPUT_DIR}/hydra")" ${OPTIMIZER_OVERRIDES[@]+"${OPTIMIZER_OVERRIDES[@]}"} "$@"
) 2>&1 | tee -a "${LOG_FILES[@]}"
LAUNCH_STATUS=("${PIPESTATUS[@]}")
if (( LAUNCH_STATUS[0] != 0 )); then exit "${LAUNCH_STATUS[0]}"; fi
exit "${LAUNCH_STATUS[1]}"
}
