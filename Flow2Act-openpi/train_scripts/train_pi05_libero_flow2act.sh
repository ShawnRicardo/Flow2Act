#!/usr/bin/env bash
set -Eeuo pipefail
umask 000

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

STAGE=${STAGE:-motion3d}
ACTION_HORIZON=${ACTION_HORIZON:-30}
FUTURE_HORIZON=${FUTURE_HORIZON:-30}
EXECUTE_HORIZON=${EXECUTE_HORIZON:-10}
MEMORY_LENGTH=${MEMORY_LENGTH:-3}
NUM_GPUS=${NUM_GPUS:-4}
FSDP_DEVICES=${FSDP_DEVICES:-${NUM_GPUS}}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
GA=${GA:-1}
MAX_STEPS=${MAX_STEPS:-30000}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
LR=${LR:-5e-5}
FINAL_LR=${FINAL_LR:-5e-6}
WARMUP_STEPS=${WARMUP_STEPS:-2000}
WARMUP_LR=${WARMUP_LR:-5e-7}
EMA_DECAY=${EMA_DECAY:-0.999}
SEED=${SEED:-42}
STUDENT_TAPS=${STUDENT_TAPS:-8,11,14,17}
STATIC_WEIGHT=${STATIC_WEIGHT:-0.02}
MOTION2D_WEIGHT=${MOTION2D_WEIGHT:-1.0}
MOTION3D_WEIGHT=${MOTION3D_WEIGHT:-0.1}
MOTION_BETA2D=${MOTION_BETA2D:-0.00390625}
MOTION_BETA3D=${MOTION_BETA3D:-0.01}
MOVING_THRESHOLD=${MOVING_THRESHOLD:-0.00390625}
READOUT_CHUNK=${READOUT_CHUNK:-4}
WORKERS=${WORKERS:-8}
AUGMENT=${AUGMENT:-1}
LOG_INTERVAL=${LOG_INTERVAL:-10}
STOP_AFTER_UPDATES=${STOP_AFTER_UPDATES:-0}
DATA_ROOT=${DATA_ROOT:-data/libero_lerobot}
BASE_PARAMS=${BASE_PARAMS:-checkpoints/openpi-assets/checkpoints/pi05_base/params}
TOKENIZER_PATH=${TOKENIZER_PATH:-checkpoints/big_vision/paligemma_tokenizer.model}
TEACHER_PATH=${TEACHER_PATH:-checkpoints/da3_flow2act/da3_giant_flow2act.safetensors}
# An existing shared native-UV cache can be passed directly; training never exports labels.
MOTION_CACHE=${MOTION_CACHE:-data/libero4mix_motion3d_flow2act}
NORM_STATS=${NORM_STATS:-assets/libero4mix_H${ACTION_HORIZON}_flow2act}
OUTPUT_DIR=${OUTPUT_DIR:-outputs_flow2act/pi05/libero4mix/${STAGE}_mem${MEMORY_LENGTH}_HA${ACTION_HORIZON}_HF${FUTURE_HORIZON}_M${EXECUTE_HORIZON}_gpu${NUM_GPUS}_fsdp${FSDP_DEVICES}_GBS${GLOBAL_BATCH_SIZE}_GA${GA}_lr${LR}-${FINAL_LR}_wu${WARMUP_STEPS}-${WARMUP_LR}_ema${EMA_DECAY}_steps${MAX_STEPS}_save${SAVE_INTERVAL}_seed${SEED}_taps${STUDENT_TAPS//,/-}_w${STATIC_WEIGHT}-${MOTION2D_WEIGHT}-${MOTION3D_WEIGHT}}

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
    CUDA_VISIBLE_DEVICES=0
    for ((card=1; card<NUM_GPUS; card++)); do CUDA_VISIBLE_DEVICES+=",${card}"; done
    export CUDA_VISIBLE_DEVICES
fi
if (( NUM_GPUS < 1 || GA < 1 || GLOBAL_BATCH_SIZE % (NUM_GPUS * GA) != 0 )); then
    printf 'Invalid NUM_GPUS/GA/GBS: %s/%s/%s\n' "${NUM_GPUS}" "${GA}" "${GLOBAL_BATCH_SIZE}" >&2
    exit 2
fi
printf 'JAX single host, one process; CUDA_VISIBLE_DEVICES=%s; GPUs=%s FSDP=%s\n' "${CUDA_VISIBLE_DEVICES}" "${NUM_GPUS}" "${FSDP_DEVICES}"
printf 'STAGE=%s GBS=%s GA=%s microbatch/device=%s HA=%s HF=%s execute=%s memory=%s\n' "${STAGE}" "${GLOBAL_BATCH_SIZE}" "${GA}" "$((GLOBAL_BATCH_SIZE / NUM_GPUS / GA))" "${ACTION_HORIZON}" "${FUTURE_HORIZON}" "${EXECUTE_HORIZON}" "${MEMORY_LENGTH}"
printf 'OUTPUT_DIR=%s\n' "${OUTPUT_DIR}"

if [[ -d ${OUTPUT_DIR} && -n $(find -P "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    printf 'Fresh training refuses nonempty OUTPUT_DIR: %s\n' "${OUTPUT_DIR}" >&2
    exit 2
fi
mkdir -p -- "${OUTPUT_DIR}"
chmod 0777 -- "${OUTPUT_DIR}"
trap 'status=$?; trap - EXIT; if find -P "${OUTPUT_DIR}" -xdev \( -type d -o -type f \) -exec chmod 0777 -- {} +; then :; elif (( status == 0 )); then status=1; fi; exit "${status}"' EXIT

args=(
    --stage "${STAGE}" --action-horizon "${ACTION_HORIZON}" --future-horizon "${FUTURE_HORIZON}"
    --execute-horizon "${EXECUTE_HORIZON}" --memory-length "${MEMORY_LENGTH}"
    --num-gpus "${NUM_GPUS}" --fsdp-devices "${FSDP_DEVICES}" --global-batch-size "${GLOBAL_BATCH_SIZE}" --ga "${GA}"
    --max-steps "${MAX_STEPS}" --save-interval "${SAVE_INTERVAL}" --lr "${LR}" --final-lr "${FINAL_LR}"
    --warmup-steps "${WARMUP_STEPS}" --warmup-lr "${WARMUP_LR}" --ema-decay "${EMA_DECAY}" --seed "${SEED}"
    --student-taps "${STUDENT_TAPS}" --static-weight "${STATIC_WEIGHT}" --motion2d-weight "${MOTION2D_WEIGHT}" --motion3d-weight "${MOTION3D_WEIGHT}"
    --motion-beta2d "${MOTION_BETA2D}" --motion-beta3d "${MOTION_BETA3D}" --moving-threshold "${MOVING_THRESHOLD}"
    --readout-chunk "${READOUT_CHUNK}" --workers "${WORKERS}" --log-interval "${LOG_INTERVAL}" --stop-after-updates "${STOP_AFTER_UPDATES}"
    --data-root "${DATA_ROOT}" --base-params "${BASE_PARAMS}" --tokenizer-path "${TOKENIZER_PATH}"
    --teacher-path "${TEACHER_PATH}" --motion-cache "${MOTION_CACHE}" --norm-stats "${NORM_STATS}" --output-dir "${OUTPUT_DIR}"
)
if [[ ${AUGMENT} == 1 ]]; then args+=(--augment); else args+=(--no-augment); fi
"${REPO_ROOT}/.venv/bin/python" scripts/train_flow2act.py "${args[@]}"
