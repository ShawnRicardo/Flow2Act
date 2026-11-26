#!/usr/bin/env bash
set -Eeuo pipefail
umask 000
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/libero_lerobot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/data/libero4mix_motion3d_flow2act}"
TRACK4WORLD_ROOT="${TRACK4WORLD_ROOT:-${REPO_ROOT}/third_party/Track4World}"
TRACK4WORLD_CHECKPOINTS_ROOT="${TRACK4WORLD_CHECKPOINTS_ROOT:-${REPO_ROOT}/checkpoints/Track4World}"
CHECKPOINT="${CHECKPOINT:-${TRACK4WORLD_CHECKPOINTS_ROOT}/track4world_da3.pth}"
MOTION_3D_ENABLED="${MOTION_3D_ENABLED:-true}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES TRACK4WORLD_CHECKPOINTS_ROOT PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${TRACK4WORLD_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
extra=()
if [[ "${MOTION_3D_ENABLED}" == true ]]; then extra+=(--motion-3d-enabled)
elif [[ "${MOTION_3D_ENABLED}" == false ]]; then extra+=(--no-motion-3d-enabled)
else echo "MOTION_3D_ENABLED must be true/false" >&2;exit 2;fi
exec "${PYTHON}" "${REPO_ROOT}/preprocess/export_motion_flow2act.py" \
 --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}" \
 --track4world-root "${TRACK4WORLD_ROOT}" --checkpoint "${CHECKPOINT}" "${extra[@]}" "$@"
