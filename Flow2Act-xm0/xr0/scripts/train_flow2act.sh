#!/usr/bin/env bash
set -Eeuo pipefail
umask 000

XR0_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
NNODES="${NNODES:-${MLP_WORKER_NUM:-1}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${RESOURCE_GPU:-8}}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 "$((NPROC_PER_NODE - 1))")}"
IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
(( ${#VISIBLE_GPUS[@]} == NPROC_PER_NODE )) || {
    echo "CUDA_VISIBLE_DEVICES count must equal NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
    exit 2
}

export CUDA_VISIBLE_DEVICES TOKENIZERS_PARALLELISM=false
export MLP_WORKER_NUM="${NNODES}" MLP_ROLE_INDEX="${NODE_RANK}"
export PYTHONPATH="${XR0_ROOT}:${XR0_ROOT}/..${PYTHONPATH:+:${PYTHONPATH}}"
TORCHRUN_LOG_DIR="${TORCHRUN_LOG_DIR:-${XR0_ROOT}/results/train/Flow2Act/torchrun_flow2act_node${NODE_RANK}}"
cd "${XR0_ROOT}"

exec torchrun \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    --log-dir "${TORCHRUN_LOG_DIR}" \
    --tee 3 \
    tools/train_flow2act.py \
    "$@"
