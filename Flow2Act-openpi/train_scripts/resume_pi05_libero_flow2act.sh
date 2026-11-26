#!/usr/bin/env bash
set -Eeuo pipefail
umask 000

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RESUME_CHECKPOINT=${RESUME_CHECKPOINT:?Set RESUME_CHECKPOINT to an explicit checkpoints/checkpoint-N directory}
INCOMPLETE_CHECKPOINT=${INCOMPLETE_CHECKPOINT:-}

args=(--resume-checkpoint "${RESUME_CHECKPOINT}")
if [[ -n ${INCOMPLETE_CHECKPOINT} ]]; then
    args+=(--incomplete-checkpoint "${INCOMPLETE_CHECKPOINT}")
fi
exec "${REPO_ROOT}/.venv/bin/python" scripts/resume_flow2act.py "${args[@]}" "$@"
