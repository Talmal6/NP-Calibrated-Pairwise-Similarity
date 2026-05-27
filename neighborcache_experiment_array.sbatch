#!/usr/bin/env bash
#SBATCH --job-name=ncache_exp
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

# ============================================================
# Slurm array worker for NeighborCache paper experiments
# ============================================================
#
# Do not usually run this file directly.
# Submit through:
#   ./submit_neighborcache_paper_experiments.sh
#
# Required exported variables:
#   COMMANDS_FILE
#   COMMANDS_TSV
#   FAILED_TSV
#   EXP_ROOT
#
# Optional:
#   ENV_SETUP
#
# ============================================================

if [[ -z "${COMMANDS_FILE:-}" ]]; then
  echo "ERROR: COMMANDS_FILE is not set."
  exit 2
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set. This script should run as a Slurm array job."
  exit 2
fi

TASK_ID="${SLURM_ARRAY_TASK_ID}"
CMD="$(sed -n "${TASK_ID}p" "${COMMANDS_FILE}")"

if [[ -z "${CMD}" ]]; then
  echo "ERROR: Empty command for task ${TASK_ID}."
  exit 2
fi

mkdir -p "${EXP_ROOT}/task_logs"

TASK_LOG="${EXP_ROOT}/task_logs/task_${TASK_ID}.log"

echo "============================================================"
echo "NeighborCache Slurm task"
echo "task_id=${TASK_ID}"
echo "job_id=${SLURM_JOB_ID:-NA}"
echo "array_job_id=${SLURM_ARRAY_JOB_ID:-NA}"
echo "hostname=$(hostname)"
echo "date_start=$(date --iso-8601=seconds)"
echo "pwd=$(pwd)"
echo "commands_file=${COMMANDS_FILE}"
echo "task_log=${TASK_LOG}"
echo "============================================================"
echo "Command:"
echo "${CMD}"
echo "============================================================"

if [[ -n "${ENV_SETUP:-}" ]]; then
  echo "Running ENV_SETUP:"
  echo "${ENV_SETUP}"
  # shellcheck disable=SC1090
  eval "${ENV_SETUP}"
fi

set +e
bash -lc "${CMD}" 2>&1 | tee "${TASK_LOG}"
EXIT_CODE="${PIPESTATUS[0]}"
set -e

echo "============================================================"
echo "date_end=$(date --iso-8601=seconds)"
echo "exit_code=${EXIT_CODE}"
echo "============================================================"

if [[ "${EXIT_CODE}" -ne 0 ]]; then
  # Try to append a useful failure row.
  # The submitter's commands.tsv has a header, so line TASK_ID+1 corresponds to this task.
  META_LINE="$(sed -n "$((TASK_ID + 1))p" "${COMMANDS_TSV}" || true)"
  echo -e "${META_LINE}\t${EXIT_CODE}" >> "${FAILED_TSV}"
fi

exit "${EXIT_CODE}"
