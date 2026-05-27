#!/bin/bash
#SBATCH --job-name=ens_no_white_4090
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=1-2%2
#SBATCH --output=logs/ens_no_white_%A_%a.out
#SBATCH --error=logs/ens_no_white_%A_%a.err

set -euo pipefail

# run_ensemble_without_whitened_compare.sbatch
#
# Minimal reliable comparison run AFTER you change WeightedEnsemble to exclude
# WhitenedCosine / PCAWhitenedCosine / StabilizedWhitenedCosine.
#
# This runs only the two dataset jobs needed for the paper ablation:
#   1. h1h0_final
#   2. lmsys_h1h0_by_cluster_h1keep_with_unormalized_embedding_fixed
#
# Fixed config:
#   alpha      = 0.05
#   tau_mode   = global
#   n_trials   = 50
#   n_train    = 1200
#   n_calib    = 1200
#   n_eval     = 1200
#   hadamard   = true
#
# Compare these results against the previous normal/with-whitened run.

mkdir -p logs

CONDA_ENV_NAME="ec"
CLI_MODULE="${CLI_MODULE:-NeighborCache.region_local_threshold.cli}"
REGION_KEY="${REGION_KEY:-global_cluster}"

ALPHA="${ALPHA:-0.05}"
TAU_MODE="${TAU_MODE:-global}"
N_TRIALS="${N_TRIALS:-50}"
SEED="${SEED:-42}"
N_TRAIN="${N_TRAIN:-1200}"
N_CALIB="${N_CALIB:-1200}"
N_EVAL="${N_EVAL:-1200}"

ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

EXP_ROOT="${EXP_ROOT:-NeighborCache/results/ensemble_without_whitened_${ARRAY_JOB_ID}}"
mkdir -p "${EXP_ROOT}" "${EXP_ROOT}/task_logs"

case "${TASK_ID}" in
  1)
    DATASET_ID="h1h0"
    DATA="NeighborCache/data/h1h0_final.npz"
    ;;
  2)
    DATASET_ID="lmsys"
    DATA="NeighborCache/data/lmsys_h1h0_by_cluster_h1keep_with_unormalized_embedding_fixed.npz"
    ;;
  *)
    echo "No task for SLURM_ARRAY_TASK_ID=${TASK_ID}; exiting."
    exit 0
    ;;
esac

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-NA}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-NA}"
echo "HOSTNAME=$(hostname)"
echo "DATE=$(date --iso-8601=seconds)"
echo "CONDA_ENV_NAME=${CONDA_ENV_NAME}"
echo "DATASET_ID=${DATASET_ID}"
echo "EXP_ROOT=${EXP_ROOT}"

# Load conda in non-interactive Slurm shell.
if command -v conda >/dev/null 2>&1; then
  echo "conda command found"
else
  for conda_sh in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"
  do
    if [[ -f "${conda_sh}" ]]; then
      # shellcheck disable=SC1090
      source "${conda_sh}"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda command not available" >&2
  exit 2
fi

PYTHON_BIN="$(conda run -n "${CONDA_ENV_NAME}" python -c 'import sys; print(sys.executable)' | tail -n 1)"
echo "PYTHON_BIN=${PYTHON_BIN}"

COMMAND=(
  conda run -n "${CONDA_ENV_NAME}" python -m "${CLI_MODULE}"
  --data "${DATA}"
  --region_key "${REGION_KEY}"
  --alpha "${ALPHA}"
  --tau_mode "${TAU_MODE}"
  --n_trials "${N_TRIALS}"
  --seed "${SEED}"
  --n_train "${N_TRAIN}"
  --n_calib "${N_CALIB}"
  --n_eval "${N_EVAL}"
  --hadamard_preprocess
)

COMMAND_STR="${COMMAND[*]}"
STARTED_AT="$(date --iso-8601=seconds)"
TASK_LOG="${EXP_ROOT}/task_logs/${TASK_ID}_${DATASET_ID}.out"
TASK_ERR="${EXP_ROOT}/task_logs/${TASK_ID}_${DATASET_ID}.err"

# Create metadata once.
{
  echo "variant=without_whitened"
  echo "note=Run this after modifying WeightedEnsemble to exclude whitened methods."
  echo "array_job_id=${ARRAY_JOB_ID}"
  echo "created_at=${STARTED_AT}"
  echo "alpha=${ALPHA}"
  echo "tau_mode=${TAU_MODE}"
  echo "n_trials=${N_TRIALS}"
  echo "seed=${SEED}"
  echo "n_train=${N_TRAIN}"
  echo "n_calib=${N_CALIB}"
  echo "n_eval=${N_EVAL}"
  echo "region_key=${REGION_KEY}"
  echo "cli_module=${CLI_MODULE}"
} > "${EXP_ROOT}/run_metadata_${TASK_ID}_${DATASET_ID}.txt"

# Header for status CSV, created safely enough for this tiny 2-task array.
if [[ ! -f "${EXP_ROOT}/task_status.csv" ]]; then
  echo "task_id,dataset_id,variant,alpha,tau_mode,seed,n_train,n_calib,n_eval,n_trials,exit_code,slurm_job_id,slurm_array_job_id,slurm_array_task_id,host,started_at,finished_at,command" \
    > "${EXP_ROOT}/task_status.csv"
fi

echo "Running command:"
printf '%q ' "${COMMAND[@]}"
echo ""
echo "Task log: ${TASK_LOG}"
echo "Task err: ${TASK_ERR}"

set +e
"${COMMAND[@]}" > "${TASK_LOG}" 2> "${TASK_ERR}"
EXIT_CODE=$?
set -e

FINISHED_AT="$(date --iso-8601=seconds)"
HOST="$(hostname)"

# Append status row.
python - <<PY
import csv
from pathlib import Path

path = Path("${EXP_ROOT}/task_status.csv")
row = {
    "task_id": "${TASK_ID}",
    "dataset_id": "${DATASET_ID}",
    "variant": "without_whitened",
    "alpha": "${ALPHA}",
    "tau_mode": "${TAU_MODE}",
    "seed": "${SEED}",
    "n_train": "${N_TRAIN}",
    "n_calib": "${N_CALIB}",
    "n_eval": "${N_EVAL}",
    "n_trials": "${N_TRIALS}",
    "exit_code": "${EXIT_CODE}",
    "slurm_job_id": "${SLURM_JOB_ID:-NA}",
    "slurm_array_job_id": "${SLURM_ARRAY_JOB_ID:-NA}",
    "slurm_array_task_id": "${SLURM_ARRAY_TASK_ID:-NA}",
    "host": "${HOST}",
    "started_at": "${STARTED_AT}",
    "finished_at": "${FINISHED_AT}",
    "command": """${COMMAND_STR}""",
}
fields = [
    "task_id", "dataset_id", "variant", "alpha", "tau_mode", "seed",
    "n_train", "n_calib", "n_eval", "n_trials", "exit_code",
    "slurm_job_id", "slurm_array_job_id", "slurm_array_task_id",
    "host", "started_at", "finished_at", "command",
]
exists = path.exists() and path.stat().st_size > 0
with path.open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    if not exists:
        writer.writeheader()
    writer.writerow(row)
PY

if [[ "${EXIT_CODE}" -ne 0 ]]; then
  echo "Task failed with exit code ${EXIT_CODE}"
  echo "See: ${TASK_ERR}"
  exit "${EXIT_CODE}"
fi

echo "Task completed successfully."
echo "Results root: ${EXP_ROOT}"
