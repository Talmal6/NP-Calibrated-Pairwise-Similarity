#!/bin/bash
#SBATCH --job-name=grand2_4090_submit
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/grand2_4090_submit_%j.out
#SBATCH --error=logs/grand2_4090_submit_%j.err

set -euo pipefail

# submit_paper_normal_4090_ec.sbatch
#
# This is a real sbatch file.
# It allocates a 4090 node and then runs the Python experiment launcher.
#
# Note:
# run_paper_experiments_normal.py itself generates and submits a Slurm array.
# Therefore, this job is a launcher job, not one of the experiment jobs.

mkdir -p logs

CONDA_ENV_NAME="ec"
SUITE="${SUITE:-paper}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

echo "Launcher job started"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA}"
echo "HOSTNAME=$(hostname)"
echo "DATE=$(date --iso-8601=seconds)"
echo "CONDA_ENV_NAME=${CONDA_ENV_NAME}"
echo "SUITE=${SUITE}"
echo "MAX_PARALLEL=${MAX_PARALLEL}"

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

echo "Using Python from env:"
conda run -n "${CONDA_ENV_NAME}" python -c 'import sys; print(sys.executable)'

conda run -n "${CONDA_ENV_NAME}" python run_paper_experiments_normal.py \
  --suite "${SUITE}" \
  --conda-env "${CONDA_ENV_NAME}" \
  --partition gpu \
  --gpus rtx_4090:1 \
  --mem 32G \
  --cpus-per-task 4 \
  --time 04:00:00 \
  --job-name grand2_4090 \
  --max-parallel "${MAX_PARALLEL}"

echo "Launcher job finished"
