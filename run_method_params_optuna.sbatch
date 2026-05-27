#!/bin/bash
#SBATCH --job-name=method_params_optuna
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/method_params_optuna_%j.out
#SBATCH --error=logs/method_params_optuna_%j.err

set -euo pipefail

mkdir -p logs

CONDA_ENV_NAME="ec"
PYTHON_BIN="/home/malmatal/.conda/envs/ec/bin/python"

echo "Method Optuna launcher started"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA}"
echo "HOSTNAME=$(hostname)"
echo "DATE=$(date --iso-8601=seconds)"
echo "CONDA_ENV_NAME=${CONDA_ENV_NAME}"
echo "PYTHON_BIN=${PYTHON_BIN}"

"${PYTHON_BIN}" -m NeighborCache.region_local_threshold.optuna_methods \
  --data NeighborCache/data/h1h0_final.npz \
  --region_key global_cluster \
  --alpha 0.05 \
  --tau_mode global \
  --n_trials 8 \
  --seed 42 \
  --n_train 1270 \
  --n_calib 1240 \
  --n_eval 1270 \
  --hadamard_preprocess

echo "Method Optuna launcher finished"