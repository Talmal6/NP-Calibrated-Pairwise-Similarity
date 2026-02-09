#!/bin/bash
#SBATCH --job-name=grand2_4090
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/grand2_4090_%j.out
#SBATCH --error=logs/grand2_4090_%j.err

mkdir -p logs

echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ec

cd ~/Projects/Multi_Dim_Threshold

python grand2_bechmark_samples.py \
  --pkl quora_question_pairs_with_embeddings.pkl \
  --alpha 0.05 \
  --n_trials 1 \
  --n_list 100,200,500,1000,2000,4000,8000 \
  --d 1024

echo "End: $(date)"
