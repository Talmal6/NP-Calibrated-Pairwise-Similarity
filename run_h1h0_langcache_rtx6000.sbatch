#!/bin/sh
#SBATCH -J h1h0_c1254
#SBATCH -p gpu
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -e

cd /home/malmatal/Projects/Multi_Dim_Threshold
mkdir -p logs

# Runtime knobs for stability on GPU
export TORCHDYNAMO_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LANGCACHE_DEVICE=cuda
export LANGCACHE_BATCH_SIZE=8

conda run -n ec python3 generate_langcache_embeddings.py
