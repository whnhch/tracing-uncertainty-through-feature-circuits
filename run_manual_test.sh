#!/bin/bash
#SBATCH --job-name=manual-test
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/manual_test_%j.out
#SBATCH --error=logs/manual_test_%j.err

#SBATCH --partition=granite-gpu-guest
#SBATCH --account=cs6966
#SBATCH --qos=granite-gpu-guest

set -e
mkdir -p logs

module load cuda/12.4.0
module load python/3.12.12

source .venv/bin/activate

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

export HUGGING_FACE_HUB_TOKEN=$(cat ~/.cache/huggingface/token)

python -u intervention/manual_test.py \
    --model gemma-2-9b \
    --layer 24 \
    --feature 1927 \
    --max-tokens 40 \
    --amplify

echo "Job finished: $(date)"
