#!/bin/bash
#SBATCH --job-name=uncertainty-tracing
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/gemma_%j.out
#SBATCH --error=logs/gemma_%j.err

#SBATCH --partition=granite-gpu-guest
#SBATCH --account=cs6966
#SBATCH --qos=granite-gpu-guest

set -e
mkdir -p logs

module load cuda/12.4.0
module load python/3.12.12

# Point HuggingFace cache to scratch space to avoid home quota issues.
export HF_HOME=/scratch/$USER/hf_cache

source .venv/bin/activate

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python intervention/run_pipeline.py --model gemma-2-9b

echo "Job finished: $(date)"
