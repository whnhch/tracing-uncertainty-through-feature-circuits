#!/bin/bash
#SBATCH --job-name=uncertainty-tracing
#SBATCH --gres=gpu:h200_3g.71gb:1
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


source .venv/bin/activate

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python intervention/run_pipeline.py --model gemma-2-9b

echo "Job finished: $(date)"
