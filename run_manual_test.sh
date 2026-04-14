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

# --- Top CA feature: single feature, default prompts, ablation only ---
echo "=== Test 1: Layer 24 Feature 1927 (top CA, ablation only) ==="
python -u intervention/manual_test.py \
    --model gemma-2-9b \
    --layer 24 --feature 1927 \
    --max-tokens 40

# --- Top CA feature: with amplification ---
echo "=== Test 2: Layer 24 Feature 1927 (with 2x and 5x amplification) ==="
python -u intervention/manual_test.py \
    --model gemma-2-9b \
    --layer 24 --feature 1927 \
    --max-tokens 40 --amplify

# --- All 11 CA features, default prompts ---
echo "=== Test 3: All 11 CA features ==="
python -u intervention/manual_test.py \
    --model gemma-2-9b \
    --all-ca \
    --max-tokens 40

# --- Top CA feature: custom prompts (harder / more ambiguous questions) ---
echo "=== Test 4: Layer 24 Feature 1927, ambiguous prompts ==="
python -u intervention/manual_test.py \
    --model gemma-2-9b \
    --layer 24 --feature 1927 \
    --max-tokens 40 \
    --prompt "Q: Who was the first person to walk on the moon?\nA:"

python -u intervention/manual_test.py \
    --model gemma-2-9b \
    --layer 24 --feature 1927 \
    --max-tokens 40 \
    --prompt "Q: What is the boiling point of water at high altitude?\nA:"

python -u intervention/manual_test.py \
    --model gemma-2-9b \
    --layer 24 --feature 1927 \
    --max-tokens 40 \
    --prompt "Q: Who invented the telephone?\nA:"

echo "Job finished: $(date)"
