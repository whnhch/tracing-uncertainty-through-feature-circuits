#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$REPO_ROOT/hf_token.txt}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/general/nfs1/u1472329/cache}"
MODEL_KEY="${MODEL_KEY:-gemma-2-2b}"
N_PROMPTS="${N_PROMPTS:-50}"

export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SCRATCH_ROOT/hf_datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$SCRATCH_ROOT/transformers}"
export TOKENIZERS_PARALLELISM="false"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

if [[ -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
  export HF_TOKEN
  huggingface-cli login --token "$HF_TOKEN"
else
  echo "Warning: HF token file not found at $HF_TOKEN_FILE"
fi

cd "$REPO_ROOT/others"
python run_pipeline.py \
  --phases 1 \
  --model "$MODEL_KEY" \
  --n-prompts "$N_PROMPTS"

echo
if [[ -f "intervention/checkpoints/phase1_neuronpedia_candidates.txt" ]]; then
  echo "Saved Neuronpedia report: $REPO_ROOT/others/intervention/checkpoints/phase1_neuronpedia_candidates.txt"
fi
