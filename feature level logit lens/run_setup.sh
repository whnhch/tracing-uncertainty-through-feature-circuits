#!/usr/bin/env bash
set -euo pipefail

module load cuda/13

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$REPO_ROOT/hf_token.txt}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/general/nfs1/u1472329/cache}"

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

chmod -R 755 "$REPO_ROOT/results"
find "$REPO_ROOT/results" -type f -exec chmod 644 {} \;
