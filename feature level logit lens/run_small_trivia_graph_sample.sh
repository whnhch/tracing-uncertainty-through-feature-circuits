#!/usr/bin/env bash
set -euo pipefail

# Optional cache location for cluster/local runs.
TRANSFORMER_PATH="${TRANSFORMER_PATH:-/scratch/general/nfs1/u1472329/cache/}"
GRAPH_CHECKPOINT_DIR="${GRAPH_CHECKPOINT_DIR:-/scratch/general/nfs1/u1472329/cache/graphs/}"
export TRANSFORMERS_CACHE="$TRANSFORMER_PATH"
export HF_HOME="$TRANSFORMER_PATH"
export HF_DATASETS_CACHE="$TRANSFORMER_PATH"

mkdir -p "$GRAPH_CHECKPOINT_DIR"

# Small sample run: few prompts + few logits.
python get_correlation_trivia_qa.py \
  --model-name google/gemma-2-9b \
  --transcoder-name gemma \
  --triviaqa-config rc.nocontext \
  --triviaqa-split validation \
  --start-index 0 \
  --n-prompts 4 \
  --append-answer-cue \
  --slug-prefix triviaqa-gemma-sample \
  --graph-dir "$GRAPH_CHECKPOINT_DIR" \
  --max-n-logits 3 \
  --top-k 20 \
  --feature-token-top-k 20 \
  --feature-lens-top-k-features 10 \
  --feature-lens-top-k-tokens 6 \
  --output-json "feature level logit lens/data/phase1_triviaqa_graph_corr_sample.json" \
  --output-csv "feature level logit lens/data/phase1_triviaqa_graph_corr_sample.csv" \
  --summary-path "feature level logit lens/data/phase1_triviaqa_graph_corr_summary_sample.txt" \
  --feature-token-heatmap-path "feature level logit lens/data/feature_token_effects_heatmap_sample.png"
