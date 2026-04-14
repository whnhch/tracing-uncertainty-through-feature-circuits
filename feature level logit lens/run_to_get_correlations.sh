TRANSFORMER_PATH="/scratch/general/nfs1/u1472329/cache/" # change this to your scratch path
export TRANSFORMERS_CACHE="$TRANSFORMER_PATH"
export HF_HOME="$TRANSFORMER_PATH"
export HF_DATASETS_CACHE="$TRANSFORMER_PATH"

python get_correlation_trivia_qa.py \
  --triviaqa-config rc.nocontext \
  --triviaqa-split validation \
  --start-index 0 \
  --n-prompts 32 \
  --slug-prefix triviaqa-gemma \
  --graph-file-dir "results/graph_files" \
  --skip-attribution \
  --top-k 100 \
  --feature-token-top-k 100 \
  --feature-lens-top-k-features 40 \
  --feature-lens-top-k-tokens 12