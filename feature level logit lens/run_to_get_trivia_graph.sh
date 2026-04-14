TRANSFORMER_PATH="/scratch/general/nfs1/u1472329/cache/" # change this to your scratch path
export TRANSFORMERS_CACHE="$TRANSFORMER_PATH"
export HF_HOME="$TRANSFORMER_PATH"
export HF_DATASETS_CACHE="$TRANSFORMER_PATH"

python get_correlation_trivia_qa.py \
  --model-name google/gemma-2-9b \
  --transcoder-name gemma \
  --triviaqa-config rc.nocontext \
  --triviaqa-split validation \
  --start-index 0 \
  --n-prompts 32 \
  --append-answer-cue \
  --slug-prefix triviaqa-gemma \
  --top-k 100 \
  --feature-token-top-k 100 \
  --feature-lens-top-k-features 40 \
  --feature-lens-top-k-tokens 12