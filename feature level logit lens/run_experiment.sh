#!/usr/bin/env bash
set -euo pipefail

TRANSFORMER_PATH="/scratch/general/nfs1/u1472329/cache/" # change this to your scratch path or not then it will let you use the default huggingface cache path (local) which only has 
export TRANSFORMERS_CACHE="$TRANSFORMER_PATH"
export HF_HOME="$TRANSFORMER_PATH"
export HF_DATASETS_CACHE="$TRANSFORMER_PATH"
# TRANSFORMER_PATH="~/.cache/huggingface/hub"

# MODEL_NAME="Qwen/Qwen3.5-0.8B" 
MODEL_NAME="Qwen/Qwen3.5-27B" 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEXT_FILE="${SCRIPT_DIR}/data/input_sentence.txt"
TRANSCODER_CKPT="${SCRIPT_DIR}/models/transcoder.pt"
RUN_TRANSCODER_TRAIN="${RUN_TRANSCODER_TRAIN:-0}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
RUN_FEATURE_LENS="${RUN_FEATURE_LENS:-1}"

if [[ "${RUN_FEATURE_LENS}" == "1" && ! -f "${TRANSCODER_CKPT}" && "${RUN_TRANSCODER_TRAIN}" != "1" ]]; then
	echo "No transcoder checkpoint found at ${TRANSCODER_CKPT}; enabling training so feature-lens can run."
	RUN_TRANSCODER_TRAIN="1"
fi

EXTRA_ARGS=()
if [[ "${RUN_FEATURE_LENS}" == "1" || "${RUN_TRANSCODER_TRAIN}" == "1" || -f "${TRANSCODER_CKPT}" ]]; then
	EXTRA_ARGS+=(
		--run-feature-lens
		--feature-transcoder-path "${TRANSCODER_CKPT}"
		--feature-plot-path "${SCRIPT_DIR}/data/feature_logit_lens_next_token.png"
		--feature-summary-path "${SCRIPT_DIR}/data/feature_logit_lens_summary.txt"
	)
fi

if [[ "${RUN_TRANSCODER_TRAIN}" == "1" ]]; then
	EXTRA_ARGS+=(
		--run-transcoder-train
		--num-epochs "${TRAIN_EPOCHS}"
		--save-path "${TRANSCODER_CKPT}"
	)
fi

echo "RUN_FEATURE_LENS=${RUN_FEATURE_LENS} RUN_TRANSCODER_TRAIN=${RUN_TRANSCODER_TRAIN}"

export CUDA_LAUNCH_BLOCKING=1
export TRANSFORMERS_VERBOSITY=debug

python -X faulthandler "${SCRIPT_DIR}/run_feature_level_logit_lens.py" \
	--transformer_path "${TRANSFORMER_PATH}" \
	--model-name "${MODEL_NAME}" \
	--text-file "${TEXT_FILE}" \
	--device "cuda" \
	--plot-path "${SCRIPT_DIR}/data/logit_lens_next_token.png" \
	--summary-path "${SCRIPT_DIR}/data/logit_lens_summary.txt" \
	"${EXTRA_ARGS[@]}"