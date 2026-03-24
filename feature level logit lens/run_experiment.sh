#!/usr/bin/env bash
set -euo pipefail

TRANSFORMER_PATH="/scratch/general/nfs1/u1472329/cache/" # change this to your scratch path
export TRANSFORMERS_CACHE="$TRANSFORMER_PATH"
export HF_HOME="$TRANSFORMER_PATH"
export HF_DATASETS_CACHE="$TRANSFORMER_PATH"
# TRANSFORMER_PATH="~/.cache/huggingface/hub"

MODEL_NAME="Qwen/Qwen3.5-0.8B" 
# MODEL_NAME="Qwen/Qwen3.5-9B" 
# MODEL_NAME="Qwen/Qwen3.5-27B" 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEXT_FILE="${SCRIPT_DIR}/data/input_sentence.txt"
FEATURE_TRANSCODER_DIR="${FEATURE_TRANSCODER_DIR:-${SCRIPT_DIR}/models/transcoder_layers}"
FEATURE_TRANSCODER_LAYER_PATTERN="${FEATURE_TRANSCODER_LAYER_PATTERN:-layer_{layer}.pt}"
RUN_TRANSCODER_TRAIN="${RUN_TRANSCODER_TRAIN:-0}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
RUN_FEATURE_LENS="${RUN_FEATURE_LENS:-1}"
TARGET_TOKEN="${TARGET_TOKEN:-"Austin"}"
FEATURE_TOKENWISE_OUTPUT_DIR="${FEATURE_TOKENWISE_OUTPUT_DIR:-${SCRIPT_DIR}/data/per_token_feature_topn}"
FEATURE_LAYER_SAMPLE_COUNT="${FEATURE_LAYER_SAMPLE_COUNT:-10}"

if [[ "${RUN_FEATURE_LENS}" == "1" && "${RUN_TRANSCODER_TRAIN}" != "1" ]]; then
	if [[ ! -d "${FEATURE_TRANSCODER_DIR}" ]]; then
		echo "No per-layer transcoder directory found at ${FEATURE_TRANSCODER_DIR}; enabling training so feature-lens can run."
		RUN_TRANSCODER_TRAIN="1"
	elif ! compgen -G "${FEATURE_TRANSCODER_DIR}/*.pt" > /dev/null; then
		echo "Per-layer transcoder directory is empty at ${FEATURE_TRANSCODER_DIR}; enabling training so feature-lens can run."
		RUN_TRANSCODER_TRAIN="1"
	fi
fi

EXTRA_ARGS=()
if [[ "${RUN_FEATURE_LENS}" == "1" || "${RUN_TRANSCODER_TRAIN}" == "1" || -d "${FEATURE_TRANSCODER_DIR}" ]]; then
	EXTRA_ARGS+=(
		--run-feature-lens
		--feature-plot-path "${SCRIPT_DIR}/data/feature_logit_lens_next_token.png"
		--feature-summary-path "${SCRIPT_DIR}/data/feature_logit_lens_summary.txt"
		--feature-layer-sample-count "${FEATURE_LAYER_SAMPLE_COUNT}"
		--feature-tokenwise-output-dir "${FEATURE_TOKENWISE_OUTPUT_DIR}"
		--feature-transcoder-dir "${FEATURE_TRANSCODER_DIR}"
		--feature-transcoder-layer-pattern "${FEATURE_TRANSCODER_LAYER_PATTERN}"
	)
fi

if [[ "${RUN_TRANSCODER_TRAIN}" == "1" ]]; then
	EXTRA_ARGS+=(
		--run-transcoder-train
		--num-epochs "${TRAIN_EPOCHS}"
		--save-dir "${FEATURE_TRANSCODER_DIR}"
		--save-layer-pattern "${FEATURE_TRANSCODER_LAYER_PATTERN}"
	)
fi

echo "RUN_FEATURE_LENS=${RUN_FEATURE_LENS} RUN_TRANSCODER_TRAIN=${RUN_TRANSCODER_TRAIN}"
echo "TARGET_TOKEN=${TARGET_TOKEN}"
echo "FEATURE_LAYER_SAMPLE_COUNT=${FEATURE_LAYER_SAMPLE_COUNT}"
echo "FEATURE_TOKENWISE_OUTPUT_DIR=${FEATURE_TOKENWISE_OUTPUT_DIR}"
echo "FEATURE_TRANSCODER_DIR=${FEATURE_TRANSCODER_DIR}"
echo "FEATURE_TRANSCODER_LAYER_PATTERN=${FEATURE_TRANSCODER_LAYER_PATTERN}"

export CUDA_LAUNCH_BLOCKING=1
export TRANSFORMERS_VERBOSITY=debug

python -X faulthandler "${SCRIPT_DIR}/run_feature_level_logit_lens.py" \
	--transformer_path "${TRANSFORMER_PATH}" \
	--model-name "${MODEL_NAME}" \
	--text-file "${TEXT_FILE}" \
	--device "cuda" \
	--num-epochs 50 \
	--plot-path "${SCRIPT_DIR}/data/logit_lens_next_token.png" \
	--summary-path "${SCRIPT_DIR}/data/logit_lens_summary.txt" \
	--target-token "${TARGET_TOKEN}" \
	"${EXTRA_ARGS[@]}"