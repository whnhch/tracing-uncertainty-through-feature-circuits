#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEXT_FILE="${SCRIPT_DIR}/data/input_sentence.txt"
TRANSCODER_CKPT="${SCRIPT_DIR}/models/transcoder.pt"

mkdir -p "${SCRIPT_DIR}/data"

if [[ ! -f "${TEXT_FILE}" ]]; then
	echo "Uncertainty can be traced through feature circuits in neural language models." > "${TEXT_FILE}"
fi

EXTRA_ARGS=()
if [[ -f "${TRANSCODER_CKPT}" ]]; then
	EXTRA_ARGS+=(
		--run-feature-lens
		--feature-transcoder-path "${TRANSCODER_CKPT}"
		--feature-plot-path "${SCRIPT_DIR}/data/feature_logit_lens_next_token.png"
		--feature-summary-path "${SCRIPT_DIR}/data/feature_logit_lens_summary.txt"
	)
fi

python "${SCRIPT_DIR}/run_feature_level_logit_lens.py" \
	--model-name "gpt2" \
	--text-file "${TEXT_FILE}" \
	--plot-path "${SCRIPT_DIR}/data/logit_lens_next_token.png" \
	--summary-path "${SCRIPT_DIR}/data/logit_lens_summary.txt" \
	"${EXTRA_ARGS[@]}"

# TODO: set scratch file for every llm settings