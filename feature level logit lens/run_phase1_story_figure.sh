#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

CANDIDATES="${CANDIDATES:-intervention/checkpoints/phase1_neuronpedia_candidates.json}"
FEATURE_NOTES="${FEATURE_NOTES:-intervention/checkpoints/neuronpedia_feature_notes.json}"
OUTPUT="${OUTPUT:-intervention/checkpoints/fig_phase1_story.png}"
EXPORT_BLOCKS="${EXPORT_BLOCKS:-1}"
BLOCK_DIR="${BLOCK_DIR:-intervention/checkpoints/story_blocks}"

cd "$REPO_ROOT/others"

if [[ ! -f "$CANDIDATES" ]]; then
  echo "Missing candidates file: $REPO_ROOT/others/$CANDIDATES"
  echo "Run Phase 1 first, or set CANDIDATES=/path/to/phase1_neuronpedia_candidates.json"
  exit 1
fi

if [[ ! -f "$FEATURE_NOTES" ]]; then
  echo "Feature notes file not found: $REPO_ROOT/others/$FEATURE_NOTES"
  echo "Creating template notes file..."
  mkdir -p "$(dirname "$FEATURE_NOTES")"
  cat > "$FEATURE_NOTES" <<'JSON'
{
  "layer_20_feature_36332": {
    "name": "Add feature name from Neuronpedia",
    "explanation": "Add a short interpretation from Neuronpedia examples and tokens.",
    "url": "https://neuronpedia.org/"
  },
  "layer_12_feature_24585": {
    "name": "Add feature name from Neuronpedia",
    "explanation": "Add a short interpretation from Neuronpedia examples and tokens.",
    "url": "https://neuronpedia.org/"
  }
}
JSON
fi

extra_args=()
if [[ "$EXPORT_BLOCKS" == "1" ]]; then
  extra_args+=(--export-blocks --block-dir "$BLOCK_DIR")
fi

"$PYTHON_BIN" plot_phase1_story_figure.py \
  --candidates "$CANDIDATES" \
  --feature-notes "$FEATURE_NOTES" \
  --output "$OUTPUT" \
  "${extra_args[@]}"

echo
echo "Story figure generated: $REPO_ROOT/others/$OUTPUT"
echo "PDF version: ${REPO_ROOT}/others/${OUTPUT%.*}.pdf"
if [[ "$EXPORT_BLOCKS" == "1" ]]; then
  echo "Panel blocks directory: $REPO_ROOT/others/$BLOCK_DIR"
fi
