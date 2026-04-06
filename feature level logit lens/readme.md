# CS6966 Data + Circuit Pipeline

This project has two stages:

1. Prompt/feature sampling with `get_data.py`
2. Circuit extraction with `get_circuit.py`

## Stage 1: Build Prompt Dataset (`get_data.py`)

`get_data.py` collects prompt-like examples from Neuronpedia feature activations for `gemma-2-2b` (`gemmascope-res-16k`).

### Input

No input file is required. Configuration is inside `get_data.py` (model, layers, sample counts, filtering rules).

### Filtering

- Prompt length: 8 to 200 chars
- English-like heuristic
- Style classification: `fact` or `equation`
- Global deduplication across sampled layers

### Output

Writes `random_prompts.json` with:

- `model`
- `selected_layers`
- `sampled_feature_ids_by_layer`
- `prompts_by_layer`
- `prompts` (flattened list)
- `prompt_count`
- filter metadata fields

## Stage 2: Extract Circuit (`get_circuit.py`)

`get_circuit.py` reads sampled features/prompts from `random_prompts.json`, fetches Neuronpedia feature records, then builds a graph using Neuronpedia graph fields (primarily `topkCosSim` in this setup).

### Inputs

- Required file: `random_prompts.json`
- Runtime flags:
	- `--input`: input JSON path
	- `--output`: output JSON path
	- `--sleep-seconds`: request delay to avoid burst traffic
	- `--logit-top-k`: number of top positive/negative logit tokens per feature
	- `--max-explanations`: max explanation strings per feature

### Per-example mode

Use `--example-index N` to run for one prompt example.

Flow in per-example mode:

1. Read one prompt from `prompts[N]`
2. Pick one seed feature per layer (highest prompt-text similarity)
3. Expand neighbors by `topkCosSim` up to `--hop-depth`
4. Fetch all encountered feature records
5. Save graph + feature metadata including explanations and logits

Example command:

```bash
python3 get_circuit.py \
	--input random_prompts.json \
	--output circuit_example_0.json \
	--example-index 0 \
	--hop-depth 1 \
	--topk-neighbors 25 \
	--sleep-seconds 1.0 \
	--logit-top-k 10 \
	--max-explanations 3
```

### Output schema (`circuit_example_0.json`)

Top-level fields:

- `model`, `method`, `mode`, `input_file`
- `parameters`
- `example_prompt` (per-example mode)
- `seed_features` (one seed per layer in per-example mode)
- `summary`
- `nodes`
- `edges`
- `fetch_failures`

`summary` includes:

- node/edge totals
- same-layer vs cross-layer edge counts
- node counts by layer
- edge counts by type

Each item in `nodes` includes:

- feature id + layer metadata
- matched prompt (if present in sampled set)
- `max_act_approx`
- `feature_explanations`
- `feature_logits`:
	- `top_positive`: token-score pairs
	- `top_negative`: token-score pairs

Each item in `edges` includes:

- source/target feature ids
- source/target layers
- edge type
- edge weight

## Notes

- In this current Gemma source setup, extracted `topkCosSim` links are mostly same-layer neighbors.
- That means the output is a Neuronpedia-style feature relationship graph, not necessarily a causal cross-layer circuit.

## Full Feature Logits From Unembedding (Local Compute)

If you want full feature logits from intermediate feature directions and the model unembedding, use:

- `feature_unembed_logits.py`

This computes:

- `logits = W_U @ d_f`

where:

- `W_U` is the model output embedding (unembedding matrix)
- `d_f` is the decoder direction for feature `f`

### Required inputs

1. A model checkpoint (HF id or local path), e.g. `google/gemma-2-2b`
2. A decoder matrix file for the SAE/transcoder source:
	 - supported: `.npy`, `.pt`/`.bin`, `.safetensors`
3. One or more feature ids

### Example command

```bash
python3 feature_unembed_logits.py \
	--model-id google/gemma-2-2b \
	--decoder-path /path/to/decoder_weights.safetensors \
	--decoder-key W_dec \
	--feature-ids 1690 3154 \
	--top-k 20 \
	--output feature_unembed_logits.json
```

### Output

`feature_unembed_logits.json` includes:

- model + decoder metadata
- decoder shape
- per-feature top positive/negative tokens with projected logit values
- optional full vocab logits when `--save-full-logits` is set

## Visualize Target Token vs Circuit Features

Use `visualize_target_token_logit.py` when you want, for one example circuit, to see how each feature contributes to one target token logit.

This script:

1. Reads a per-example circuit file (for example `circuit_example_0.json`)
2. Loads model unembedding and decoder directions
3. Computes per-feature contribution for one target token:
	 - `contribution = <W_U[target_token], d_f>`
4. Saves a CSV/JSON table plus a PNG plot

### Example command

```bash
python3 visualize_target_token_logit.py \
	--circuit-json circuit_example_0.json \
	--model-id google/gemma-2-2b \
	--decoder-path /path/to/decoder_weights.safetensors \
	--decoder-key W_dec \
	--target-token-string "=" \
	--out-dir viz_target_token \
	--prefix example0
```

If you omit `--target-token-id` and `--target-token-string`, it uses the last token of `example_prompt` in the circuit JSON.

### Outputs

- `viz_target_token/example0_token_by_feature.csv`
- `viz_target_token/example0_token_by_feature.json`
- `viz_target_token/example0_token_by_feature.png`
