# Data Collection Summary

This project collects prompt-like text examples from the Neuronpedia feature API for Gemma (`gemma-2-2b`) using the `gemmascope-res-16k` feature set.

## What We Collect

- Text examples reconstructed from feature `activations -> tokens`.
- Up to 20 examples per selected layer.
- 5 total layers per run (with layer 21 always included, plus 4 random valid layers).

## How Examples Are Filtered

- Length filter: keep only text with 8 to 200 characters.
- English-like filter: reject examples with clear Hangul/CJK usage and keep samples with mostly English letters.
- Style filter: keep only prompts classified as either:
	- `fact` (e.g., QA/factual style patterns)
	- `equation` (e.g., arithmetic/equation style patterns)
- Deduplication: globally remove duplicates across all selected layers.

## Output File

Results are saved to `random_prompts.json` with:

- Selected layers and required layer
- Feature IDs used per layer
- Prompts grouped by layer (`prompts_by_layer`)
- Flattened prompt list (`prompts`) and total count (`prompt_count`)
- Metadata for filters (`min_prompt_chars`, `max_prompt_chars`, `require_english`, `allowed_styles`)

## TODO

- Poster dataset plan: use manually curated examples (not only random API samples), e.g., gemma-fact-dallas-austin.
