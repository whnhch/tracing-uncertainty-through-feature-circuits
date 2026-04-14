"""
Manual inspection tool for CA features.

Loads the model, runs a prompt with and without a feature ablated, and prints
side-by-side before/after text so you can check for hedging or other changes.

Usage:
  # Test the top CA feature on a few default prompts
  python intervention/manual_test.py --model gemma-2-9b --layer 24 --feature 1927

  # Test all 11 CA features
  python intervention/manual_test.py --model gemma-2-9b --all-ca

  # Use your own prompt
  python intervention/manual_test.py --model gemma-2-9b --layer 24 --feature 1927 \\
      --prompt "Q: What is the capital of France?\\nA:"

  # Generate more tokens to see fuller responses
  python intervention/manual_test.py --model gemma-2-9b --all-ca --max-tokens 40

  # Also test amplification at 2x and 5x
  python intervention/manual_test.py --model gemma-2-9b --layer 24 --feature 1927 --amplify
"""

import argparse
import sys
import os
import torch

# Allow running from repo root or from inside intervention/
sys.path.insert(0, os.path.dirname(__file__))

from config import ExperimentConfig, device
from utils import ablate_feature, amplify_feature, contains_hedging, output_entropy
from run_pipeline import load_model_and_saes


# The 11 CA features from the Gemma-2 9B run, ranked by change in entropy
CA_FEATURES = [
    (24, 1927),
    (32, 5927),
    (32, 3619),
    (32, 10570),
    (8, 7703),
    (32, 10792),
    (40, 10332),
    (40, 2882),
    (24, 5307),
    (40, 8187),
    (24, 8740),
]

DEFAULT_PROMPTS = [
    "Q: What is the capital of France?\nA:",
    "Q: Who wrote the play Hamlet?\nA:",
    "Q: What element does the chemical symbol Au represent?\nA:",
    "Q: In what year did World War II end?\nA:",
    "Q: What is the largest planet in the solar system?\nA:",
]


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    dev: str,
    hooks: list = None,
) -> tuple[str, float]:
    """
    Greedy-decode from prompt, optionally with hooks active.
    Returns (generated_text, entropy_at_first_token).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(dev)
    input_ids = inputs["input_ids"]

    first_entropy = None
    generated = input_ids.clone()

    with torch.no_grad():
        for step in range(max_new_tokens):
            if hooks:
                logits = model.run_with_hooks(generated, fwd_hooks=hooks)
            else:
                logits = model(generated)

            if step == 0:
                first_entropy = output_entropy(logits[:, -1:, :]).item()

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            if (next_token == tokenizer.eos_token_id).all():
                break

    new_tokens = generated[:, input_ids.shape[1] :]
    text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    return text, first_entropy


def run_test(
    model,
    saes: dict,
    hook_names: dict,
    layer: int,
    feature: int,
    prompt: str,
    max_new_tokens: int,
    amplify: bool = False,
    amplify_scales: list = None,
):
    dev = str(next(model.parameters()).device)
    tokenizer = model.tokenizer
    hook_name = hook_names[layer]
    sae = saes[layer]

    if amplify_scales is None:
        amplify_scales = [2.0, 5.0]

    # Baseline
    baseline_text, baseline_entropy = generate(
        model, tokenizer, prompt, max_new_tokens, dev
    )

    # Ablation
    def ablation_hook(value, hook):
        return ablate_feature(value, sae, feature)

    ablated_text, ablated_entropy = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens,
        dev,
        hooks=[(hook_name, ablation_hook)],
    )

    delta = ablated_entropy - baseline_entropy

    print(f"\n{'─'*65}")
    print(f"  Layer {layer}  |  Feature {feature}")
    print(f"  Prompt: {prompt.strip()[:80]}")
    print(f"{'─'*65}")
    print(f"  BEFORE  (entropy={baseline_entropy:.3f})")
    print(f"    {baseline_text!r}")
    print(f"    hedging: {'YES' if contains_hedging(baseline_text) else 'no'}")
    print()
    print(f"  AFTER ablation  (entropy={ablated_entropy:.3f}, Δ={delta:+.3f})")
    print(f"    {ablated_text!r}")
    print(f"    hedging: {'YES <--' if contains_hedging(ablated_text) else 'no'}")

    if amplify:
        for scale in amplify_scales:

            def amp_hook(value, hook, s=scale):
                return amplify_feature(value, sae, feature, s)

            amp_text, amp_entropy = generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens,
                dev,
                hooks=[(hook_name, amp_hook)],
            )
            amp_delta = amp_entropy - baseline_entropy
            print()
            print(
                f"  AFTER amplify {scale}x  (entropy={amp_entropy:.3f}, Δ={amp_delta:+.3f})"
            )
            print(f"    {amp_text!r}")
            print(f"    hedging: {'YES <--' if contains_hedging(amp_text) else 'no'}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Manual CA feature inspection")
    parser.add_argument(
        "--model", default="gemma-2-9b", choices=["gpt2-small", "gemma-2-9b"]
    )
    parser.add_argument(
        "--layer", type=int, default=None, help="Layer index of the feature to test"
    )
    parser.add_argument(
        "--feature", type=int, default=None, help="Feature index to ablate"
    )
    parser.add_argument(
        "--all-ca",
        action="store_true",
        help="Test all 11 CA features from the Gemma-2 9B run",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt string (default: cycle through built-in prompts)",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=3,
        help="How many of the default prompts to run (1-5, default 3)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=30,
        help="Max new tokens to generate (default 30)",
    )
    parser.add_argument(
        "--amplify", action="store_true", help="Also test 2x and 5x amplification"
    )
    args = parser.parse_args()

    if not args.all_ca and (args.layer is None or args.feature is None):
        parser.error("Provide --layer and --feature, or use --all-ca")

    cfg = ExperimentConfig(model_key=args.model)
    print(f"[setup] loading {cfg.model.model_name}...")
    model, saes, hook_names = load_model_and_saes(cfg)
    model.eval()

    features_to_test = CA_FEATURES if args.all_ca else [(args.layer, args.feature)]
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS[: args.n_prompts]

    print(
        f"\n[manual_test] {len(features_to_test)} feature(s) x {len(prompts)} prompt(s)"
        f"  |  max_tokens={args.max_tokens}\n"
    )

    for layer, feature in features_to_test:
        if layer not in saes:
            print(f"[skip] layer {layer} not loaded (loaded: {list(saes.keys())})")
            continue
        for prompt in prompts:
            run_test(
                model,
                saes,
                hook_names,
                layer=layer,
                feature=feature,
                prompt=prompt,
                max_new_tokens=args.max_tokens,
                amplify=args.amplify,
            )

    print("Done.")


if __name__ == "__main__":
    main()
