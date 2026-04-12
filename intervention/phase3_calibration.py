"""
Phase 3: Apply the Calibration-Appropriate Constraint

For each intervention result from Phase 2, classify the outcome as one of:
  - CALIBRATION_APPROPRIATE: entropy increased AND output remains coherent
  - SEMANTIC_DRIFT:          semantic similarity to pre-intervention output dropped
  - DEGRADATION:             correct answer fell out of top-k; output is incoherent

The four tests applied per (layer, feature, intervention_type):

  Test 1 — Top-k coherence:
    Correct answer must remain in top-k after intervention.

  Test 2 — Semantic similarity:
    Cosine similarity between pre- and post-intervention generated text must
    be above cfg.semantic_similarity_threshold.

  Test 3 — Behavioral calibration:
    Across the set of confident prompts, does expressed uncertainty (measured
    as hedging rate in generated text) correlate with lower accuracy?

  Test 4 — Hedging audit:
    Does the post-intervention generated text contain hedging language?

Output (saved to checkpoint):
  {
    "per_layer": {
        layer_idx: {
            feature_idx: {
                "ablation":            ClassificationResult,
                "amplification": {
                    scale: ClassificationResult
                }
            }
        }
    }
  }

where ClassificationResult = {
    "label":             str,   "calibration_appropriate" | "semantic_drift" | "degradation"
    "entropy_delta":     float, mean(post_entropy - pre_entropy)
    "coherence_rate":    float, fraction of prompts passing top-k coherence
    "semantic_sim":      float, mean cosine similarity pre/post
    "hedging_rate":      float, fraction of post-intervention outputs containing hedging
    "calibration_score": float, Spearman rho between expressed uncertainty and accuracy
}
"""

import torch
import numpy as np
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from transformer_lens import HookedTransformer
from sae_lens import SAE

from config import ExperimentConfig, device
from dataset import PromptBatch, get_batches
from utils import (
    ablate_feature,
    amplify_feature,
    contains_hedging,
    save_checkpoint,
)


CALIBRATION_APPROPRIATE = "calibration_appropriate"
SEMANTIC_DRIFT = "semantic_drift"
DEGRADATION = "degradation"


def _generate_texts(
    model: HookedTransformer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    sae: SAE | None = None,
    hook_name: str | None = None,
    feature_idx: int | None = None,
    mode: str | None = None,
    scale: float = 1.0,
) -> list[str]:
    """
    Greedy-decode up to max_new_tokens from input_ids, optionally with a
    feature intervention active at every forward pass during generation.

    Returns list of decoded new tokens (one string per prompt).
    """
    tokenizer = model.tokenizer

    def intervention_hook(value, hook):
        if mode == "ablation":
            return ablate_feature(value, sae, feature_idx)
        else:
            return amplify_feature(value, sae, feature_idx, scale)

    hooks = []
    if sae is not None and hook_name is not None:
        hooks = [(hook_name, intervention_hook)]

    generated = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if hooks:
                logits = model.run_with_hooks(generated, fwd_hooks=hooks)
            else:
                logits = model(generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            # Stop if all sequences have produced an EOS
            if (next_token == tokenizer.eos_token_id).all():
                break

    # Decode only the newly generated tokens
    new_tokens = generated[:, input_ids.shape[1]:]
    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in new_tokens]


def _classify_intervention(
    pre_entropy: torch.Tensor,
    post_entropy: torch.Tensor,
    coherence_rate: float,
    semantic_sim: float,
    hedging_rate: float,
    calibration_score: float,
    coherence_threshold: float = 0.5,
    semantic_threshold: float = 0.6,
    min_entropy_delta: float = 0.05,
) -> str:
    """
    Apply the classification logic from the plan.

    Order of checks:
      1. Degradation:             coherence_rate < threshold (correct answer fell out of top-k)
      2. Semantic drift:          semantic_sim < threshold
      3. Calibration-appropriate: entropy increased by at least min_entropy_delta AND passes
                                  coherence + semantic checks
      4. Otherwise: degradation  (entropy effect too small to be meaningful)

    min_entropy_delta filters out interventions that technically pass the coherence and
    semantic checks but produce no meaningful shift in confidence (Δentropy ≈ 0.000).
    """
    if coherence_rate < coherence_threshold:
        return DEGRADATION
    if semantic_sim < semantic_threshold:
        return SEMANTIC_DRIFT
    entropy_delta = (post_entropy - pre_entropy).mean().item()
    if entropy_delta >= min_entropy_delta:
        return CALIBRATION_APPROPRIATE
    return DEGRADATION


def run_phase3(
    cfg: ExperimentConfig,
    model: HookedTransformer,
    saes: dict[int, SAE],
    hook_names: dict[int, str],
    data: PromptBatch,
    phase2_results: dict,
) -> dict:
    """
    Apply the calibration-appropriate constraint to all Phase 2 interventions.

    Args:
        cfg:             experiment config
        model:           loaded HookedTransformer
        saes:            {layer_idx: SAE}
        data:            full PromptBatch
        phase2_results:  output of run_phase2

    Returns:
        results dict (also saved to checkpoint)
    """
    dev = device()
    sentence_encoder = SentenceTransformer(cfg.sentence_encoder_model, device=dev)

    confident_indices = phase2_results["confident_prompt_indices"]
    confident_data = PromptBatch(
        questions=[data.questions[i] for i in confident_indices.tolist()],
        prompts=[data.prompts[i] for i in confident_indices.tolist()],
        answers=[data.answers[i] for i in confident_indices.tolist()],
        input_ids=data.input_ids[confident_indices],
        attention_mask=data.attention_mask[confident_indices],
        correct_token_ids=data.correct_token_ids[confident_indices],
    )

    # Pre-intervention generated texts (baseline for semantic similarity)
    print("[phase3] generating pre-intervention baseline texts...")
    pre_texts: list[str] = []
    for mini in get_batches(confident_data, cfg.batch_size):
        pre_texts.extend(
            _generate_texts(model, mini.input_ids, cfg.generation_max_tokens)
        )

    pre_embeddings = sentence_encoder.encode(pre_texts, convert_to_tensor=True)

    per_layer = {}

    for layer_idx in cfg.model.layers_to_analyze:
        hook_name = hook_names[layer_idx]
        sae = saes[layer_idx]
        layer_phase2 = phase2_results["per_layer"][layer_idx]

        # Limit to top-N features by absolute ablation entropy delta to cut runtime.
        # Phase 2 stores all top_k_features; we only classify the strongest signal.
        feature_items = list(layer_phase2.items())
        if cfg.phase3_top_k < len(feature_items):
            feature_items = sorted(
                feature_items,
                key=lambda kv: abs((kv[1]["ablation"]["post_entropy"] - kv[1]["pre_entropy"]).mean().item()),
                reverse=True,
            )[:cfg.phase3_top_k]

        print(f"[phase3] layer {layer_idx}: classifying {len(feature_items)} / "
              f"{len(layer_phase2)} features "
              f"({'ablation only' if cfg.phase3_ablation_only else 'all interventions'})...")
        layer_results = {}

        for feature_idx, feat_data in feature_items:
            feature_results = {}

            interventions_to_classify = [("ablation", None, feat_data["ablation"])]
            if not cfg.phase3_ablation_only:
                for scale, amp_data in feat_data["amplification"].items():
                    interventions_to_classify.append(("amplification", scale, amp_data))

            for mode, scale, iv_data in interventions_to_classify:
                pre_entropy = feat_data["pre_entropy"]
                post_entropy = iv_data["post_entropy"]
                post_top_k_correct = iv_data["correct_mask"]

                # Test 1: top-k coherence rate
                coherence_rate = post_top_k_correct.float().mean().item()

                # Test 2: semantic similarity — generate post-intervention texts
                post_texts: list[str] = []
                for mini in get_batches(confident_data, cfg.batch_size):
                    post_texts.extend(
                        _generate_texts(
                            model, mini.input_ids, cfg.generation_max_tokens,
                            sae=sae, hook_name=hook_name,
                            feature_idx=feature_idx,
                            mode=mode, scale=scale if scale else 1.0,
                        )
                    )

                post_embeddings = sentence_encoder.encode(
                    post_texts, convert_to_tensor=True
                )
                # cosine similarity per prompt, then mean
                cos_sims = torch.nn.functional.cosine_similarity(
                    pre_embeddings, post_embeddings, dim=-1
                )
                semantic_sim = cos_sims.mean().item()

                # Test 4: hedging rate
                h_rate = sum(contains_hedging(t) for t in post_texts) / len(post_texts)

                # Test 3: behavioral calibration
                # Spearman rho between per-prompt entropy delta and per-prompt accuracy
                entropy_delta = (post_entropy - pre_entropy).numpy()
                accuracy = iv_data["correct_mask"].float().numpy()
                # Negative correlation expected: higher entropy delta → lower accuracy
                # (calibration-appropriate means the model expresses appropriate uncertainty)
                rho, _ = spearmanr(entropy_delta, accuracy)
                calibration_score = float(rho) if not np.isnan(rho) else 0.0

                label = _classify_intervention(
                    pre_entropy, post_entropy,
                    coherence_rate=coherence_rate,
                    semantic_sim=semantic_sim,
                    hedging_rate=h_rate,
                    calibration_score=calibration_score,
                    coherence_threshold=0.5,
                    semantic_threshold=cfg.semantic_similarity_threshold,
                    min_entropy_delta=cfg.min_entropy_delta,
                )

                result = {
                    "label": label,
                    "entropy_delta": (post_entropy - pre_entropy).mean().item(),
                    "coherence_rate": coherence_rate,
                    "semantic_sim": semantic_sim,
                    "hedging_rate": h_rate,
                    "calibration_score": calibration_score,
                }

                key = "ablation" if mode == "ablation" else scale
                feature_results[key] = result

                print(f"  layer {layer_idx} feat {feature_idx} {mode}"
                      f"{'x'+str(scale) if scale else ''}: {label} "
                      f"(Δentropy={result['entropy_delta']:+.3f}, "
                      f"coherence={coherence_rate:.2f}, "
                      f"sem_sim={semantic_sim:.2f})")

            layer_results[feature_idx] = feature_results

        per_layer[layer_idx] = layer_results

    results = {"per_layer": per_layer}

    ckpt_path = f"{cfg.checkpoint_dir}/phase3_calibration.pt"
    save_checkpoint(results, ckpt_path)
    return results


def summarize_phase3(results: dict) -> None:
    """
    Print a clear summary of Phase 3 classification outcomes.

    Breaks down results by:
      - Overall label counts
      - Ablation vs amplification separately (ablation results are the cleanest signal)
      - Per-layer calibration-appropriate counts
      - Top CA features by Δentropy (ablation only, the most scientifically defensible)
    """
    from collections import defaultdict

    # Collect all results
    all_results = []
    for layer_idx, layer_data in results["per_layer"].items():
        for feat_idx, feat_data in layer_data.items():
            for key, result in feat_data.items():
                iv_type = "ablation" if key == "ablation" else f"amp×{key}"
                all_results.append((layer_idx, feat_idx, iv_type, result))

    total = len(all_results)
    ablation_results  = [(l, f, t, r) for l, f, t, r in all_results if t == "ablation"]
    amp_results       = [(l, f, t, r) for l, f, t, r in all_results if t != "ablation"]

    def count_label(rows, label):
        return sum(1 for _, _, _, r in rows if r["label"] == label)

    print("\n" + "="*60)
    print("  Phase 3 Summary")
    print("="*60)

    # Overall
    ca_total   = count_label(all_results, CALIBRATION_APPROPRIATE)
    deg_total  = count_label(all_results, DEGRADATION)
    drift_total = count_label(all_results, SEMANTIC_DRIFT)
    print(f"\n  Overall ({total} interventions):")
    print(f"    calibration-appropriate : {ca_total:4d}  ({100*ca_total/total:.1f}%)")
    print(f"    degradation             : {deg_total:4d}  ({100*deg_total/total:.1f}%)")
    print(f"    semantic drift          : {drift_total:4d}  ({100*drift_total/total:.1f}%)")

    # Ablation vs amplification split
    ca_abl = count_label(ablation_results, CALIBRATION_APPROPRIATE)
    ca_amp = count_label(amp_results, CALIBRATION_APPROPRIATE)
    print(f"\n  Calibration-appropriate by intervention type:")
    print(f"    ablation only           : {ca_abl:4d} / {len(ablation_results)}  ({100*ca_abl/max(len(ablation_results),1):.1f}%)")
    print(f"    amplification only      : {ca_amp:4d} / {len(amp_results)}  ({100*ca_amp/max(len(amp_results),1):.1f}%)")

    # Per-layer CA count (ablation only)
    print(f"\n  Calibration-appropriate ablations per layer:")
    layer_ca = defaultdict(int)
    layer_total = defaultdict(int)
    for l, f, t, r in ablation_results:
        layer_total[l] += 1
        if r["label"] == CALIBRATION_APPROPRIATE:
            layer_ca[l] += 1
    for l in sorted(layer_ca.keys()):
        print(f"    layer {l:2d}: {layer_ca[l]:3d} / {layer_total[l]}  ({100*layer_ca[l]/layer_total[l]:.1f}%)")

    # Top CA features by Δentropy (ablation only — cleanest causal signal)
    ca_ablations = [
        (r["entropy_delta"], l, f, r)
        for l, f, t, r in ablation_results
        if r["label"] == CALIBRATION_APPROPRIATE
    ]
    ca_ablations.sort(reverse=True)

    print(f"\n  Top calibration-appropriate ablations by Δentropy:")
    print(f"    {'Δentropy':>10}  {'layer':>6}  {'feature':>8}  {'coherence':>9}  {'sem_sim':>7}  {'hedging':>7}")
    for delta, layer, feat, r in ca_ablations[:15]:
        print(f"    {delta:>10.4f}  {layer:>6}  {feat:>8}  "
              f"{r['coherence_rate']:>9.2f}  {r['semantic_sim']:>7.2f}  {r['hedging_rate']:>7.2f}")

    print("="*60 + "\n")
