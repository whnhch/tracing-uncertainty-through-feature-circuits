"""
Phase 1: Identify Uncertainty-Linked Features (Correlational)

For each layer in layers_to_analyze:
  - Run forward passes on the full prompt dataset
  - Extract SAE feature activations at the last token position
  - Compute output logit entropy for each prompt
  - Compute Spearman correlation between each feature's activation and entropy
  - Identify top-k features most correlated with uncertainty

Output (saved to checkpoint):
  {
    "entropy_scores":   Tensor [n_prompts],
    "correct_mask":     Tensor [n_prompts] bool,  # model answered correctly
    "per_layer": {
        layer_idx: {
            "correlations":     Tensor [n_features],  # Spearman rho per feature
            "top_feature_ids":  Tensor [top_k],       # indices of top correlated features
            "top_correlations": Tensor [top_k],
        }
    }
  }
"""

import torch
import numpy as np
from functools import partial
from scipy.stats import spearmanr
from transformer_lens import HookedTransformer
from sae_lens import SAE

from config import ExperimentConfig, device
from dataset import PromptBatch, get_batches
from utils import output_entropy, correct_answer_in_top_k, save_checkpoint, checkpoint_exists


def _extract_sae_activations_and_entropy(
    model: HookedTransformer,
    sae: SAE,
    hook_name: str,
    batch: PromptBatch,
    dev: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single forward pass for a batch: returns SAE feature activations at the
    last token position, output entropy, and a boolean mask for correct answers.

    Returns:
        feature_acts:  [batch, n_features]
        entropy:       [batch]
        correct_mask:  [batch] bool
    """
    feature_acts_list = []

    def cache_hook(value, hook):
        # value: [batch, seq, d_model] — residual stream at hook_name
        acts = sae.encode(value)          # [batch, seq, n_features]
        feature_acts_list.append(acts[:, -1, :].detach().cpu())  # last token only
        return value  # do not modify; pure observation

    with torch.no_grad():
        logits = model.run_with_hooks(
            batch.input_ids,
            fwd_hooks=[(hook_name, cache_hook)],
        )

    entropy = output_entropy(logits).cpu()                      # [batch]
    last_logits = logits[:, -1, :].cpu()
    correct_mask = correct_answer_in_top_k(                     # [batch] bool
        last_logits, batch.correct_token_ids.cpu(), k=5
    )

    return feature_acts_list[0], entropy, correct_mask


def run_phase1(
    cfg: ExperimentConfig,
    model: HookedTransformer,
    saes: dict[int, SAE],
    hook_names: dict[int, str],
    data: PromptBatch,
) -> dict:
    """
    Run the full correlational phase.

    Args:
        cfg:   experiment config
        model: loaded HookedTransformer
        saes:  {layer_idx: SAE} for each layer in cfg.model.layers_to_analyze
        data:  full PromptBatch

    Returns:
        results dict (also saved to checkpoint)
    """
    dev = device()
    batch_size = cfg.batch_size
    n_prompts = len(data.questions)

    # Accumulate entropy, correct_mask across all batches first
    all_entropy = []
    all_correct = []

    # Also accumulate per-layer feature activations: {layer: list of [batch, n_feat]}
    layer_acts: dict[int, list[torch.Tensor]] = {
        layer: [] for layer in cfg.model.layers_to_analyze
    }

    print(f"[phase1] running {n_prompts} prompts in batches of {batch_size}...")

    for i, mini_batch in enumerate(get_batches(data, batch_size)):
        print(f"  batch {i+1}/{(n_prompts + batch_size - 1) // batch_size}", end="\r")

        # Collect activations for all layers in one pass per batch
        acts_this_batch: dict[int, list] = {l: [] for l in cfg.model.layers_to_analyze}
        entropy_this_batch = None
        correct_this_batch = None

        for layer_idx in cfg.model.layers_to_analyze:
            hook_name = hook_names[layer_idx]
            sae = saes[layer_idx]

            acts, entropy, correct = _extract_sae_activations_and_entropy(
                model, sae, hook_name, mini_batch, dev
            )
            acts_this_batch[layer_idx] = acts   # [batch, n_features]

            # Entropy and correct_mask are the same regardless of layer;
            # only record once (from the last layer, since they share the same logits)
            entropy_this_batch = entropy
            correct_this_batch = correct

        for layer_idx in cfg.model.layers_to_analyze:
            layer_acts[layer_idx].append(acts_this_batch[layer_idx])

        all_entropy.append(entropy_this_batch)
        all_correct.append(correct_this_batch)

    print()  # newline after \r progress

    entropy_scores = torch.cat(all_entropy, dim=0)       # [n_prompts]
    correct_mask = torch.cat(all_correct, dim=0)          # [n_prompts] bool

    print(f"[phase1] mean entropy: {entropy_scores.mean():.3f} | "
          f"correct@1: {correct_mask.float().mean():.3f}")

    # --- Correlational analysis per layer ---
    entropy_np = entropy_scores.numpy()
    per_layer = {}

    for layer_idx in cfg.model.layers_to_analyze:
        acts = torch.cat(layer_acts[layer_idx], dim=0).numpy()  # [n_prompts, n_features]
        n_features = acts.shape[1]

        print(f"[phase1] layer {layer_idx}: computing Spearman correlations "
              f"for {n_features} features...")

        correlations = np.zeros(n_features, dtype=np.float32)
        for feat_idx in range(n_features):
            rho, _ = spearmanr(acts[:, feat_idx], entropy_np)
            # NaN arises when a feature is always zero; treat as no correlation
            correlations[feat_idx] = rho if not np.isnan(rho) else 0.0

        correlations_t = torch.from_numpy(correlations)
        top_k = min(cfg.top_k_features, n_features)
        top_correlations, top_feature_ids = torch.topk(correlations_t.abs(), top_k)
        # Restore sign
        top_correlations = correlations_t[top_feature_ids]

        per_layer[layer_idx] = {
            "correlations": correlations_t,
            "top_feature_ids": top_feature_ids,
            "top_correlations": top_correlations,
        }

        print(f"  top-5 corr: {top_correlations[:5].tolist()}")

    results = {
        "entropy_scores": entropy_scores,
        "correct_mask": correct_mask,
        "per_layer": per_layer,
    }

    ckpt_path = f"{cfg.checkpoint_dir}/phase1_correlational.pt"
    save_checkpoint(results, ckpt_path)
    return results
