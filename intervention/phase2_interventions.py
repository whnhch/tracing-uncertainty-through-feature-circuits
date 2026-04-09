"""
Phase 2: Causal Intervention (Ablation / Amplification)

For each candidate uncertainty feature identified in Phase 1:
  - Select prompts where the model is confidently correct (low entropy, correct@1)
  - Run ablation: zero out feature's contribution to the residual stream
  - Run amplification: scale feature's contribution by each factor in cfg.amplification_factors
  - Record pre- and post-intervention entropy, top-k token ids, and correct_mask

Output (saved to checkpoint):
  {
    "confident_prompt_indices": Tensor [n_confident],
    "per_layer": {
        layer_idx: {
            feature_idx: {
                "pre_entropy":          Tensor [n_confident],
                "pre_top_k":            Tensor [n_confident, k],
                "ablation": {
                    "post_entropy":     Tensor [n_confident],
                    "post_top_k":       Tensor [n_confident, k],
                    "correct_mask":     Tensor [n_confident] bool,
                },
                "amplification": {
                    scale: {
                        "post_entropy": Tensor [n_confident],
                        "post_top_k":   Tensor [n_confident, k],
                        "correct_mask": Tensor [n_confident] bool,
                    }
                    for scale in cfg.amplification_factors
                },
            }
        }
    }
  }
"""

import torch
from functools import partial
from transformer_lens import HookedTransformer
from sae_lens import SAE

from config import ExperimentConfig, device
from dataset import PromptBatch, get_batches
from utils import (
    output_entropy,
    correct_answer_in_top_k,
    ablate_feature,
    amplify_feature,
    save_checkpoint,
    load_checkpoint,
)


def _run_intervention_batch(
    model: HookedTransformer,
    sae: SAE,
    hook_name: str,
    input_ids: torch.Tensor,
    correct_token_ids: torch.Tensor,
    feature_idx: int,
    mode: str,           # "ablation" or "amplification"
    scale: float = 1.0,  # used only for amplification
    top_k: int = 5,
) -> dict:
    """
    Run one intervention on a single batch.

    Returns dict with post_entropy, post_top_k, correct_mask.
    """
    def intervention_hook(value, hook):
        if mode == "ablation":
            return ablate_feature(value, sae, feature_idx)
        else:
            return amplify_feature(value, sae, feature_idx, scale)

    with torch.no_grad():
        logits = model.run_with_hooks(
            input_ids,
            fwd_hooks=[(hook_name, intervention_hook)],
        )

    last_logits = logits[:, -1, :].cpu()
    post_entropy = output_entropy(last_logits)
    _, post_top_k = torch.topk(last_logits.float(), top_k, dim=-1)
    correct_mask = correct_answer_in_top_k(
        last_logits, correct_token_ids.cpu(), k=top_k
    )

    return {
        "post_entropy": post_entropy,
        "post_top_k": post_top_k,
        "correct_mask": correct_mask,
    }


def run_phase2(
    cfg: ExperimentConfig,
    model: HookedTransformer,
    saes: dict[int, SAE],
    hook_names: dict[int, str],
    data: PromptBatch,
    phase1_results: dict,
) -> dict:
    """
    Run causal interventions on the top candidate features from Phase 1.

    Args:
        cfg:             experiment config
        model:           loaded HookedTransformer
        saes:            {layer_idx: SAE}
        data:            full PromptBatch
        phase1_results:  output of run_phase1

    Returns:
        results dict (also saved to checkpoint)
    """
    dev = device()
    top_k = cfg.coherence_top_k

    # Select confident+correct prompts
    entropy_scores = phase1_results["entropy_scores"]
    correct_mask = phase1_results["correct_mask"]

    confident_mask = (
        (entropy_scores < cfg.entropy_confident_threshold) & correct_mask
    )
    confident_indices = torch.where(confident_mask)[0]

    # Cap at n_confident_prompts
    if len(confident_indices) > cfg.n_confident_prompts:
        confident_indices = confident_indices[:cfg.n_confident_prompts]

    print(f"[phase2] {len(confident_indices)} confident+correct prompts selected")

    # Build a sub-PromptBatch for confident prompts
    confident_data = PromptBatch(
        questions=[data.questions[i] for i in confident_indices.tolist()],
        prompts=[data.prompts[i] for i in confident_indices.tolist()],
        answers=[data.answers[i] for i in confident_indices.tolist()],
        input_ids=data.input_ids[confident_indices],
        attention_mask=data.attention_mask[confident_indices],
        correct_token_ids=data.correct_token_ids[confident_indices],
    )

    # Compute pre-intervention entropy and top-k once (shared across all features)
    print("[phase2] computing pre-intervention baseline...")
    pre_entropy_list = []
    pre_top_k_list = []

    with torch.no_grad():
        for mini in get_batches(confident_data, cfg.batch_size):
            logits = model(mini.input_ids)
            last = logits[:, -1, :].cpu().float()
            pre_entropy_list.append(output_entropy(last))
            _, tk = torch.topk(last, top_k, dim=-1)
            pre_top_k_list.append(tk)

    pre_entropy = torch.cat(pre_entropy_list)   # [n_confident]
    pre_top_k = torch.cat(pre_top_k_list)       # [n_confident, top_k]

    per_layer = {}

    for layer_idx in cfg.model.layers_to_analyze:
        hook_name = hook_names[layer_idx]
        sae = saes[layer_idx]
        top_feature_ids = phase1_results["per_layer"][layer_idx]["top_feature_ids"]

        print(f"[phase2] layer {layer_idx}: intervening on "
              f"{len(top_feature_ids)} features...")

        layer_results = {}

        for rank, feature_idx in enumerate(top_feature_ids.tolist()):
            print(f"  feature {feature_idx} (rank {rank+1}/{len(top_feature_ids)})", end="\r")

            feature_result: dict = {
                "pre_entropy": pre_entropy,
                "pre_top_k": pre_top_k,
            }

            # --- Ablation ---
            abl_entropy, abl_top_k, abl_correct = [], [], []
            for mini in get_batches(confident_data, cfg.batch_size):
                out = _run_intervention_batch(
                    model, sae, hook_name,
                    mini.input_ids, mini.correct_token_ids,
                    feature_idx, mode="ablation", top_k=top_k,
                )
                abl_entropy.append(out["post_entropy"])
                abl_top_k.append(out["post_top_k"])
                abl_correct.append(out["correct_mask"])

            feature_result["ablation"] = {
                "post_entropy": torch.cat(abl_entropy),
                "post_top_k": torch.cat(abl_top_k),
                "correct_mask": torch.cat(abl_correct),
            }

            # --- Amplification ---
            amp_results = {}
            for scale in cfg.amplification_factors:
                amp_entropy, amp_top_k, amp_correct = [], [], []
                for mini in get_batches(confident_data, cfg.batch_size):
                    out = _run_intervention_batch(
                        model, sae, hook_name,
                        mini.input_ids, mini.correct_token_ids,
                        feature_idx, mode="amplification", scale=scale, top_k=top_k,
                    )
                    amp_entropy.append(out["post_entropy"])
                    amp_top_k.append(out["post_top_k"])
                    amp_correct.append(out["correct_mask"])

                amp_results[scale] = {
                    "post_entropy": torch.cat(amp_entropy),
                    "post_top_k": torch.cat(amp_top_k),
                    "correct_mask": torch.cat(amp_correct),
                }

            feature_result["amplification"] = amp_results
            layer_results[feature_idx] = feature_result

        print()  # newline after \r
        per_layer[layer_idx] = layer_results

    results = {
        "confident_prompt_indices": confident_indices,
        "per_layer": per_layer,
    }

    ckpt_path = f"{cfg.checkpoint_dir}/phase2_interventions.pt"
    save_checkpoint(results, ckpt_path)
    return results
