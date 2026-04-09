"""
Phase 4: Propagation Analysis

Starting from the calibration-appropriate features identified in Phase 3,
trace where the uncertainty signal is introduced in the residual stream and
how it evolves across layers.

Two analyses:

  4a — Layer potency sweep:
    For each calibration-appropriate feature at its home layer L, run the
    same intervention (ablation) at every OTHER layer and measure the
    resulting entropy change. This identifies where in the residual stream
    the feature's effect is most concentrated.

  4b — Redundancy check:
    Ablate the feature at its home layer L. Then measure entropy at layers
    L+2, L+4, … to see whether the uncertainty signal re-emerges (suggesting
    compensatory circuits).

Output (saved to checkpoint):
  {
    "calibration_appropriate_features": {
        layer_idx: [feature_idx, ...]
    },
    "layer_potency": {
        (home_layer, feature_idx): {
            intervention_layer: mean_entropy_delta,
            ...
        }
    },
    "redundancy": {
        (home_layer, feature_idx): {
            probe_layer: mean_entropy_delta_after_home_ablation,
            ...
        }
    }
  }
"""

import torch
import numpy as np
from functools import partial
from transformer_lens import HookedTransformer
from sae_lens import SAE

from config import ExperimentConfig, device
from dataset import PromptBatch, get_batches
from utils import (
    output_entropy,
    ablate_feature,
    save_checkpoint,
)
from phase3_calibration import CALIBRATION_APPROPRIATE


def _mean_entropy_after_ablation(
    model: HookedTransformer,
    sae: SAE,
    hook_name: str,
    feature_idx: int,
    data: PromptBatch,
    batch_size: int,
) -> float:
    """
    Ablate feature_idx at hook_name and return the mean output entropy over data.
    """
    def ablation_hook(value, hook):
        return ablate_feature(value, sae, feature_idx)

    entropies = []
    with torch.no_grad():
        for mini in get_batches(data, batch_size):
            logits = model.run_with_hooks(
                mini.input_ids,
                fwd_hooks=[(hook_name, ablation_hook)],
            )
            entropies.append(output_entropy(logits[:, -1, :].cpu()))

    return torch.cat(entropies).mean().item()


def _probe_residual_entropy_after_ablation(
    model: HookedTransformer,
    home_sae: SAE,
    home_hook: str,
    probe_hook: str,
    feature_idx: int,
    data: PromptBatch,
    batch_size: int,
) -> float:
    """
    Ablate feature at home_hook, then read the logit-implied entropy as
    reconstructed from the residual stream AT probe_hook (via the logit lens).

    Uses a capture hook inside run_with_hooks to record the residual at
    probe_hook mid-forward-pass alongside the ablation hook.
    """
    def ablation_hook(value, hook):
        return ablate_feature(value, home_sae, feature_idx)

    entropies = []
    with torch.no_grad():
        for mini in get_batches(data, batch_size):
            captured = []

            def capture_hook(value, hook):
                captured.append(value[:, -1, :].detach().cpu())  # last token
                return value

            model.run_with_hooks(
                mini.input_ids,
                fwd_hooks=[
                    (home_hook, ablation_hook),
                    (probe_hook, capture_hook),
                ],
            )
            probe_residual = captured[0]   # [batch, d_model]
            # Project through final layer norm and unembed (logit lens)
            probe_logits = model.unembed(
                model.ln_final(probe_residual.to(model.cfg.device))
            )
            entropies.append(output_entropy(probe_logits.cpu()))

    return torch.cat(entropies).mean().item()


def _baseline_entropy(
    model: HookedTransformer,
    data: PromptBatch,
    batch_size: int,
) -> float:
    entropies = []
    with torch.no_grad():
        for mini in get_batches(data, batch_size):
            logits = model(mini.input_ids)
            entropies.append(output_entropy(logits[:, -1, :].cpu()))
    return torch.cat(entropies).mean().item()


def run_phase4(
    cfg: ExperimentConfig,
    model: HookedTransformer,
    saes: dict[int, SAE],
    hook_names: dict[int, str],
    data: PromptBatch,
    phase2_results: dict,
    phase3_results: dict,
) -> dict:
    """
    Run propagation analysis on calibration-appropriate features.

    Args:
        cfg:             experiment config
        model:           loaded HookedTransformer
        saes:            {layer_idx: SAE}
        data:            full PromptBatch (or the confident subset)
        phase2_results:  output of run_phase2
        phase3_results:  output of run_phase3

    Returns:
        results dict (also saved to checkpoint)
    """
    dev = device()
    batch_size = cfg.batch_size
    all_layers = cfg.model.layers_to_analyze

    # Collect calibration-appropriate features
    ca_features: dict[int, list[int]] = {}
    for layer_idx, layer_data in phase3_results["per_layer"].items():
        ca_for_layer = []
        for feat_idx, feat_data in layer_data.items():
            # Check ablation result; use ablation as the primary classification
            if feat_data.get("ablation", {}).get("label") == CALIBRATION_APPROPRIATE:
                ca_for_layer.append(feat_idx)
        if ca_for_layer:
            ca_features[layer_idx] = ca_for_layer

    print(f"[phase4] calibration-appropriate features: "
          f"{ {l: len(f) for l, f in ca_features.items()} }")

    # Use confident prompts only
    confident_indices = phase2_results["confident_prompt_indices"]
    confident_data = PromptBatch(
        questions=[data.questions[i] for i in confident_indices.tolist()],
        prompts=[data.prompts[i] for i in confident_indices.tolist()],
        answers=[data.answers[i] for i in confident_indices.tolist()],
        input_ids=data.input_ids[confident_indices],
        attention_mask=data.attention_mask[confident_indices],
        correct_token_ids=data.correct_token_ids[confident_indices],
    )

    baseline = _baseline_entropy(model, confident_data, batch_size)
    print(f"[phase4] baseline entropy on confident prompts: {baseline:.4f}")

    layer_potency: dict[tuple, dict[int, float]] = {}
    redundancy: dict[tuple, dict[int, float]] = {}

    for home_layer, feature_list in ca_features.items():
        for feature_idx in feature_list:
            key = (home_layer, feature_idx)
            print(f"[phase4] layer {home_layer}, feature {feature_idx}")

            # 4a: Layer potency sweep — ablate at each layer, measure output entropy
            potency = {}
            for probe_layer in all_layers:
                probe_hook = hook_names[probe_layer]
                probe_sae = saes[probe_layer]
                mean_ent = _mean_entropy_after_ablation(
                    model, probe_sae, probe_hook, feature_idx,
                    confident_data, batch_size,
                )
                potency[probe_layer] = mean_ent - baseline
                print(f"  ablate@layer{probe_layer}: Δentropy={potency[probe_layer]:+.4f}")

            layer_potency[key] = potency

            # 4b: Redundancy check — ablate at home layer, probe residual at later layers
            home_hook = hook_names[home_layer]
            home_sae = saes[home_layer]
            redund = {}

            probe_layers_after = [l for l in all_layers if l > home_layer]
            for probe_layer in probe_layers_after:
                probe_hook = hook_names[probe_layer]
                mid_ent = _probe_residual_entropy_after_ablation(
                    model, home_sae, home_hook, probe_hook,
                    feature_idx, confident_data, batch_size,
                )
                redund[probe_layer] = mid_ent - baseline
                print(f"  redundancy probe@layer{probe_layer}: "
                      f"Δentropy={redund[probe_layer]:+.4f}")

            redundancy[key] = redund

    results = {
        "calibration_appropriate_features": ca_features,
        "baseline_entropy": baseline,
        "layer_potency": layer_potency,
        "redundancy": redundancy,
    }

    ckpt_path = f"{cfg.checkpoint_dir}/phase4_propagation.pt"
    save_checkpoint(results, ckpt_path)
    return results


def summarize_phase4(results: dict) -> None:
    """Print the most potent layers and any redundancy signals."""
    print("\n=== Phase 4: Layer Potency ===")
    for (home_layer, feat_idx), potency in results["layer_potency"].items():
        best_layer = max(potency, key=lambda l: abs(potency[l]))
        print(f"  home=layer{home_layer} feat={feat_idx}: "
              f"most potent at layer {best_layer} "
              f"(Δentropy={potency[best_layer]:+.4f})")

    print("\n=== Phase 4: Redundancy ===")
    for (home_layer, feat_idx), redund in results["redundancy"].items():
        recovering = {l: d for l, d in redund.items() if d < -0.05}
        if recovering:
            print(f"  home=layer{home_layer} feat={feat_idx}: "
                  f"signal partially recovered at layers {list(recovering.keys())} "
                  f"→ possible compensatory circuit")
        else:
            print(f"  home=layer{home_layer} feat={feat_idx}: no recovery detected")
    print()
