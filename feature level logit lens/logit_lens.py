from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.transcoder import Transcoder


def _get_final_norm_module(model: AutoModelForCausalLM) -> torch.nn.Module | None:
    """Return the final normalization module used before unembedding for common architectures."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "final_layer_norm"):
        return model.gpt_neox.final_layer_norm
    return None


def _apply_pre_unembed_norm(hidden: Tensor, final_norm: torch.nn.Module | None) -> Tensor:
    if final_norm is None:
        return hidden
    return final_norm(hidden)


def _clean_token(tokenizer: AutoTokenizer, token_id: int) -> str:
    token = tokenizer.decode([token_id], skip_special_tokens=False)
    token = token.replace("\n", "\\n")
    return token if token else "<empty>"


def compute_logit_lens_next_token(
    model_name: str,
    prompt: str,
    device: torch.device,
    top_k: int = 10,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)

    hidden_states = outputs.hidden_states
    if hidden_states is None or len(hidden_states) <= 1:
        raise RuntimeError("Model did not return per-layer hidden states.")

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("Model does not expose output embeddings for logit lens.")

    final_norm = _get_final_norm_module(model)

    # Exclude the embedding layer; keep transformer block outputs only.
    layer_hidden_states = hidden_states[1:]
    layer_indices = list(range(1, len(layer_hidden_states) + 1))

    # Use final layer top-k tokens as stable columns in the heatmap.
    final_hidden = _apply_pre_unembed_norm(layer_hidden_states[-1], final_norm)
    final_logits = lm_head(final_hidden[:, -1, :])
    final_probs = torch.softmax(final_logits, dim=-1)

    k = min(top_k, final_probs.shape[-1])
    topk = torch.topk(final_probs, k=k, dim=-1)
    candidate_token_ids = topk.indices[0].tolist()
    candidate_labels = [_clean_token(tokenizer, token_id) for token_id in candidate_token_ids]

    probability_rows: list[list[float]] = []
    top_predictions: list[dict[str, Any]] = []

    for layer_idx, hidden in zip(layer_indices, layer_hidden_states):
        normed_hidden = _apply_pre_unembed_norm(hidden, final_norm)
        layer_logits = lm_head(normed_hidden[:, -1, :])
        layer_probs = torch.softmax(layer_logits, dim=-1)

        row = layer_probs[0, candidate_token_ids].detach().cpu().tolist()
        probability_rows.append(row)

        top_prob, top_token_id = torch.max(layer_probs[0], dim=-1)
        top_predictions.append(
            {
                "layer": layer_idx,
                "token_id": int(top_token_id.item()),
                "token": _clean_token(tokenizer, int(top_token_id.item())),
                "prob": float(top_prob.item()),
            }
        )

    next_token_id = int(torch.argmax(final_probs[0]).item())
    next_token_text = _clean_token(tokenizer, next_token_id)

    return {
        "prompt": prompt,
        "model_name": model_name,
        "candidate_labels": candidate_labels,
        "candidate_token_ids": candidate_token_ids,
        "layer_indices": layer_indices,
        "probability_rows": probability_rows,
        "top_predictions": top_predictions,
        "final_next_token_id": next_token_id,
        "final_next_token_text": next_token_text,
    }


def save_logit_lens_heatmap(
    layer_indices: list[int],
    candidate_labels: list[str],
    probability_rows: list[list[float]],
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig_w = max(8, 0.7 * len(candidate_labels))
    fig_h = max(5, 0.22 * len(layer_indices))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    tensor = torch.tensor(probability_rows, dtype=torch.float32)
    image = ax.imshow(tensor.numpy(), aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(candidate_labels)))
    ax.set_xticklabels(candidate_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(layer_indices)))
    ax.set_yticklabels(layer_indices)

    ax.set_xlabel("Candidate next tokens (top-k from final layer)")
    ax.set_ylabel("Layer index")
    ax.set_title(title)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("P(token | layer hidden state)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def compute_feature_logit_lens_next_token(
    model_name: str,
    prompt: str,
    device: torch.device,
    transcoder: Transcoder,
    feature_top_n: int = 16,
    feature_layer: int = -1,
    top_k: int = 10,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)

    hidden_states = outputs.hidden_states
    if hidden_states is None or len(hidden_states) <= 1:
        raise RuntimeError("Model did not return per-layer hidden states.")

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("Model does not expose output embeddings for logit lens.")

    final_norm = _get_final_norm_module(model)

    # Exclude embedding layer to align with transformer block indexing.
    layer_hidden_states = hidden_states[1:]
    num_layers = len(layer_hidden_states)

    if feature_layer == -1:
        selected_layer_zero_idx = num_layers - 1
    elif 1 <= feature_layer <= num_layers:
        selected_layer_zero_idx = feature_layer - 1
    else:
        raise ValueError(f"feature_layer must be -1 or in [1, {num_layers}], got {feature_layer}")

    selected_layer_one_idx = selected_layer_zero_idx + 1
    selected_hidden = layer_hidden_states[selected_layer_zero_idx][:, -1, :]

    model_dtype = selected_hidden.dtype
    model_device = selected_hidden.device

    transcoder_device = next(transcoder.parameters()).device
    transcoder_dtype = next(transcoder.parameters()).dtype

    with torch.no_grad():
        feature_acts = transcoder.encode(selected_hidden.to(device=transcoder_device, dtype=transcoder_dtype))[0]

    if feature_acts.numel() == 0:
        raise RuntimeError("Transcoder produced empty feature activations.")

    feature_count = int(feature_acts.shape[0])
    k_features = min(feature_top_n, feature_count)
    top_feature_vals, top_feature_ids = torch.topk(feature_acts, k=k_features, dim=0)

    # Use top-k next-token candidates from the selected whole-layer hidden state as stable columns.
    selected_normed = _apply_pre_unembed_norm(layer_hidden_states[selected_layer_zero_idx], final_norm)
    selected_logits = lm_head(selected_normed[:, -1, :])
    selected_probs = torch.softmax(selected_logits, dim=-1)

    k_tokens = min(top_k, selected_probs.shape[-1])
    selected_topk = torch.topk(selected_probs, k=k_tokens, dim=-1)
    candidate_token_ids = selected_topk.indices[0].tolist()
    candidate_labels = [_clean_token(tokenizer, token_id) for token_id in candidate_token_ids]

    feature_rows: list[list[float]] = []
    feature_summaries: list[dict[str, Any]] = []

    decoder_weight = transcoder.W_dec.weight.data

    for feat_id, feat_val in zip(top_feature_ids.tolist(), top_feature_vals.tolist()):
        # Single-feature decoded contribution in residual stream space.
        contribution = decoder_weight[:, feat_id] * feat_val
        contribution = contribution.to(device=model_device, dtype=model_dtype)
        contribution = contribution.unsqueeze(0).unsqueeze(0)

        contribution_normed = _apply_pre_unembed_norm(contribution, final_norm)
        feat_logits = lm_head(contribution_normed[:, -1, :])
        feat_probs = torch.softmax(feat_logits, dim=-1)

        row = feat_probs[0, candidate_token_ids].detach().cpu().tolist()
        feature_rows.append(row)

        top_prob, top_tok = torch.max(feat_probs[0], dim=-1)
        feature_summaries.append(
            {
                "feature_id": int(feat_id),
                "feature_value": float(feat_val),
                "top_token_id": int(top_tok.item()),
                "top_token": _clean_token(tokenizer, int(top_tok.item())),
                "top_prob": float(top_prob.item()),
            }
        )

    return {
        "prompt": prompt,
        "model_name": model_name,
        "selected_layer": selected_layer_one_idx,
        "candidate_token_ids": candidate_token_ids,
        "candidate_labels": candidate_labels,
        "feature_rows": feature_rows,
        "feature_summaries": feature_summaries,
    }


def save_feature_lens_heatmap(
    feature_summaries: list[dict[str, Any]],
    candidate_labels: list[str],
    feature_rows: list[list[float]],
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig_w = max(8, 0.7 * len(candidate_labels))
    fig_h = max(5, 0.45 * max(len(feature_summaries), 1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    tensor = torch.tensor(feature_rows, dtype=torch.float32)
    image = ax.imshow(tensor.numpy(), aspect="auto", cmap="magma")

    ax.set_xticks(range(len(candidate_labels)))
    ax.set_xticklabels(candidate_labels, rotation=45, ha="right")

    y_labels = [
        f"f{item['feature_id']} ({item['feature_value']:.3f})"
        for item in feature_summaries
    ]
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)

    ax.set_xlabel("Candidate next tokens")
    ax.set_ylabel("Top activated features")
    ax.set_title(title)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("P(token | single-feature contribution)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
