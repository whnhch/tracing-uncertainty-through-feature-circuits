from __future__ import annotations

import os
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


def _resolve_target_token_id(tokenizer: AutoTokenizer, target_token: str) -> tuple[int, str]:
    token_ids = tokenizer.encode(target_token, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Could not tokenize target token {target_token!r}.")
    if len(token_ids) > 1:
        # Use the first sub-token to keep the visualization simple and deterministic.
        token_id = int(token_ids[0])
    else:
        token_id = int(token_ids[0])
    return token_id, _clean_token(tokenizer, token_id)


def compute_logit_lens_next_token(
    model_name: str,
    prompt: str,
    device: torch.device,
    top_k: int = 10,
) -> dict[str, Any]:
    
    # Load model and tokenizer, prepare inputs, and compute hidden states.
    print(f"Loading model '{model_name}' on device {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer loaded. Vocabulary size: {tokenizer.vocab_size}. Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            low_cpu_mem_usage=True,         # prevents CPU spike
            # use_safetensors=False           # avoids NFS mmap crash
        ) 
    print(f"Model loaded. Preparing inputs and computing hidden states...")
    model.to(device)
    model.eval()

    print(f"Computing logit lens for model '{model_name}' and prompt: {prompt!r}")
    
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


def compute_feature_token_trajectory_by_layer(
    model_name: str,
    prompt: str,
    device: torch.device,
    transcoder: Transcoder,
    target_token: str = "2",
    feature_top_n: int = 10,
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
    layer_hidden_states = hidden_states[1:]
    layer_indices = list(range(1, len(layer_hidden_states) + 1))

    target_token_id, target_token_label = _resolve_target_token_id(tokenizer, target_token)

    model_dtype = layer_hidden_states[0].dtype
    model_device = layer_hidden_states[0].device
    transcoder_device = next(transcoder.parameters()).device
    transcoder_dtype = next(transcoder.parameters()).dtype

    feature_acts_by_layer: list[Tensor] = []
    with torch.no_grad():
        for layer_hidden in layer_hidden_states:
            layer_last_hidden = layer_hidden[:, -1, :]
            feature_acts = transcoder.encode(
                layer_last_hidden.to(device=transcoder_device, dtype=transcoder_dtype)
            )[0]
            feature_acts_by_layer.append(feature_acts.detach().cpu())

    feature_act_matrix = torch.stack(feature_acts_by_layer, dim=0)
    if feature_act_matrix.numel() == 0:
        raise RuntimeError("Transcoder produced empty feature activations.")

    feature_count = int(feature_act_matrix.shape[1])
    k_features = min(feature_top_n, feature_count)
    max_acts, _ = torch.max(feature_act_matrix, dim=0)
    top_vals, top_feature_ids = torch.topk(max_acts, k=k_features, dim=0)

    decoder_weight = transcoder.W_dec.weight.data
    token_prob_rows: list[list[float]] = []
    activation_rows: list[list[float]] = []
    feature_summaries: list[dict[str, Any]] = []

    for feat_id, peak_act in zip(top_feature_ids.tolist(), top_vals.tolist()):
        per_layer_probs: list[float] = []
        per_layer_acts: list[float] = []

        for layer_pos, _layer_idx in enumerate(layer_indices):
            feat_val = float(feature_act_matrix[layer_pos, feat_id].item())
            per_layer_acts.append(feat_val)

            contribution = decoder_weight[:, feat_id] * feat_val
            contribution = contribution.to(device=model_device, dtype=model_dtype)
            contribution = contribution.unsqueeze(0).unsqueeze(0)

            contribution_normed = _apply_pre_unembed_norm(contribution, final_norm)
            feat_logits = lm_head(contribution_normed[:, -1, :])
            feat_probs = torch.softmax(feat_logits, dim=-1)
            per_layer_probs.append(float(feat_probs[0, target_token_id].item()))

        best_layer_idx = int(torch.tensor(per_layer_probs).argmax().item())
        feature_summaries.append(
            {
                "feature_id": int(feat_id),
                "peak_activation": float(peak_act),
                "best_layer": int(layer_indices[best_layer_idx]),
                "best_token_prob": float(per_layer_probs[best_layer_idx]),
            }
        )
        token_prob_rows.append(per_layer_probs)
        activation_rows.append(per_layer_acts)

    return {
        "prompt": prompt,
        "model_name": model_name,
        "target_token": target_token,
        "target_token_id": target_token_id,
        "target_token_label": target_token_label,
        "layer_indices": layer_indices,
        "feature_ids": [int(x) for x in top_feature_ids.tolist()],
        "token_prob_rows": token_prob_rows,
        "activation_rows": activation_rows,
        "feature_summaries": feature_summaries,
    }


def save_feature_token_trajectory_heatmap(
    feature_ids: list[int],
    layer_indices: list[int],
    token_prob_rows: list[list[float]],
    output_path: Path,
    title: str,
    target_token_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig_w = max(10, 0.32 * len(layer_indices))
    fig_h = max(5, 0.45 * max(len(feature_ids), 1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    tensor = torch.tensor(token_prob_rows, dtype=torch.float32)
    image = ax.imshow(tensor.numpy(), aspect="auto", cmap="plasma")

    ax.set_xticks(range(len(layer_indices)))
    ax.set_xticklabels(layer_indices, rotation=0)
    ax.set_yticks(range(len(feature_ids)))
    ax.set_yticklabels([f"f{fid}" for fid in feature_ids])

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Top features")
    ax.set_title(title)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(f"P('{target_token_label}' | single-feature contribution)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def compute_layerwise_top_feature_token_effects(
    model_name: str,
    prompt: str,
    device: torch.device,
    transcoder: Transcoder,
    target_token: str = "2",
    feature_top_n: int = 10,
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
    layer_hidden_states = hidden_states[1:]
    layer_indices = list(range(1, len(layer_hidden_states) + 1))
    target_token_id, target_token_label = _resolve_target_token_id(tokenizer, target_token)

    model_dtype = layer_hidden_states[0].dtype
    model_device = layer_hidden_states[0].device
    transcoder_device = next(transcoder.parameters()).device
    transcoder_dtype = next(transcoder.parameters()).dtype
    decoder_weight = transcoder.W_dec.weight.data

    feature_count = int(transcoder.dict_size)
    k_features = min(feature_top_n, feature_count)

    token_prob_rows: list[list[float]] = [[0.0 for _ in layer_indices] for _ in range(k_features)]
    feature_id_rows: list[list[int]] = [[-1 for _ in layer_indices] for _ in range(k_features)]
    activation_rows: list[list[float]] = [[0.0 for _ in layer_indices] for _ in range(k_features)]

    per_layer_summaries: list[dict[str, Any]] = []

    with torch.no_grad():
        for layer_pos, layer_hidden in enumerate(layer_hidden_states):
            layer_last_hidden = layer_hidden[:, -1, :]
            feature_acts = transcoder.encode(
                layer_last_hidden.to(device=transcoder_device, dtype=transcoder_dtype)
            )[0]

            top_vals, top_ids = torch.topk(feature_acts, k=k_features, dim=0)

            current_layer_items: list[dict[str, Any]] = []
            for rank_idx, (feat_val_t, feat_id_t) in enumerate(zip(top_vals, top_ids)):
                feat_val = float(feat_val_t.item())
                feat_id = int(feat_id_t.item())

                contribution = decoder_weight[:, feat_id] * feat_val
                contribution = contribution.to(device=model_device, dtype=model_dtype)
                contribution = contribution.unsqueeze(0).unsqueeze(0)

                contribution_normed = _apply_pre_unembed_norm(contribution, final_norm)
                feat_logits = lm_head(contribution_normed[:, -1, :])
                feat_probs = torch.softmax(feat_logits, dim=-1)
                target_prob = float(feat_probs[0, target_token_id].item())

                token_prob_rows[rank_idx][layer_pos] = target_prob
                feature_id_rows[rank_idx][layer_pos] = feat_id
                activation_rows[rank_idx][layer_pos] = feat_val

                current_layer_items.append(
                    {
                        "rank": rank_idx + 1,
                        "feature_id": feat_id,
                        "activation": feat_val,
                        "target_prob": target_prob,
                    }
                )

            per_layer_summaries.append({"layer": int(layer_indices[layer_pos]), "top_features": current_layer_items})

    return {
        "prompt": prompt,
        "model_name": model_name,
        "target_token": target_token,
        "target_token_id": target_token_id,
        "target_token_label": target_token_label,
        "layer_indices": layer_indices,
        "rank_labels": [f"top-{i + 1}" for i in range(k_features)],
        "token_prob_rows": token_prob_rows,
        "feature_id_rows": feature_id_rows,
        "activation_rows": activation_rows,
        "per_layer_summaries": per_layer_summaries,
    }


def save_layerwise_top_feature_token_heatmap(
    rank_labels: list[str],
    layer_indices: list[int],
    token_prob_rows: list[list[float]],
    output_path: Path,
    title: str,
    target_token_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig_w = max(10, 0.32 * len(layer_indices))
    fig_h = max(5, 0.45 * max(len(rank_labels), 1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    tensor = torch.tensor(token_prob_rows, dtype=torch.float32)
    image = ax.imshow(tensor.numpy(), aspect="auto", cmap="cividis")

    ax.set_xticks(range(len(layer_indices)))
    ax.set_xticklabels(layer_indices, rotation=0)
    ax.set_yticks(range(len(rank_labels)))
    ax.set_yticklabels(rank_labels)

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Feature rank within each layer")
    ax.set_title(title)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(f"P('{target_token_label}' | layer-local top feature)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
