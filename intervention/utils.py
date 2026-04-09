"""
Shared utilities used across all pipeline phases.
"""

import torch
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

def output_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute Shannon entropy of the next-token distribution from the last
    sequence position.

    Args:
        logits: [batch, seq, vocab] or [batch, vocab]

    Returns:
        entropy: [batch]  (nats)
    """
    if logits.dim() == 3:
        logits = logits[:, -1, :]   # take last position
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def logit_gap(logits: torch.Tensor) -> torch.Tensor:
    """
    Difference in probability between the top-1 and top-2 predicted tokens.
    High gap → high confidence.

    Args:
        logits: [batch, seq, vocab] or [batch, vocab]

    Returns:
        gap: [batch]
    """
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    probs = F.softmax(logits.float(), dim=-1)
    top2, _ = torch.topk(probs, 2, dim=-1)
    return top2[:, 0] - top2[:, 1]


# ---------------------------------------------------------------------------
# Top-k coherence check (Phase 3, test 1)
# ---------------------------------------------------------------------------

def correct_answer_in_top_k(
    logits: torch.Tensor,
    correct_token_ids: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """
    Check whether the correct answer token is still among the top-k predictions
    after an intervention. Used to distinguish calibration-appropriate uncertainty
    from degradation/drift.

    Args:
        logits:            [batch, vocab] — post-intervention last-position logits
        correct_token_ids: [batch]        — token id of the expected answer
        k:                 int

    Returns:
        mask: [batch] bool, True where correct token is in top-k
    """
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    _, top_k_indices = torch.topk(logits.float(), k, dim=-1)   # [batch, k]
    match = (top_k_indices == correct_token_ids.unsqueeze(1))  # [batch, k]
    return match.any(dim=-1)


# ---------------------------------------------------------------------------
# Hedging detection (Phase 3, test 4)
# ---------------------------------------------------------------------------

HEDGING_PHRASES: list[str] = [
    "i think", "i believe", "i'm not sure", "i am not sure",
    "probably", "perhaps", "maybe", "might", "could be",
    "it's possible", "it is possible", "not certain", "not sure",
    "i'm uncertain", "i am uncertain", "i'm unsure", "i am unsure",
    "hard to say", "difficult to say", "it depends",
]


def contains_hedging(text: str) -> bool:
    """Return True if the text contains any known hedging phrase."""
    lowered = text.lower()
    return any(phrase in lowered for phrase in HEDGING_PHRASES)


def hedging_rate(texts: list[str]) -> float:
    """Fraction of texts in a list that contain hedging."""
    if not texts:
        return 0.0
    return sum(contains_hedging(t) for t in texts) / len(texts)


# ---------------------------------------------------------------------------
# SAE feature intervention helpers
# ---------------------------------------------------------------------------

def ablate_feature(
    residual: torch.Tensor,
    sae,
    feature_idx: int,
) -> torch.Tensor:
    """
    Zero out the contribution of a single SAE feature from the residual stream.

    The SAE approximates x ≈ b_dec + Σ_i f_i * W_dec[i].
    Ablation removes the term for feature_idx:
        x_ablated = x - f[feature_idx] * W_dec[feature_idx]

    Args:
        residual:    [..., d_model]
        sae:         SAELens SAE object
        feature_idx: which feature to ablate

    Returns:
        modified residual [..., d_model]
    """
    feature_acts = sae.encode(residual)                        # [..., n_features]
    act = feature_acts[..., feature_idx]                       # [...] scalar per position
    direction = sae.W_dec[feature_idx]                        # [d_model]
    contribution = act.unsqueeze(-1) * direction               # [..., d_model]
    return residual - contribution


def amplify_feature(
    residual: torch.Tensor,
    sae,
    feature_idx: int,
    scale: float,
) -> torch.Tensor:
    """
    Scale the contribution of a single SAE feature by `scale`.

    x_amplified = x + (scale - 1) * f[feature_idx] * W_dec[feature_idx]

    Args:
        residual:    [..., d_model]
        sae:         SAELens SAE object
        feature_idx: which feature to amplify
        scale:       multiplicative factor (e.g. 2.0, 5.0)

    Returns:
        modified residual [..., d_model]
    """
    feature_acts = sae.encode(residual)
    act = feature_acts[..., feature_idx]
    direction = sae.W_dec[feature_idx]
    contribution = act.unsqueeze(-1) * direction
    return residual + (scale - 1.0) * contribution


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

import os
import json
from pathlib import Path


def save_checkpoint(data: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    print(f"[checkpoint] saved → {path}")


def load_checkpoint(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    data = torch.load(path, map_location="cpu")
    print(f"[checkpoint] loaded ← {path}")
    return data


def checkpoint_exists(path: str) -> bool:
    return os.path.exists(path)
