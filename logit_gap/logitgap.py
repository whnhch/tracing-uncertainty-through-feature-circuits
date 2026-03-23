from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch


@dataclass
class LogitGapResult:
    """
    Output of the standard feature-level logit gap computation.

    Shapes:
        logit_gaps:        [batch, pos, n_features] or [n_features]
        target_logits:     [batch, pos, n_features] or [n_features]
        competitor_logits: [batch, pos, n_features] or [n_features]
        competitor_ids:    [batch, pos, n_features] or [n_features]
    """

    logit_gaps: torch.Tensor
    target_logits: torch.Tensor
    competitor_logits: torch.Tensor
    competitor_ids: torch.Tensor


def compute_feature_vocab_directions(
    decoder_directions: torch.Tensor,
    unembedding: torch.Tensor,
) -> torch.Tensor:
    """
    Compute each feature's fixed projection into vocabulary space.

    If:
        decoder_directions = [n_features, d_model]
        unembedding        = [d_model, vocab_size]

    then:
        feature_vocab_directions = decoder_directions @ unembedding
                                = [n_features, vocab_size]

    This corresponds to the fixed term:
        W_U d_i
    up to row/column convention.
    """
    if decoder_directions.ndim != 2:
        raise ValueError(
            f"decoder_directions must have shape [n_features, d_model], got {tuple(decoder_directions.shape)}"
        )
    if unembedding.ndim != 2:
        raise ValueError(
            f"unembedding must have shape [d_model, vocab_size], got {tuple(unembedding.shape)}"
        )
    if decoder_directions.shape[1] != unembedding.shape[0]:
        raise ValueError(
            "decoder_directions and unembedding are incompatible: "
            f"{tuple(decoder_directions.shape)} vs {tuple(unembedding.shape)}"
        )

    return decoder_directions @ unembedding


def compute_feature_logit_attributions(
    feature_activations: torch.Tensor,
    decoder_directions: torch.Tensor,
    unembedding: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-feature logit attributions over the vocabulary.

    For feature i:
        contribution_i = z_i * (decoder_i @ unembedding)

    Args:
        feature_activations:
            Either:
                [batch, pos, n_features]
            or:
                [n_features]
        decoder_directions:
            [n_features, d_model]
        unembedding:
            [d_model, vocab_size]

    Returns:
        If input activations are [batch, pos, n_features]:
            [batch, pos, n_features, vocab_size]
        If input activations are [n_features]:
            [n_features, vocab_size]
    """
    original_1d = feature_activations.ndim == 1

    if original_1d:
        feature_activations = feature_activations.unsqueeze(0).unsqueeze(0)
    elif feature_activations.ndim != 3:
        raise ValueError(
            "feature_activations must have shape [batch, pos, n_features] "
            f"or [n_features], got {tuple(feature_activations.shape)}"
        )

    n_features = decoder_directions.shape[0]
    if feature_activations.shape[-1] != n_features:
        raise ValueError(
            "feature_activations last dimension must match decoder_directions first dimension: "
            f"{feature_activations.shape[-1]} != {n_features}"
        )

    # [n_features, vocab_size]
    feature_vocab_directions = compute_feature_vocab_directions(
        decoder_directions, unembedding
    )

    # [batch, pos, n_features, vocab_size]
    feature_logit_attributions = feature_activations.unsqueeze(
        -1
    ) * feature_vocab_directions.unsqueeze(0).unsqueeze(0)

    if original_1d:
        return feature_logit_attributions.squeeze(0).squeeze(0)

    return feature_logit_attributions


def _normalize_target_token_ids(
    target_token_ids: Union[int, torch.Tensor],
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Normalize target token ids to shape [batch, pos].

    Accepts:
        scalar
        [batch]
        [pos]
        [batch, pos]
    """
    if not isinstance(target_token_ids, torch.Tensor):
        target_token_ids = torch.tensor(target_token_ids, device=device)

    target_token_ids = target_token_ids.to(device).long()

    if target_token_ids.ndim == 0:
        target_token_ids = target_token_ids.view(1, 1).expand(batch_size, seq_len)

    elif target_token_ids.ndim == 1:
        if target_token_ids.shape[0] == batch_size:
            target_token_ids = target_token_ids.unsqueeze(1).expand(batch_size, seq_len)
        elif target_token_ids.shape[0] == seq_len:
            target_token_ids = target_token_ids.unsqueeze(0).expand(batch_size, seq_len)
        else:
            raise ValueError(
                "1D target_token_ids must have length batch_size or seq_len, got "
                f"{target_token_ids.shape[0]}"
            )

    elif target_token_ids.ndim == 2:
        if target_token_ids.shape != (batch_size, seq_len):
            raise ValueError(
                f"2D target_token_ids must have shape {(batch_size, seq_len)}, "
                f"got {tuple(target_token_ids.shape)}"
            )
    else:
        raise ValueError(
            "target_token_ids must be scalar, 1D, or 2D, got "
            f"{tuple(target_token_ids.shape)}"
        )

    return target_token_ids


def compute_logit_gaps(
    feature_logit_attributions: torch.Tensor,
    target_token_ids: Union[int, torch.Tensor],
) -> LogitGapResult:
    """
    Compute the standard feature-level targeted logit gap.

    For each feature i and target token t:
        LogitGap_{i,t} = logit_{i,t} - max_{w != t}(logit_{i,w})

    Args:
        feature_logit_attributions:
            Either:
                [batch, pos, n_features, vocab_size]
            or:
                [n_features, vocab_size]
        target_token_ids:
            Scalar, [batch], [pos], or [batch, pos]

    Returns:
        LogitGapResult containing:
            logit_gaps
            target_logits
            competitor_logits
            competitor_ids
    """
    original_2d = feature_logit_attributions.ndim == 2

    if original_2d:
        feature_logit_attributions = feature_logit_attributions.unsqueeze(0).unsqueeze(
            0
        )
    elif feature_logit_attributions.ndim != 4:
        raise ValueError(
            "feature_logit_attributions must have shape [batch, pos, n_features, vocab_size] "
            f"or [n_features, vocab_size], got {tuple(feature_logit_attributions.shape)}"
        )

    batch_size, seq_len, n_features, vocab_size = feature_logit_attributions.shape
    device = feature_logit_attributions.device

    target_token_ids = _normalize_target_token_ids(
        target_token_ids=target_token_ids,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )

    # Gather target logits
    # [batch, pos, n_features, 1]
    gather_index = (
        target_token_ids.unsqueeze(-1)
        .unsqueeze(-1)
        .expand(batch_size, seq_len, n_features, 1)
    )

    # [batch, pos, n_features]
    target_logits = torch.gather(
        feature_logit_attributions,
        dim=-1,
        index=gather_index,
    ).squeeze(-1)

    # Mask target token so we can find max competitor
    masked = feature_logit_attributions.clone()
    masked.scatter_(dim=-1, index=gather_index, value=float("-inf"))

    # [batch, pos, n_features]
    competitor_logits, competitor_ids = masked.max(dim=-1)

    # [batch, pos, n_features]
    logit_gaps = target_logits - competitor_logits

    if original_2d:
        return LogitGapResult(
            logit_gaps=logit_gaps.squeeze(0).squeeze(0),
            target_logits=target_logits.squeeze(0).squeeze(0),
            competitor_logits=competitor_logits.squeeze(0).squeeze(0),
            competitor_ids=competitor_ids.squeeze(0).squeeze(0),
        )

    return LogitGapResult(
        logit_gaps=logit_gaps,
        target_logits=target_logits,
        competitor_logits=competitor_logits,
        competitor_ids=competitor_ids,
    )


if __name__ == "__main__":
    # Tiny sanity-check example
    torch.manual_seed(0)

    batch = 2
    pos = 3
    n_features = 4
    d_model = 6
    vocab_size = 10

    # z_i activations
    z = torch.randn(batch, pos, n_features)

    # decoder directions d_i
    decoder = torch.randn(n_features, d_model)

    # model unembedding W_U
    unembedding = torch.randn(d_model, vocab_size)

    # Step 1: feature -> vocab directions
    feature_vocab_dirs = compute_feature_vocab_directions(decoder, unembedding)
    print(
        "feature_vocab_dirs shape:", feature_vocab_dirs.shape
    )  # [n_features, vocab_size]

    # Step 2: per-feature logit attributions
    feature_attr = compute_feature_logit_attributions(z, decoder, unembedding)
    print(
        "feature_attr shape:", feature_attr.shape
    )  # [batch, pos, n_features, vocab_size]

    # Step 3: target logit gaps
    target_ids = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    result = compute_logit_gaps(feature_attr, target_ids)
    print("logit_gaps shape:", result.logit_gaps.shape)  # [batch, pos, n_features]
    print("target_logits shape:", result.target_logits.shape)
    print("competitor_logits shape:", result.competitor_logits.shape)
    print("competitor_ids shape:", result.competitor_ids.shape)
