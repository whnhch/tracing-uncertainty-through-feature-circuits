from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from logit_gap import (
    compute_feature_logit_attributions,
    compute_logit_gaps,
)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_decoder_directions(path: str, device: torch.device) -> torch.Tensor:
    """
    Expected shape:
        [n_features, d_model]

    Supported formats:
        - .pt / .pth containing either:
            * a tensor directly
            * a dict with one of:
                - 'decoder_directions'
                - 'W_dec'
                - 'decoder'
    """
    obj = torch.load(path, map_location=device)

    if isinstance(obj, torch.Tensor):
        decoder = obj
    elif isinstance(obj, dict):
        for key in ["decoder_directions", "W_dec", "decoder"]:
            if key in obj:
                decoder = obj[key]
                break
        else:
            raise KeyError(
                f"Could not find decoder directions in {path}. "
                "Expected one of keys: decoder_directions, W_dec, decoder"
            )
    else:
        raise TypeError(f"Unsupported decoder file format: {type(obj)}")

    if decoder.ndim != 2:
        raise ValueError(
            f"Decoder directions must have shape [n_features, d_model], got {tuple(decoder.shape)}"
        )

    return decoder.to(device=device, dtype=torch.float32)


def get_unembedding_matrix(model: AutoModelForCausalLM) -> torch.Tensor:
    """
    Returns unembedding matrix with shape [d_model, vocab_size].
    """
    output_emb = model.get_output_embeddings()
    if output_emb is None or not hasattr(output_emb, "weight"):
        raise ValueError("Model does not expose output embedding weights")

    # HF lm_head.weight is usually [vocab_size, d_model]
    unembedding = output_emb.weight.detach().T.contiguous()
    return unembedding


def extract_last_token_hidden_states(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """
    Returns hidden states at the final prompt position for the requested layer.

    Output shape:
        [batch, d_model]
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise ValueError("Model did not return hidden_states")

    if layer < 0:
        layer = len(hidden_states) + layer

    if layer < 0 or layer >= len(hidden_states):
        raise IndexError(
            f"Requested layer {layer}, but model returned {len(hidden_states)} hidden-state tensors"
        )

    layer_hidden = hidden_states[layer]  # [batch, seq_len, d_model]

    # final non-padding token index for each example
    last_positions = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(input_ids.shape[0], device=input_ids.device)

    return layer_hidden[batch_idx, last_positions, :]


def encode_features_from_hidden(
    hidden: torch.Tensor,
    decoder_directions: torch.Tensor,
    normalize_decoder: bool = False,
) -> torch.Tensor:
    """
    Approximate feature activations from hidden states using decoder directions.

    This is a simple projection-based fallback:
        z_i = h · d_i

    Args:
        hidden:
            [batch, d_model]
        decoder_directions:
            [n_features, d_model]
        normalize_decoder:
            Whether to L2-normalize decoder directions before projection.

    Returns:
        feature activations:
            [batch, n_features]
    """
    dec = decoder_directions
    if normalize_decoder:
        dec = dec / (dec.norm(dim=-1, keepdim=True) + 1e-8)

    # [batch, d_model] @ [d_model, n_features] -> [batch, n_features]
    z = hidden @ dec.T
    return z


def get_single_token_target_id(
    tokenizer: AutoTokenizer,
    target_text: str,
) -> int:
    """
    Enforce that target_text maps to exactly one token.
    """
    token_ids = tokenizer.encode(target_text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Target {target_text!r} does not map to exactly one token. "
            f"Got ids={token_ids}. Use a different target or filter the dataset."
        )
    return token_ids[0]


def decode_token(tokenizer: AutoTokenizer, token_id: int) -> str:
    try:
        return repr(tokenizer.decode([token_id]))
    except Exception:
        return f"<tok {token_id}>"


def summarize_top_features(
    logit_gaps: torch.Tensor,
    target_logits: torch.Tensor,
    competitor_logits: torch.Tensor,
    competitor_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Input tensors should all be shape [n_features].
    Returns:
        (top_positive, top_negative)
    """
    if logit_gaps.ndim != 1:
        raise ValueError("Expected 1D per-feature tensors")

    k = min(top_k, logit_gaps.shape[0])

    pos_vals, pos_idx = torch.topk(logit_gaps, k=k, largest=True)
    neg_vals, neg_idx = torch.topk(logit_gaps, k=k, largest=False)

    def build_rows(values: torch.Tensor, indices: torch.Tensor) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for v, i in zip(values.tolist(), indices.tolist()):
            competitor_id = int(competitor_ids[i].item())
            rows.append(
                {
                    "feature_idx": int(i),
                    "logit_gap": float(v),
                    "target_logit": float(target_logits[i].item()),
                    "competitor_logit": float(competitor_logits[i].item()),
                    "competitor_id": competitor_id,
                    "competitor_token": tokenizer.decode([competitor_id]),
                }
            )
        return rows

    return build_rows(pos_vals, pos_idx), build_rows(neg_vals, neg_idx)


def print_feature_table(
    title: str,
    rows: List[Dict[str, Any]],
) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for row in rows:
        feature_idx = row["feature_idx"]
        gap = row["logit_gap"]
        tlog = row["target_logit"]
        clog = row["competitor_logit"]
        comp_id = row["competitor_id"]
        comp_tok = repr(row["competitor_token"])
        print(
            f"feature={feature_idx:>6} | "
            f"gap={gap:>10.4f} | "
            f"target_logit={tlog:>10.4f} | "
            f"best_other={clog:>10.4f} | "
            f"competitor_id={comp_id:>6} | "
            f"competitor_token={comp_tok}"
        )


def run_one_example(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    decoder_directions: torch.Tensor,
    prompt: str,
    target: str,
    layer: int,
    top_k: int,
    normalize_decoder: bool,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Full pipeline for one prompt:
      prompt -> hidden state at last prompt token -> feature activations
      -> per-feature vocab attribution -> target logit gap
    """
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    target_id = get_single_token_target_id(tokenizer, target)

    hidden = extract_last_token_hidden_states(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        layer=layer,
    )  # [1, d_model]

    z = encode_features_from_hidden(
        hidden=hidden,
        decoder_directions=decoder_directions,
        normalize_decoder=normalize_decoder,
    )  # [1, n_features]

    unembedding = get_unembedding_matrix(model).to(device=device, dtype=torch.float32)

    # Expand to [batch, pos, n_features] with pos=1
    feature_activations = z.unsqueeze(1)

    feature_logit_attr = compute_feature_logit_attributions(
        feature_activations=feature_activations,
        decoder_directions=decoder_directions,
        unembedding=unembedding,
    )  # [1, 1, n_features, vocab]

    result = compute_logit_gaps(
        feature_logit_attributions=feature_logit_attr,
        target_token_ids=target_id,
    )

    # squeeze to [n_features]
    logit_gaps = result.logit_gaps[0, 0]
    target_logits = result.target_logits[0, 0]
    competitor_logits = result.competitor_logits[0, 0]
    competitor_ids = result.competitor_ids[0, 0]

    top_positive, top_negative = summarize_top_features(
        logit_gaps=logit_gaps,
        target_logits=target_logits,
        competitor_logits=competitor_logits,
        competitor_ids=competitor_ids,
        tokenizer=tokenizer,
        top_k=top_k,
    )

    return {
        "prompt": prompt,
        "target": target,
        "target_id": target_id,
        "target_token_decoded": tokenizer.decode([target_id]),
        "num_features": int(logit_gaps.shape[0]),
        "top_positive": top_positive,
        "top_negative": top_negative,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the standard feature-level target logit gap pipeline."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face causal LM name or local path",
    )
    parser.add_argument(
        "--decoder-path",
        type=str,
        required=True,
        help="Path to .pt/.pth file containing decoder directions [n_features, d_model]",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="logit_gap_test_dataset.jsonl",
        help="Path to JSONL dataset with fields: id, prompt, target",
    )
    parser.add_argument(
        "--example-id",
        type=str,
        default=None,
        help="Optional dataset example id to run. If omitted, runs first row.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional manual prompt override",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Optional manual target override. Must tokenize to exactly one token.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Hidden-state layer index. -1 means final layer hidden state.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top positive/negative-gap features to print",
    )
    parser.add_argument(
        "--normalize-decoder",
        action="store_true",
        help="L2-normalize decoder directions before projecting hidden states",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path to save the result JSON",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    decoder_directions = load_decoder_directions(args.decoder_path, device=device)

    if args.prompt is not None and args.target is not None:
        prompt = args.prompt
        target = args.target
        example_meta: Dict[str, Any] = {"id": "manual"}
    else:
        dataset = load_jsonl(args.dataset)
        if not dataset:
            raise ValueError(f"Dataset is empty: {args.dataset}")

        if args.example_id is None:
            row = dataset[0]
        else:
            matches = [r for r in dataset if r.get("id") == args.example_id]
            if not matches:
                raise ValueError(
                    f"Could not find example_id={args.example_id!r} in {args.dataset}"
                )
            row = matches[0]

        prompt = row["prompt"]
        target = row["target"]
        example_meta = row

    result = run_one_example(
        model=model,
        tokenizer=tokenizer,
        decoder_directions=decoder_directions,
        prompt=prompt,
        target=target,
        layer=args.layer,
        top_k=args.top_k,
        normalize_decoder=args.normalize_decoder,
        device=device,
    )

    print("\nExample")
    print("-------")
    print(f"id: {example_meta.get('id', 'manual')}")
    print(f"prompt: {prompt!r}")
    print(f"target: {target!r}")
    print(f"target_id: {result['target_id']}")
    print(f"decoded_target_token: {result['target_token_decoded']!r}")
    print(f"num_features: {result['num_features']}")

    print_feature_table("Top positive-gap features", result["top_positive"])
    print_feature_table("Top negative-gap features", result["top_negative"])

    if args.save_json is not None:
        payload = {
            "example": example_meta,
            "result": result,
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved result to {args.save_json}")


if __name__ == "__main__":
    main()
