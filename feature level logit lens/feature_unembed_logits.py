import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_decoder_matrix(path: Path, key: str | None) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D decoder matrix in {path}, got shape {arr.shape}.")
        return arr

    if suffix in {".pt", ".bin"}:
        obj = torch.load(path, map_location="cpu")
        return object_to_2d_numpy(obj, key=key, source=str(path))

    if suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            if not keys:
                raise ValueError(f"No tensors found in {path}.")
            chosen_key = key or guess_decoder_key(keys)
            if chosen_key not in keys:
                raise ValueError(
                    f"Key '{chosen_key}' not found in {path}. Available keys: {keys[:20]}"
                )
            tensor = f.get_tensor(chosen_key)
        arr = tensor.detach().cpu().float().numpy()
        if arr.ndim != 2:
            raise ValueError(
                f"Expected 2D decoder tensor for key '{chosen_key}', got shape {arr.shape}."
            )
        return arr

    raise ValueError(
        f"Unsupported decoder file extension '{suffix}'. Use .npy, .pt/.bin, or .safetensors"
    )


def guess_decoder_key(keys: list[str]) -> str:
    preferred = [
        "W_dec",
        "decoder.weight",
        "decoder",
        "sae.W_dec",
        "model.W_dec",
    ]
    lower_map = {k.lower(): k for k in keys}
    for p in preferred:
        if p.lower() in lower_map:
            return lower_map[p.lower()]

    # Fallback: first key that hints decoder.
    for k in keys:
        if "dec" in k.lower() or "decoder" in k.lower():
            return k

    # Last fallback: first tensor.
    return keys[0]


def object_to_2d_numpy(obj: Any, key: str | None, source: str) -> np.ndarray:
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().float().numpy()
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D tensor in {source}, got shape {arr.shape}.")
        return arr

    if isinstance(obj, dict):
        if key is None:
            keys = list(obj.keys())
            guessed = guess_decoder_key([str(k) for k in keys])
            if guessed in obj:
                key = guessed
            elif keys:
                key = keys[0]
        if key not in obj:
            raise ValueError(f"Key '{key}' not found in {source}. Available keys: {list(obj.keys())[:20]}")
        tensor = obj[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Object at key '{key}' is not a tensor in {source}.")
        arr = tensor.detach().cpu().float().numpy()
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D tensor at key '{key}', got shape {arr.shape}.")
        return arr

    raise ValueError(f"Unsupported object type in {source}: {type(obj)}")


def pick_feature_vector(decoder: np.ndarray, feature_id: int, d_model: int) -> np.ndarray:
    # Handle both [d_model, n_features] and [n_features, d_model].
    if decoder.shape[0] == d_model and feature_id < decoder.shape[1]:
        return decoder[:, feature_id]

    if decoder.shape[1] == d_model and feature_id < decoder.shape[0]:
        return decoder[feature_id, :]

    raise ValueError(
        "Could not infer decoder layout. "
        f"decoder shape={decoder.shape}, d_model={d_model}, feature_id={feature_id}."
    )


def top_tokens(logits: np.ndarray, tokenizer: AutoTokenizer, k: int) -> dict[str, list[dict]]:
    k = max(1, k)
    if k > logits.shape[0]:
        k = logits.shape[0]

    top_pos_idx = np.argpartition(logits, -k)[-k:]
    top_neg_idx = np.argpartition(logits, k)[:k]

    top_pos_sorted = top_pos_idx[np.argsort(logits[top_pos_idx])[::-1]]
    top_neg_sorted = top_neg_idx[np.argsort(logits[top_neg_idx])]

    def to_rows(indices: np.ndarray) -> list[dict]:
        rows = []
        for idx in indices.tolist():
            rows.append(
                {
                    "token_id": int(idx),
                    "token": tokenizer.convert_ids_to_tokens(int(idx)),
                    "logit": float(logits[idx]),
                }
            )
        return rows

    return {
        "top_positive": to_rows(top_pos_sorted),
        "top_negative": to_rows(top_neg_sorted),
    }


def parse_token_ids_from_strings(tokenizer: AutoTokenizer, token_strings: list[str]) -> list[int]:
    token_ids: list[int] = []
    for token_str in token_strings:
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        for token_id in ids:
            token_ids.append(int(token_id))
    return token_ids


def build_token_by_feature_view(
    token_ids: list[int],
    feature_logits: list[tuple[int, np.ndarray]],
    tokenizer: AutoTokenizer,
    top_k_features: int,
) -> list[dict]:
    rows: list[dict] = []

    for token_id in sorted(set(token_ids)):
        per_feature = []
        for feature_id, logits in feature_logits:
            per_feature.append(
                {
                    "feature_id": int(feature_id),
                    "logit": float(logits[token_id]),
                }
            )

        strongest_positive = sorted(per_feature, key=lambda x: x["logit"], reverse=True)[:top_k_features]
        strongest_negative = sorted(per_feature, key=lambda x: x["logit"])[:top_k_features]

        rows.append(
            {
                "token_id": int(token_id),
                "token": tokenizer.convert_ids_to_tokens(int(token_id)),
                "strongest_positive_features": strongest_positive,
                "strongest_negative_features": strongest_negative,
            }
        )

    return rows


def load_feature_ids_from_circuit(circuit_json_path: Path) -> list[int]:
    with open(circuit_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    nodes = payload.get("nodes") or []
    feature_ids: set[int] = set()
    for node in nodes:
        feature_id = node.get("feature_id")
        if isinstance(feature_id, int):
            feature_ids.add(feature_id)
        elif isinstance(feature_id, str) and feature_id.isdigit():
            feature_ids.add(int(feature_id))

    if not feature_ids:
        raise ValueError(f"No feature_id values found in circuit file: {circuit_json_path}")

    return sorted(feature_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute feature logits via unembedding projection: logits = W_U @ d_f"
    )
    parser.add_argument("--model-id", default="google/gemma-2-2b", help="HF model id or local path")
    parser.add_argument("--decoder-path", required=True, help="Path to decoder matrix file")
    parser.add_argument("--decoder-key", default=None, help="Tensor key for .pt/.safetensors files")
    parser.add_argument("--feature-ids", type=int, nargs="*", default=None, help="Feature ids to project")
    parser.add_argument(
        "--circuit-json",
        default=None,
        help="Optional circuit JSON (for example circuit_example_0.json) to load all node feature IDs",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Number of positive/negative tokens to keep")
    parser.add_argument(
        "--token-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional token ids for token-by-feature view",
    )
    parser.add_argument(
        "--token-strings",
        nargs="*",
        default=None,
        help="Optional raw strings to tokenize and include in token-by-feature view",
    )
    parser.add_argument(
        "--token-feature-top-k",
        type=int,
        default=10,
        help="Top features to show per token for strongest positive/negative contributions",
    )
    parser.add_argument(
        "--disable-auto-token-view",
        action="store_true",
        help="Disable automatic token-by-feature view from union of top positive/negative tokens",
    )
    parser.add_argument(
        "--save-full-logits",
        action="store_true",
        help="Save full vocabulary logits per feature (large output)",
    )
    parser.add_argument("--output", default="feature_unembed_logits.json", help="Output JSON path")
    args = parser.parse_args()

    decoder_path = Path(args.decoder_path)
    if not decoder_path.exists():
        raise FileNotFoundError(f"Decoder file not found: {decoder_path}")

    feature_ids: list[int] = []
    if args.circuit_json:
        circuit_path = Path(args.circuit_json)
        if not circuit_path.exists():
            raise FileNotFoundError(f"Circuit file not found: {circuit_path}")
        feature_ids = load_feature_ids_from_circuit(circuit_path)
        if args.feature_ids:
            # Union manual and circuit-derived ids.
            feature_ids = sorted(set(feature_ids) | set(args.feature_ids))
    elif args.feature_ids:
        feature_ids = sorted(set(args.feature_ids))

    if not feature_ids:
        raise ValueError("Provide --feature-ids and/or --circuit-json to select at least one feature.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model.eval()

    lm_head = model.get_output_embeddings().weight.detach().cpu().float().numpy()
    vocab_size, d_model = lm_head.shape

    decoder = load_decoder_matrix(decoder_path, key=args.decoder_key)

    rows = []
    feature_logits: list[tuple[int, np.ndarray]] = []
    auto_token_ids: set[int] = set()
    for feature_id in feature_ids:
        d_f = pick_feature_vector(decoder, feature_id=feature_id, d_model=d_model)
        logits = lm_head @ d_f
        feature_logits.append((int(feature_id), logits))

        top = top_tokens(logits=logits, tokenizer=tokenizer, k=args.top_k)
        if not args.disable_auto_token_view:
            for item in top["top_positive"]:
                auto_token_ids.add(int(item["token_id"]))
            for item in top["top_negative"]:
                auto_token_ids.add(int(item["token_id"]))

        row = {
            "feature_id": int(feature_id),
            "decoder_vector_norm": float(np.linalg.norm(d_f)),
            "vocab_size": int(vocab_size),
            "top_tokens": top,
        }
        if args.save_full_logits:
            row["full_logits"] = logits.tolist()
        rows.append(row)

    requested_token_ids: set[int] = set(args.token_ids or [])
    if args.token_strings:
        requested_token_ids.update(parse_token_ids_from_strings(tokenizer, args.token_strings))

    all_token_ids = requested_token_ids | auto_token_ids
    token_by_feature = build_token_by_feature_view(
        token_ids=list(all_token_ids),
        feature_logits=feature_logits,
        tokenizer=tokenizer,
        top_k_features=max(1, args.token_feature_top_k),
    )

    out = {
        "model_id": args.model_id,
        "decoder_path": str(decoder_path),
        "decoder_key": args.decoder_key,
        "decoder_shape": [int(decoder.shape[0]), int(decoder.shape[1])],
        "feature_count": int(len(feature_ids)),
        "formula": "logits = W_U @ d_f",
        "features": rows,
        "token_by_feature": token_by_feature,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
