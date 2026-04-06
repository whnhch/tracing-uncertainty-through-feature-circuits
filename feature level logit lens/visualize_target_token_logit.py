import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from feature_unembed_logits import load_decoder_matrix, pick_feature_vector


def resolve_target_token_id(
    tokenizer: AutoTokenizer,
    target_token_id: int | None,
    target_token_string: str | None,
    example_prompt: str | None,
) -> tuple[int, str, str]:
    if target_token_id is not None:
        token_str = tokenizer.convert_ids_to_tokens(target_token_id)
        return target_token_id, token_str, "explicit_token_id"

    if target_token_string:
        ids = tokenizer.encode(target_token_string, add_special_tokens=False)
        if not ids:
            raise ValueError(f"target-token-string '{target_token_string}' produced no token ids")
        token_id = int(ids[-1])
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        return token_id, token_str, "explicit_token_string_last_piece"

    if example_prompt:
        ids = tokenizer.encode(example_prompt, add_special_tokens=False)
        if not ids:
            raise ValueError("example_prompt produced no token ids for target inference")
        token_id = int(ids[-1])
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        return token_id, token_str, "inferred_last_token_of_example_prompt"

    raise ValueError(
        "Could not resolve target token. Provide --target-token-id or --target-token-string, "
        "or ensure circuit json contains example_prompt."
    )


def save_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "feature_id",
                "contribution_logit",
                "feature_explanation",
            ],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def plot_scatter_and_mean(records: list[dict], target_token_text: str, out_png: Path) -> None:
    layers = np.array([r["layer"] for r in records], dtype=np.int32)
    contribs = np.array([r["contribution_logit"] for r in records], dtype=np.float32)

    unique_layers = sorted(set(layers.tolist()))
    mean_by_layer = []
    for layer in unique_layers:
        vals = contribs[layers == layer]
        mean_by_layer.append(float(np.mean(vals)) if len(vals) else 0.0)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    colors = np.where(contribs >= 0, "#1b9e77", "#d95f02")
    axes[0].scatter(layers, contribs, c=colors, alpha=0.75, s=22)
    axes[0].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Feature -> target logit")
    axes[0].set_title(f"Per-feature contribution for target token: {target_token_text}")

    axes[1].plot(unique_layers, mean_by_layer, marker="o", color="#2c3e50")
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Mean contribution")
    axes[1].set_title("Layer-wise mean contribution")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize target-token logit contributions for every feature in a per-example circuit."
    )
    parser.add_argument("--circuit-json", required=True, help="Path to circuit example JSON")
    parser.add_argument("--model-id", default="google/gemma-2-2b", help="HF model id or local path")
    parser.add_argument("--decoder-path", required=True, help="Path to decoder matrix file")
    parser.add_argument("--decoder-key", default=None, help="Tensor key for .pt/.safetensors decoder files")
    parser.add_argument("--target-token-id", type=int, default=None, help="Target token id")
    parser.add_argument("--target-token-string", default=None, help="Target token string")
    parser.add_argument("--out-dir", default="viz_target_token", help="Output directory")
    parser.add_argument("--prefix", default="example", help="Output file prefix")
    args = parser.parse_args()

    circuit_path = Path(args.circuit_json)
    if not circuit_path.exists():
        raise FileNotFoundError(f"Circuit json not found: {circuit_path}")

    decoder_path = Path(args.decoder_path)
    if not decoder_path.exists():
        raise FileNotFoundError(f"Decoder file not found: {decoder_path}")

    with open(circuit_path, "r", encoding="utf-8") as f:
        circuit = json.load(f)

    nodes = circuit.get("nodes") or []
    if not nodes:
        raise ValueError(f"No nodes found in {circuit_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model.eval()

    lm_head = model.get_output_embeddings().weight.detach().cpu().float().numpy()
    vocab_size, d_model = lm_head.shape
    decoder = load_decoder_matrix(decoder_path, key=args.decoder_key)

    target_token_id, target_token_text, target_source = resolve_target_token_id(
        tokenizer=tokenizer,
        target_token_id=args.target_token_id,
        target_token_string=args.target_token_string,
        example_prompt=circuit.get("example_prompt"),
    )
    if target_token_id < 0 or target_token_id >= vocab_size:
        raise ValueError(f"target token id {target_token_id} out of vocab range [0, {vocab_size})")

    target_unembed = lm_head[target_token_id, :]

    records: list[dict] = []
    for node in nodes:
        feature_id = node.get("feature_id")
        layer = node.get("layer")
        if not isinstance(feature_id, int) or not isinstance(layer, int):
            continue

        d_f = pick_feature_vector(decoder, feature_id=feature_id, d_model=d_model)
        contribution = float(np.dot(target_unembed, d_f))
        explanations = node.get("feature_explanations") or []
        explanation = explanations[0] if explanations else ""

        records.append(
            {
                "layer": layer,
                "feature_id": feature_id,
                "contribution_logit": contribution,
                "feature_explanation": explanation,
            }
        )

    records.sort(key=lambda r: (r["layer"], -r["contribution_logit"], r["feature_id"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{args.prefix}_token_by_feature.csv"
    png_path = out_dir / f"{args.prefix}_token_by_feature.png"
    json_path = out_dir / f"{args.prefix}_token_by_feature.json"

    save_csv(records, csv_path)
    plot_scatter_and_mean(records, target_token_text=target_token_text, out_png=png_path)

    payload = {
        "circuit_json": str(circuit_path),
        "model_id": args.model_id,
        "decoder_path": str(decoder_path),
        "decoder_key": args.decoder_key,
        "target_token_id": int(target_token_id),
        "target_token_text": target_token_text,
        "target_selection_source": target_source,
        "record_count": len(records),
        "records": records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
