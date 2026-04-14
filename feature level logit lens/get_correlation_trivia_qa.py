from __future__ import annotations
import os
from pathlib import Path

hf_token_path = Path(__file__).with_name("hf_token.txt")
if hf_token_path.exists():
    hf_token = hf_token_path.read_text(encoding="utf-8").strip()
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        print("Loaded Hugging Face token from hf_token.txt")
        
import argparse
import csv
import json
import re
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import numpy as np
import torch
from scipy.stats import spearmanr  # type: ignore[import-not-found]
from transformers import AutoTokenizer



def _normalize_path_arg(path_str: str) -> Path:
    p = Path(path_str)
    cleaned_parts: list[str] = []
    for part in p.parts:
        if p.anchor and part == p.anchor:
            cleaned_parts.append(part)
            continue
        stripped = part.strip()
        if stripped:
            cleaned_parts.append(stripped)

    if not cleaned_parts:
        return Path(".")
    return Path(*cleaned_parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local circuit-tracer attribution on TriviaQA prompts, then compute "
            "Phase-1 style feature correlations from graph files using per-feature "
            "target-token logit effects."
        )
    )
    parser.add_argument("--model-name", default="google/gemma-2-2b")
    parser.add_argument("--transcoder-name", default="gemma")
    parser.add_argument("--backend", default="transformerlens", choices=["transformerlens", "nnsight"])
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])

    parser.add_argument("--triviaqa-config", default="rc.nocontext")
    parser.add_argument("--triviaqa-split", default="validation")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--n-prompts", type=int, default=32)
    parser.add_argument("--append-answer-cue", action="store_true")

    parser.add_argument("--slug-prefix", default="triviaqa-graph")
    parser.add_argument("--graph-dir", default="results/graphs")
    parser.add_argument("--graph-file-dir", default="results/graph_files")
    parser.add_argument("--skip-attribution", action="store_true")

    parser.add_argument("--max-n-logits", type=int, default=10)
    parser.add_argument("--desired-logit-prob", type=float, default=0.95)
    parser.add_argument("--max-feature-nodes", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--offload", default="cpu", choices=["cpu", "disk", "none"])
    parser.add_argument("--node-threshold", type=float, default=0.8)
    parser.add_argument("--edge-threshold", type=float, default=0.98)

    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--output-json",
        default="results/data/phase1_triviaqa_graph_corr.json",
    )
    parser.add_argument(
        "--output-csv",
        default="results/data/phase1_triviaqa_graph_corr.csv",
    )
    parser.add_argument(
        "--summary-path",
        default="results/data/phase1_triviaqa_graph_corr_summary.txt",
    )
    parser.add_argument(
        "--feature-token-top-k",
        type=int,
        default=50,
        help="Top-k features to report for per-feature token-level logit effects.",
    )
    parser.add_argument(
        "--feature-lens-top-k-features",
        type=int,
        default=30,
        help="How many features to include in the feature-level lens section.",
    )
    parser.add_argument(
        "--feature-lens-top-k-tokens",
        type=int,
        default=12,
        help="How many token logits to keep per feature (sorted by |mean effect|).",
    )
    parser.add_argument(
        "--feature-token-heatmap-path",
        default="results/data/feature_token_effects_heatmap.png",
        help="Path to save heatmap of feature token effects (prev/current/predicted).",
    )
    return parser.parse_args()


def to_torch_dtype(dtype_str: str) -> Any:
    import torch  # type: ignore[import-not-found]

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_str]


def load_triviaqa_prompts(args: argparse.Namespace) -> list[dict[str, Any]]:
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset("trivia_qa", args.triviaqa_config, split=args.triviaqa_split)
    if len(ds) == 0:
        raise ValueError("TriviaQA split is empty.")

    rows: list[dict[str, Any]] = []
    for i in range(args.n_prompts):
        idx = (args.start_index + i) % len(ds)
        row = ds[idx]
        question = row.get("question", "")
        if not question:
            continue
        prompt = f"{question} Answer:" if args.append_answer_cue else question
        rows.append({"dataset_index": idx, "prompt": prompt})

    if not rows:
        raise ValueError("No valid TriviaQA prompts were loaded.")
    return rows


def generate_graphs(args: argparse.Namespace, prompts: list[dict[str, Any]]) -> list[Path]:
    from circuit_tracer import ReplacementModel, attribute  # type: ignore[import-not-found]
    from circuit_tracer.utils import create_graph_files  # type: ignore[import-not-found]

    offload = None if args.offload == "none" else args.offload
    model = ReplacementModel.from_pretrained(
        args.model_name,
        args.transcoder_name,
        dtype=to_torch_dtype(args.dtype),
        backend=args.backend,
    )

    graph_dir = _normalize_path_arg(args.graph_dir)
    graph_file_dir = _normalize_path_arg(args.graph_file_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_file_dir.mkdir(parents=True, exist_ok=True)

    out_json_paths: list[Path] = []
    for i, row in enumerate(prompts):
        slug = f"{args.slug_prefix}-{row['dataset_index']}"
        print(f"[{i+1}/{len(prompts)}] attributing prompt index={row['dataset_index']} slug={slug}")

        graph = attribute(
            prompt=row["prompt"],
            model=model,
            max_n_logits=args.max_n_logits,
            desired_logit_prob=args.desired_logit_prob,
            batch_size=args.batch_size,
            max_feature_nodes=args.max_feature_nodes,
            offload=offload,
            verbose=True,
        )

        graph_path = graph_dir / f"{slug}.pt"
        graph.to_pt(graph_path)

        create_graph_files(
            graph_or_path=graph_path,
            slug=slug,
            output_path=str(graph_file_dir),
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
        )
        out_json_paths.append(graph_file_dir / f"{slug}.json")

    return out_json_paths


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _feature_key(node: dict[str, Any]) -> tuple[int, int] | None:
    if node.get("is_target_logit", False):
        return None
    feature_type = str(node.get("feature_type", "")).lower()
    layer = str(node.get("layer", ""))
    feature = node.get("feature")
    if not ("transcoder" in feature_type or layer.isdigit()):
        return None
    if not isinstance(feature, int) or feature < 0:
        return None
    if not layer.isdigit():
        return None
    return int(layer), int(feature)


def _parse_token_from_clerp(clerp: str) -> str:
    match = re.search(r'Output\s+"(.*?)"', clerp)
    if not match:
        return "<unknown>"
    return match.group(1)


def parse_graph_target_effects(graph_json_path: Path) -> dict[str, Any]:
    data = json.loads(graph_json_path.read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])
    links = data.get("links", [])

    node_by_id = {str(n.get("node_id", "")): n for n in nodes}

    target_nodes = [n for n in nodes if n.get("is_target_logit", False)]
    if not target_nodes:
        raise ValueError(f"No target logit node found in {graph_json_path}")

    target = max(target_nodes, key=lambda n: _to_float(n.get("token_prob", 0.0)))
    target_id = str(target["node_id"])
    target_ctx_idx = int(target.get("ctx_idx", -1))
    target_prob = _to_float(target.get("token_prob", 0.0))
    target_token = _parse_token_from_clerp(str(target.get("clerp", "")))
    target_token_id = int(target.get("feature", -1))

    per_feature_effect: dict[tuple[int, int], float] = defaultdict(float)
    logit_nodes = [n for n in nodes if str(n.get("feature_type", "")).lower() == "logit"]
    logit_node_id_to_tid: dict[str, int] = {}
    logit_tid_to_text: dict[int, str] = {}
    for n in logit_nodes:
        nid = str(n.get("node_id", ""))
        tid = int(n.get("feature", -1))
        if tid < 0:
            continue
        logit_node_id_to_tid[nid] = tid
        logit_tid_to_text[tid] = _parse_token_from_clerp(str(n.get("clerp", "")))

    feature_token_effects: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for link in links:
        source_id = str(link.get("source", ""))
        source_node = node_by_id.get(source_id)
        if source_node is None:
            continue

        source_ctx_idx = int(source_node.get("ctx_idx", -10**9))
        if source_ctx_idx != target_ctx_idx:
            continue

        key = _feature_key(source_node)
        if key is None:
            continue
        target_id_this = str(link.get("target", ""))
        weight = _to_float(link.get("weight", 0.0))

        if target_id_this == target_id:
            per_feature_effect[key] += weight

        tid = logit_node_id_to_tid.get(target_id_this)
        if tid is not None:
            fkey = f"{key[0]}:{key[1]}"
            feature_token_effects[fkey][str(tid)] += weight

    return {
        "graph_file": str(graph_json_path),
        "slug": data.get("metadata", {}).get("slug"),
        "prompt": data.get("metadata", {}).get("prompt"),
        "target_token": target_token,
        "target_token_id": target_token_id,
        "target_token_prob": target_prob,
        "target_node_id": target_id,
        "target_ctx_idx": target_ctx_idx,
        "target_logit_effect_total": float(sum(per_feature_effect.values())),
        "feature_effects": {f"{k[0]}:{k[1]}": float(v) for k, v in per_feature_effect.items()},
        "logit_tokens": {str(tid): tok for tid, tok in logit_tid_to_text.items()},
        "feature_token_effects": {
            fkey: {tk: float(tv) for tk, tv in token_map.items()}
            for fkey, token_map in feature_token_effects.items()
        },
    }


def compute_feature_token_lens(
    samples: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    model_name: str,
    top_k: int,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt_by_idx = {int(p["dataset_index"]): p["prompt"] for p in prompts}
    tracked_by_sample: dict[int, dict[str, int]] = {}

    for s in samples:
        idx = int(s["slug"].rsplit("-", 1)[-1]) if str(s.get("slug", "")).rsplit("-", 1)[-1].isdigit() else None
        if idx is None or idx not in prompt_by_idx:
            continue

        ids = tokenizer(prompt_by_idx[idx], return_tensors="pt")["input_ids"][0].tolist()
        current_id = int(ids[-1]) if ids else -1
        previous_id = int(ids[-2]) if len(ids) >= 2 else current_id
        predicted_id = int(s.get("target_token_id", -1))

        tracked_by_sample[idx] = {
            "previous": previous_id,
            "current": current_id,
            "predicted": predicted_id,
        }

    feature_keys = sorted({k for s in samples for k in s.get("feature_token_effects", {}).keys()})
    rows: list[dict[str, Any]] = []

    for fk in feature_keys:
        prev_vals: list[float] = []
        curr_vals: list[float] = []
        pred_vals: list[float] = []

        layer_str, feat_str = fk.split(":", 1)
        layer = int(layer_str)
        feat = int(feat_str)

        for s in samples:
            slug = str(s.get("slug", ""))
            idx = int(slug.rsplit("-", 1)[-1]) if slug.rsplit("-", 1)[-1].isdigit() else None
            token_map = s.get("feature_token_effects", {}).get(fk, {})

            if idx is None or idx not in tracked_by_sample:
                prev_vals.append(0.0)
                curr_vals.append(0.0)
                pred_vals.append(0.0)
                continue

            tr = tracked_by_sample[idx]
            prev_vals.append(_to_float(token_map.get(str(tr["previous"]), 0.0)))
            curr_vals.append(_to_float(token_map.get(str(tr["current"]), 0.0)))
            pred_vals.append(_to_float(token_map.get(str(tr["predicted"]), 0.0)))

        rows.append(
            {
                "layer": layer,
                "feature_id": feat,
                "mean_prev_token_logit_effect": float(np.mean(prev_vals)) if prev_vals else 0.0,
                "mean_current_token_logit_effect": float(np.mean(curr_vals)) if curr_vals else 0.0,
                "mean_predicted_token_logit_effect": float(np.mean(pred_vals)) if pred_vals else 0.0,
                "std_predicted_token_logit_effect": float(np.std(pred_vals)) if pred_vals else 0.0,
                "abs_mean_predicted": float(abs(np.mean(pred_vals))) if pred_vals else 0.0,
            }
        )

    ranked = sorted(rows, key=lambda r: r["abs_mean_predicted"], reverse=True)
    return {
        "n_features": len(rows),
        "top_features": ranked[: min(top_k, len(ranked))],
        "all_features": ranked,
    }


def compute_feature_level_lens(
    samples: list[dict[str, Any]],
    top_k_features: int,
    top_k_tokens: int,
) -> dict[str, Any]:
    # Aggregate per-feature token effects across prompts.
    token_effects_by_feature: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    target_effects_by_feature: dict[str, list[float]] = defaultdict(list)
    token_text_by_id: dict[str, str] = {}

    for sample in samples:
        token_map = sample.get("logit_tokens", {})
        for tid, tok in token_map.items():
            token_text_by_id[str(tid)] = str(tok)

        feature_token_effects = sample.get("feature_token_effects", {})
        feature_target_effects = sample.get("feature_effects", {})

        for fk, val in feature_target_effects.items():
            target_effects_by_feature[fk].append(_to_float(val))

        for fk, tok_effects in feature_token_effects.items():
            for tid, val in tok_effects.items():
                token_effects_by_feature[fk][str(tid)].append(_to_float(val))

    rows: list[dict[str, Any]] = []
    for fk, per_token_lists in token_effects_by_feature.items():
        layer_str, feat_str = fk.split(":", 1)
        layer = int(layer_str)
        feature_id = int(feat_str)

        token_stats: list[dict[str, Any]] = []
        for tid, vals in per_token_lists.items():
            mean_val = float(np.mean(vals)) if vals else 0.0
            std_val = float(np.std(vals)) if vals else 0.0
            token_stats.append(
                {
                    "token_id": int(tid),
                    "token": token_text_by_id.get(tid, "<unknown>"),
                    "mean_logit_effect": mean_val,
                    "std_logit_effect": std_val,
                    "abs_mean": float(abs(mean_val)),
                }
            )

        token_stats_sorted = sorted(token_stats, key=lambda r: r["abs_mean"], reverse=True)
        top_token_stats = token_stats_sorted[: min(top_k_tokens, len(token_stats_sorted))]

        target_vals = target_effects_by_feature.get(fk, [])
        target_mean = float(np.mean(target_vals)) if target_vals else 0.0
        target_std = float(np.std(target_vals)) if target_vals else 0.0

        # Simple diagnostics: how concentrated this feature is on its strongest token.
        total_abs = float(sum(abs(x["mean_logit_effect"]) for x in token_stats_sorted))
        max_abs = float(token_stats_sorted[0]["abs_mean"]) if token_stats_sorted else 0.0
        focus_ratio = max_abs / total_abs if total_abs > 0 else 0.0

        rows.append(
            {
                "layer": layer,
                "feature_id": feature_id,
                "target_token_mean_logit_effect": target_mean,
                "target_token_std_logit_effect": target_std,
                "n_tokens_seen": len(token_stats_sorted),
                "focus_ratio": focus_ratio,
                "top_token_logits": top_token_stats,
            }
        )

    rows_sorted = sorted(rows, key=lambda r: abs(r["target_token_mean_logit_effect"]), reverse=True)
    top_rows = rows_sorted[: min(top_k_features, len(rows_sorted))]

    return {
        "n_features": len(rows_sorted),
        "top_features": top_rows,
        "all_features": rows_sorted,
    }


def compute_correlations(samples: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    y = np.array([_to_float(s["target_token_prob"]) for s in samples], dtype=np.float32)
    feature_keys = sorted({k for s in samples for k in s["feature_effects"].keys()})

    rows: list[dict[str, Any]] = []
    for feature_key in feature_keys:
        x = np.array([_to_float(s["feature_effects"].get(feature_key, 0.0)) for s in samples], dtype=np.float32)
        rho, p = spearmanr(x, y)
        if np.isnan(rho):
            rho = 0.0
        if np.isnan(p):
            p = 1.0

        layer_str, feature_str = feature_key.split(":", 1)
        rows.append(
            {
                "layer": int(layer_str),
                "feature_id": int(feature_str),
                "rho_target_prob": float(rho),
                "p_value": float(p),
                "mean_target_logit_effect": float(np.mean(x)),
                "std_target_logit_effect": float(np.std(x)),
                "nonzero_count": int(np.count_nonzero(x)),
                "abs_rho": float(abs(rho)),
            }
        )

    rows_sorted = sorted(rows, key=lambda r: r["abs_rho"], reverse=True)
    top_rows = rows_sorted[: min(top_k, len(rows_sorted))]

    return {
        "n_prompts": len(samples),
        "target_token_prob_mean": float(np.mean(y)) if len(y) else 0.0,
        "target_token_prob_std": float(np.std(y)) if len(y) else 0.0,
        "n_features": len(rows_sorted),
        "top_features": top_rows,
        "all_features": rows_sorted,
    }


def write_outputs(result: dict[str, Any], output_json: Path, output_csv: Path, summary_path: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "feature_id",
                "rho_target_prob",
                "p_value",
                "mean_target_logit_effect",
                "std_target_logit_effect",
                "nonzero_count",
                "abs_rho",
            ],
        )
        writer.writeheader()
        for row in result["correlation"]["all_features"]:
            writer.writerow(row)

    feature_token_csv = output_csv.with_name(output_csv.stem + "_feature_token_lens.csv")
    with feature_token_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "feature_id",
                "mean_prev_token_logit_effect",
                "mean_current_token_logit_effect",
                "mean_predicted_token_logit_effect",
                "std_predicted_token_logit_effect",
                "abs_mean_predicted",
            ],
        )
        writer.writeheader()
        for row in result["feature_token_lens"]["all_features"]:
            writer.writerow(row)

    lines = [
        f"Prompts analyzed: {result['correlation']['n_prompts']}",
        f"Features analyzed: {result['correlation']['n_features']}",
        (
            "Target token prob mean/std: "
            f"{result['correlation']['target_token_prob_mean']:.6f}/"
            f"{result['correlation']['target_token_prob_std']:.6f}"
        ),
        "",
        "Top features by |Spearman rho| with target token probability:",
    ]
    for i, row in enumerate(result["correlation"]["top_features"][:20], start=1):
        lines.append(
            (
                f"{i:>2}. layer={row['layer']:>2} feature={row['feature_id']:>7} "
                f"rho={row['rho_target_prob']:+.4f} p={row['p_value']:.3g} "
                f"mean_logit_effect={row['mean_target_logit_effect']:+.6f} "
                f"nonzero={row['nonzero_count']}"
            )
        )

    lines.extend([
        "",
        "Top features by absolute mean predicted-token logit effect:",
    ])
    for i, row in enumerate(result["feature_token_lens"]["top_features"][:20], start=1):
        lines.append(
            (
                f"{i:>2}. layer={row['layer']:>2} feature={row['feature_id']:>7} "
                f"prev={row['mean_prev_token_logit_effect']:+.6f} "
                f"curr={row['mean_current_token_logit_effect']:+.6f} "
                f"pred={row['mean_predicted_token_logit_effect']:+.6f}"
            )
        )

    lines.extend([
        "",
        "Top feature-level lens entries (target-token effect + strongest token logits):",
    ])
    for i, row in enumerate(result["feature_level_lens"]["top_features"][:10], start=1):
        lines.append(
            (
                f"{i:>2}. layer={row['layer']:>2} feature={row['feature_id']:>7} "
                f"target_mean={row['target_token_mean_logit_effect']:+.6f} "
                f"focus={row['focus_ratio']:.3f}"
            )
        )
        top_tokens = row.get("top_token_logits", [])[:5]
        if top_tokens:
            token_str = ", ".join(
                f"{t['token']}:{t['mean_logit_effect']:+.4f}" for t in top_tokens
            )
            lines.append(f"    tokens -> {token_str}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_feature_token_heatmap(feature_rows: list[dict[str, Any]], output_path: Path, max_rows: int = 30) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = feature_rows[: min(max_rows, len(feature_rows))]
    if not selected:
        return

    matrix = np.array(
        [
            [
                float(r["mean_prev_token_logit_effect"]),
                float(r["mean_current_token_logit_effect"]),
                float(r["mean_predicted_token_logit_effect"]),
            ]
            for r in selected
        ],
        dtype=np.float32,
    )

    labels = [f"L{int(r['layer'])}:F{int(r['feature_id'])}" for r in selected]

    fig_h = max(5, 0.3 * len(selected))
    fig, ax = plt.subplots(figsize=(8, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["previous", "current", "predicted"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Token tracked at last position")
    ax.set_ylabel("Feature (layer:id)")
    ax.set_title("Per-feature token logit effects")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean logit effect")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
            
    args = parse_args()
    graph_file_dir = _normalize_path_arg(args.graph_file_dir)

    prompts = load_triviaqa_prompts(args)
    if args.skip_attribution:
        graph_paths = [graph_file_dir / f"{args.slug_prefix}-{p['dataset_index']}.json" for p in prompts]
    else:
        graph_paths = generate_graphs(args, prompts)

    missing = [p for p in graph_paths if not p.exists()]
    if missing:
        missing_display = "\n".join(str(p) for p in missing[:10])
        raise FileNotFoundError(f"Missing graph JSON files. First missing:\n{missing_display}")

    samples = [parse_graph_target_effects(p) for p in graph_paths]
    corr = compute_correlations(samples, top_k=args.top_k)
    feature_token_lens = compute_feature_token_lens(
        samples=samples,
        prompts=prompts,
        model_name=args.model_name,
        top_k=args.feature_token_top_k,
    )
    feature_level_lens = compute_feature_level_lens(
        samples=samples,
        top_k_features=args.feature_lens_top_k_features,
        top_k_tokens=args.feature_lens_top_k_tokens,
    )

    result = {
        "config": vars(args),
        "samples": samples,
        "correlation": corr,
        "feature_token_lens": feature_token_lens,
        "feature_level_lens": feature_level_lens,
    }

    write_outputs(
        result,
        output_json=Path(args.output_json),
        output_csv=Path(args.output_csv),
        summary_path=Path(args.summary_path),
    )

    save_feature_token_heatmap(
        feature_rows=result["feature_token_lens"]["all_features"],
        output_path=Path(args.feature_token_heatmap_path),
    )

    print(f"Saved JSON: {args.output_json}")
    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved summary: {args.summary_path}")
    print(f"Saved heatmap: {args.feature_token_heatmap_path}")


if __name__ == "__main__":
    main()