import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoModelForCausalLM

from logit_lens import (
    compute_feature_logit_lens_next_token,
    compute_feature_target_token_trajectory_by_input_token,
    compute_layerwise_top_feature_activations_by_prompt_token,
    compute_layerwise_top_feature_token_effects,
    compute_logit_lens_next_token,
    save_feature_lens_heatmap,
    save_feature_target_token_by_input_token_heatmap,
    save_layerwise_feature_best_token_prob_heatmap,
    save_layerwise_top_feature_activation_heatmap,
    save_layerwise_top_feature_token_heatmap,
    save_logit_lens_heatmap,
)
from models.transcoder import Transcoder
from train.train_transcoder import train_transcoder
from util import set_environment, set_random_seed

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature-level logit lens training.")
    parser.add_argument("--model-name", type=str, default="gpt2", help="Hugging Face causal LM for logit lens.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k candidate tokens shown in visualization.")
    parser.add_argument(
        "--plot-path",
        type=str,
        default="feature level logit lens/data/logit_lens_next_token.png",
        help="Path to save logit lens heatmap image.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default="feature level logit lens/data/logit_lens_summary.txt",
        help="Path to save per-layer top-token summary.",
    )
    parser.add_argument(
        "--run-transcoder-train",
        "--run_transcoder_train",
        action="store_true",
        help="Run transcoder training step after logit lens analysis.",
    )
    parser.add_argument(
        "--run-feature-lens",
        "--run_feature_lens",
        action="store_true",
        help="Run feature-level logit lens using a transcoder checkpoint.",
    )
    parser.add_argument(
        "--feature-transcoder-dir",
        "--feature_transcoder_dir",
        type=str,
        default="",
        help="Directory with per-layer transcoder checkpoints, e.g. layer_1.pt ... layer_L.pt.",
    )
    parser.add_argument(
        "--feature-transcoder-layer-pattern",
        "--feature_transcoder_layer_pattern",
        type=str,
        default="layer_{layer}.pt",
        help="Filename pattern inside --feature-transcoder-dir. Must include {layer}.",
    )
    parser.add_argument(
        "--feature-layer",
        type=int,
        default=-1,
        help="Transformer layer for feature lens (-1 means final layer, else 1..L).",
    )
    parser.add_argument("--feature-top-n", type=int, default=10, help="Number of top activated features to visualize.")
    parser.add_argument(
        "--target-token",
        "--target_token",
        "--feature-target-token",
        "--feature_target_token",
        type=str,
        default="2",
        help="Target token to track across layers in feature trajectory heatmap.",
    )
    parser.add_argument(
        "--feature-plot-path",
        type=str,
        default="feature level logit lens/data/feature_logit_lens_next_token.png",
        help="Path to save feature-level logit lens heatmap image.",
    )
    parser.add_argument(
        "--feature-summary-path",
        type=str,
        default="feature level logit lens/data/feature_logit_lens_summary.txt",
        help="Path to save feature-level summary text.",
    )
    parser.add_argument(
        "--feature-layer-plot-path",
        type=str,
        default="feature level logit lens/data/feature_token_trajectory_by_layer.png",
        help="Path to save feature-by-layer target-token trajectory heatmap.",
    )
    parser.add_argument(
        "--feature-layer-summary-path",
        type=str,
        default="feature level logit lens/data/feature_token_trajectory_by_layer_summary.txt",
        help="Path to save feature-by-layer target-token trajectory summary.",
    )
    parser.add_argument(
        "--feature-layer-local-plot-path",
        type=str,
        default="feature level logit lens/data/feature_token_layer_local_topn.png",
        help="Path to save per-layer local top-feature target-token heatmap.",
    )
    parser.add_argument(
        "--feature-layer-local-summary-path",
        type=str,
        default="feature level logit lens/data/feature_token_layer_local_topn_summary.txt",
        help="Path to save per-layer local top-feature target-token summary.",
    )
    parser.add_argument(
        "--feature-tokenwise-output-dir",
        type=str,
        default="feature level logit lens/data/per_token_feature_topn",
        help="Directory to save per-token, per-layer top-feature heatmaps and summaries.",
    )
    parser.add_argument(
        "--feature-layer-sample-count",
        type=int,
        default=10,
        help=(
            "When --feature-layer is -1, visualize all layers up to this count; "
            "if there are more layers, use an even sample."
        ),
    )
    parser.add_argument("--input-dim", type=int, default=768, help="Input/output feature dimension.")
    parser.add_argument("--dict-size", type=int, default=4096, help="Dictionary size for transcoder features.")
    parser.add_argument("--num-samples", type=int, default=8192, help="Number of synthetic samples to generate.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--l1-coefficient", type=float, default=1e-3, help="L1 penalty coefficient on features.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on (e.g. cpu, cuda).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--transformer_path", type=str, default="", help="Path to transformer model.")
    parser.add_argument("--text-file", type=str, default="", help="Optional text file containing one sentence.")
    parser.add_argument(
        "--save-dir",
        "--save_dir",
        type=str,
        default="",
        help="Optional directory to save per-layer transcoder weights.",
    )
    parser.add_argument(
        "--save-layer-pattern",
        "--save_layer_pattern",
        type=str,
        default="layer_{layer}.pt",
        help="Filename pattern for per-layer weights in --save-dir. Must include {layer}.",
    )
    return parser.parse_args()


def _load_transcoder_from_checkpoint(checkpoint_path: Path, device: torch.device) -> Transcoder:
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")

    if "W_enc.weight" not in checkpoint:
        raise KeyError("Checkpoint is missing W_enc.weight; cannot infer transcoder dimensions.")

    w_enc = checkpoint["W_enc.weight"]
    dict_size = int(w_enc.shape[0])
    input_dim = int(w_enc.shape[1])

    transcoder = Transcoder(input_dim=input_dim, dict_size=dict_size)
    transcoder.load_state_dict(checkpoint)
    transcoder.to(device)
    transcoder.eval()
    return transcoder


def _normalize_layer_pattern(pattern: str, arg_name: str, *, allow_autofix_missing: bool) -> str:
    """Normalize layer filename placeholders to the canonical '{layer}' form."""
    normalized = pattern
    if "${layer}" in normalized:
        normalized = normalized.replace("${layer}", "{layer}")
    if r"\{layer\}" in normalized:
        normalized = normalized.replace(r"\{layer\}", "{layer}")
    if "{}" in normalized and "{layer}" not in normalized:
        normalized = normalized.replace("{}", "{layer}", 1)
    if "flayer" in normalized and "{layer}" not in normalized:
        normalized = normalized.replace("flayer", "{layer}", 1)
    if "{layer" in normalized and "{layer}" not in normalized:
        normalized = normalized.replace("{layer", "{layer}", 1)
    if "layer}" in normalized and "{layer}" not in normalized:
        normalized = normalized.replace("layer}", "{layer}", 1)

    # Preserve the canonical token and remove any other unmatched braces.
    placeholder = "__LAYER_PLACEHOLDER__"
    normalized = normalized.replace("{layer}", placeholder)
    normalized = normalized.replace("{", "").replace("}", "")
    normalized = normalized.replace(placeholder, "{layer}")

    if "{layer}" not in normalized:
        if not allow_autofix_missing:
            raise ValueError(
                f"{arg_name} must include '{{layer}}', e.g. 'layer_{{layer}}.pt'. "
                f"Got: {pattern!r}"
            )

        path = Path(normalized)
        if path.suffix:
            normalized = f"{path.stem}_{{layer}}{path.suffix}"
        else:
            normalized = f"{normalized}_{{layer}}"
        print(
            f"Warning: {arg_name} did not include '{{layer}}'. "
            f"Using auto-fixed pattern: {normalized}"
        )

    return normalized


def _render_layer_pattern(pattern: str, layer_idx: int) -> str:
    """Render layer filename by replacing only the canonical layer token."""
    return pattern.replace("{layer}", str(layer_idx))


def _token_label_to_filename(label: str) -> str:
    safe = []
    for ch in label:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    filename = "".join(safe).strip("_")
    return filename or "token"


def _select_layers_for_trajectory(
    available_layers: list[int],
    feature_layer: int,
    sample_count: int,
) -> list[int]:
    if not available_layers:
        raise ValueError("No available layers found for trajectory visualization.")

    if feature_layer != -1:
        if feature_layer not in available_layers:
            raise ValueError(
                f"Requested --feature-layer={feature_layer}, but available layers are "
                f"{available_layers[0]}..{available_layers[-1]}"
            )
        return [feature_layer]

    max_layers = max(1, int(sample_count))
    if len(available_layers) <= max_layers:
        return list(available_layers)

    sampled_indices: list[int] = []
    for i in range(max_layers):
        pos = round(i * (len(available_layers) - 1) / (max_layers - 1))
        if pos not in sampled_indices:
            sampled_indices.append(pos)

    return [available_layers[i] for i in sampled_indices]


def _load_transcoders_by_layer(
    checkpoint_dir: Path,
    layer_indices: list[int],
    layer_pattern: str,
    *,
    device: torch.device,
    expected_input_dim: int,
) -> dict[int, Transcoder]:
    layer_pattern = _normalize_layer_pattern(
        layer_pattern,
        "--feature-transcoder-layer-pattern",
        allow_autofix_missing=True,
    )

    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Per-layer transcoder directory not found: {checkpoint_dir}")

    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
    if len(checkpoint_files) == 0:
        raise FileNotFoundError(
            f"Per-layer transcoder directory is empty: {checkpoint_dir}. "
            "Run with --run-transcoder-train to create checkpoints."
        )

    transcoders_by_layer: dict[int, Transcoder] = {}
    for layer_idx in layer_indices:
        checkpoint_path = checkpoint_dir / _render_layer_pattern(layer_pattern, layer_idx)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing per-layer checkpoint for layer {layer_idx}: {checkpoint_path}")

        loaded = _load_transcoder_from_checkpoint(checkpoint_path, device)
        if loaded.input_dim != expected_input_dim:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has input_dim={loaded.input_dim}, "
                f"expected {expected_input_dim}."
            )
        transcoders_by_layer[layer_idx] = loaded

    return transcoders_by_layer


def _get_model_hidden_size(model_name: str) -> int:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Different architectures expose this with different names.
    for attr in ("hidden_size", "n_embd", "d_model", "dim", "model_dim", "word_embed_proj_dim"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return int(value)

    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        for attr in ("hidden_size", "n_embd", "d_model", "dim", "model_dim"):
            value = getattr(text_config, attr, None)
            if isinstance(value, int) and value > 0:
                return int(value)

    config_dict = config.to_dict()
    for key in (
        "hidden_size",
        "n_embd",
        "d_model",
        "dim",
        "model_dim",
        "word_embed_proj_dim",
    ):
        value = config_dict.get(key)
        if isinstance(value, int) and value > 0:
            return int(value)

    text_dict = config_dict.get("text_config")
    if isinstance(text_dict, dict):
        for key in ("hidden_size", "n_embd", "d_model", "dim", "model_dim"):
            value = text_dict.get(key)
            if isinstance(value, int) and value > 0:
                return int(value)

    # Last resort: inspect embedding width from the model directly.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    embedding = model.get_input_embeddings()
    if embedding is None:
        raise ValueError(f"Could not infer hidden size for '{model_name}': no input embeddings found.")

    if hasattr(embedding, "embedding_dim") and isinstance(embedding.embedding_dim, int):
        return int(embedding.embedding_dim)

    if hasattr(embedding, "weight") and embedding.weight.ndim == 2:
        return int(embedding.weight.shape[1])

    raise ValueError(f"Could not infer hidden size for '{model_name}' from config or embeddings.")


def _run_and_save_feature_lens(
    *,
    model_name: str,
    prompt: str,
    device: torch.device,
    transcoders_by_layer: dict[int, Transcoder],
    feature_top_n: int,
    feature_layer: int,
    top_k: int,
    feature_target_token: str,
    feature_plot_path: Path,
    feature_summary_path: Path,
    feature_layer_plot_path: Path,
    feature_layer_summary_path: Path,
    feature_layer_local_plot_path: Path,
    feature_layer_local_summary_path: Path,
    feature_tokenwise_output_dir: Path,
    feature_layer_sample_count: int,
) -> None:
    feature_result = compute_feature_logit_lens_next_token(
        model_name=model_name,
        prompt=prompt,
        device=device,
        transcoder=None,
        transcoders_by_layer=transcoders_by_layer,
        feature_top_n=feature_top_n,
        feature_layer=feature_layer,
        top_k=top_k,
    )

    feature_plot_path.parent.mkdir(parents=True, exist_ok=True)
    save_feature_lens_heatmap(
        feature_summaries=feature_result["feature_summaries"],
        candidate_labels=feature_result["candidate_labels"],
        feature_rows=feature_result["feature_rows"],
        output_path=feature_plot_path,
        title=(
            "Feature Logit Lens | "
            f"{feature_result['model_name']} | layer {feature_result['selected_layer']}"
        ),
    )
    print(f"Saved feature-level logit lens plot to {feature_plot_path}")

    feature_summary_path.parent.mkdir(parents=True, exist_ok=True)
    feature_lines = [
        f"Model: {feature_result['model_name']}",
        f"Prompt: {feature_result['prompt']}",
        f"Selected layer: {feature_result['selected_layer']}",
        "",
        "Top features and their top predicted token:",
    ]
    for item in feature_result["feature_summaries"]:
        feature_lines.append(
            (
                f"Feature {item['feature_id']:>5} | act={item['feature_value']:.6f} "
                f"| token='{item['top_token']}' id={item['top_token_id']} "
                f"| prob={item['top_prob']:.6f}"
            )
        )
    feature_summary_path.write_text("\n".join(feature_lines) + "\n", encoding="utf-8")
    print(f"Saved feature-level summary to {feature_summary_path}")

    available_layers = sorted(int(k) for k in transcoders_by_layer.keys())
    trajectory_layers = _select_layers_for_trajectory(
        available_layers=available_layers,
        feature_layer=feature_layer,
        sample_count=feature_layer_sample_count,
    )

    feature_layer_plot_path.parent.mkdir(parents=True, exist_ok=True)
    feature_layer_summary_path.parent.mkdir(parents=True, exist_ok=True)
    multi_layer = len(trajectory_layers) > 1
    plot_suffix = feature_layer_plot_path.suffix if feature_layer_plot_path.suffix else ".png"

    token_traj_lines = [
        f"Model: {model_name}",
        f"Prompt: {prompt}",
        f"Requested --feature-layer={feature_layer}",
        f"Visualized layers: {trajectory_layers}",
        (
            f"Selection mode: {'single layer' if not multi_layer else 'all/sampled layers'} "
            f"(sample_count={feature_layer_sample_count})"
        ),
        "",
        "Trajectory outputs:",
    ]

    for current_layer in trajectory_layers:
        token_traj_result = compute_feature_target_token_trajectory_by_input_token(
            model_name=model_name,
            prompt=prompt,
            device=device,
            transcoders_by_layer=transcoders_by_layer,
            target_token=feature_target_token,
            feature_top_n=feature_top_n,
            feature_layer=current_layer,
        )

        if multi_layer:
            current_plot_path = feature_layer_plot_path.with_name(
                f"{feature_layer_plot_path.stem}_layer_{current_layer:03d}{plot_suffix}"
            )
        else:
            current_plot_path = feature_layer_plot_path

        save_feature_target_token_by_input_token_heatmap(
            feature_ids=token_traj_result["feature_ids"],
            token_positions=token_traj_result["token_positions"],
            token_labels=token_traj_result["token_labels"],
            token_prob_rows=token_traj_result["token_prob_rows"],
            output_path=current_plot_path,
            title=(
                "Target Token Probability Across Input Tokens | "
                f"{token_traj_result['model_name']} | layer {token_traj_result['selected_layer']}"
            ),
            target_token_label=token_traj_result["target_token_label"],
        )
        print(f"Saved feature target-token-by-input-token plot to {current_plot_path}")

        token_traj_lines.append(
            (
                f"Layer {token_traj_result['selected_layer']}: "
                f"plot={current_plot_path.name} "
                f"target='{token_traj_result['target_token_label']}'"
            )
        )
        for item in token_traj_result["feature_summaries"]:
            token_traj_lines.append(
                (
                    f"  feature=f{item['feature_id']:>5} peak_act={item['peak_activation']:.6f} "
                    f"best_pos={item['best_token_position']} "
                    f"best_input_token='{item['best_token_label']}' "
                    f"target_prob={item['best_target_prob']:.6f}"
                )
            )
        token_traj_lines.append("")

    feature_layer_summary_path.write_text("\n".join(token_traj_lines) + "\n", encoding="utf-8")
    print(f"Saved feature target-token-by-input-token summary to {feature_layer_summary_path}")

    # Layer-local view: each layer uses its own top-N features.
    local_result = compute_layerwise_top_feature_token_effects(
        model_name=model_name,
        prompt=prompt,
        device=device,
        transcoder=None,
        transcoders_by_layer=transcoders_by_layer,
        target_token=feature_target_token,
        feature_top_n=feature_top_n,
    )

    save_layerwise_top_feature_token_heatmap(
        rank_labels=local_result["rank_labels"],
        layer_indices=local_result["layer_indices"],
        token_prob_rows=local_result["token_prob_rows"],
        output_path=feature_layer_local_plot_path,
        title=(
            "Layer-Local Top Features | "
            f"{local_result['model_name']} | token='{local_result['target_token_label']}'"
        ),
        target_token_label=local_result["target_token_label"],
    )
    print(f"Saved layer-local top-feature token plot to {feature_layer_local_plot_path}")

    feature_layer_local_summary_path.parent.mkdir(parents=True, exist_ok=True)
    local_lines = [
        f"Model: {local_result['model_name']}",
        f"Prompt: {local_result['prompt']}",
        (
            f"Target token: '{local_result['target_token']}' "
            f"(resolved='{local_result['target_token_label']}', id={local_result['target_token_id']})"
        ),
        "",
        "Per-layer local top features:",
    ]
    for layer_item in local_result["per_layer_summaries"]:
        local_lines.append(f"Layer {layer_item['layer']}")
        for feat_item in layer_item["top_features"]:
            local_lines.append(
                (
                    f"  rank={feat_item['rank']:>2} feature={feat_item['feature_id']:>5} "
                    f"act={feat_item['activation']:.6f} prob={feat_item['target_prob']:.6f}"
                )
            )
    feature_layer_local_summary_path.write_text("\n".join(local_lines) + "\n", encoding="utf-8")
    print(f"Saved layer-local top-feature token summary to {feature_layer_local_summary_path}")

    # Per-token view: for each input token position, save a layer-local top-feature activation heatmap.
    tokenwise_result = compute_layerwise_top_feature_activations_by_prompt_token(
        model_name=model_name,
        prompt=prompt,
        device=device,
        transcoders_by_layer=transcoders_by_layer,
        feature_top_n=feature_top_n,
    )

    feature_tokenwise_output_dir.mkdir(parents=True, exist_ok=True)
    tokenwise_index_lines = [
        f"Model: {tokenwise_result['model_name']}",
        f"Prompt: {tokenwise_result['prompt']}",
        "",
        "Per-token outputs:",
    ]

    for token_item in tokenwise_result["token_results"]:
        token_pos = token_item["token_position"]
        token_label = token_item["token_label"]
        token_id = token_item["token_id"]
        token_file_stub = f"token_{token_pos:03d}_{_token_label_to_filename(token_label)}"

        token_plot_path = feature_tokenwise_output_dir / f"{token_file_stub}.png"
        token_best_prob_plot_path = feature_tokenwise_output_dir / f"{token_file_stub}_best_token_prob.png"
        token_summary_path = feature_tokenwise_output_dir / f"{token_file_stub}.txt"

        save_layerwise_top_feature_activation_heatmap(
            rank_labels=token_item["rank_labels"],
            layer_indices=tokenwise_result["layer_indices"],
            activation_rows=token_item["activation_rows"],
            output_path=token_plot_path,
            title=(
                "Per-Token Layer-Local Top Features | "
                f"token[{token_pos}]='{token_label}'"
            ),
        )
        save_layerwise_feature_best_token_prob_heatmap(
            rank_labels=token_item["rank_labels"],
            layer_indices=tokenwise_result["layer_indices"],
            best_token_prob_rows=token_item["best_token_prob_rows"],
            best_token_label_rows=token_item["best_token_label_rows"],
            output_path=token_best_prob_plot_path,
            title=(
                "Per-Token Feature Best-Token Probability | "
                f"token[{token_pos}]='{token_label}'"
            ),
        )

        token_lines = [
            f"Model: {tokenwise_result['model_name']}",
            f"Prompt: {tokenwise_result['prompt']}",
            f"Token position: {token_pos}",
            f"Token: '{token_label}' (id={token_id})",
            "",
            "Per-layer local top features:",
        ]
        for layer_item in token_item["per_layer_summaries"]:
            token_lines.append(f"Layer {layer_item['layer']}")
            for feat_item in layer_item["top_features"]:
                token_lines.append(
                    (
                        f"  rank={feat_item['rank']:>2} feature=f{feat_item['feature_id']:>5} "
                        f"activation={feat_item['activation']:.6f} "
                        f"best_token='{feat_item['best_token_label']}' "
                        f"best_token_id={feat_item['best_token_id']} "
                        f"best_token_prob={feat_item['best_token_prob']:.6f}"
                    )
                )
        token_summary_path.write_text("\n".join(token_lines) + "\n", encoding="utf-8")

        tokenwise_index_lines.append(
            (
                f"token[{token_pos}]='{token_label}' id={token_id}: "
                f"{token_plot_path.name}, {token_best_prob_plot_path.name}, {token_summary_path.name}"
            )
        )

    tokenwise_index_path = feature_tokenwise_output_dir / "index_summary.txt"
    tokenwise_index_path.write_text("\n".join(tokenwise_index_lines) + "\n", encoding="utf-8")
    print(f"Saved per-token feature outputs to {feature_tokenwise_output_dir}")


def main() -> None:
    args = parse_args()
    # Set environment variables for Hugging Face transformers cache and reproducibility.
    set_environment(transformer_path=args.transformer_path)
    set_random_seed(args.seed)

    sentence = "Uncertainty can be traced through feature circuits in neural language models."

    if args.text_file:
        text_path = Path(args.text_file)
        if not text_path.exists():
            raise FileNotFoundError(f"Text file not found: {text_path}")
        sentence = text_path.read_text(encoding="utf-8").strip()
        if not sentence:
            raise ValueError(f"Text file is empty: {text_path}")
        print(f"Loaded sentence from {text_path}: {sentence}")
    else:
        print(f"Using default sentence: {sentence}")

    # Check device availability and fall back to CPU if CUDA is requested but not available.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda and args.device == "cuda" else torch.device("cpu")
    if args.device == "cuda" and not use_cuda:
        print("CUDA requested but not available; falling back to CPU.")
    else:
        print(f"Using device: {device}")

    model_hidden_size = _get_model_hidden_size(args.model_name)
    if args.input_dim != model_hidden_size:
        print(
            f"Overriding --input-dim={args.input_dim} with model hidden size {model_hidden_size} "
            f"for transcoder compatibility."
        )
    # Compute logit lens results and save visualizations and summaries.
    lens_result = compute_logit_lens_next_token(
        model_name=args.model_name,
        prompt=sentence,
        device=device,
        top_k=args.top_k,
    )
    plot_path = Path(args.plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    save_logit_lens_heatmap(
        layer_indices=lens_result["layer_indices"],
        candidate_labels=lens_result["candidate_labels"],
        probability_rows=lens_result["probability_rows"],
        output_path=plot_path,
        title=f"Logit Lens Next-Token Trajectory | {lens_result['model_name']}",
    )
    print(f"Saved logit lens plot to {plot_path}")
    print(
        "Final-layer next token: "
        f"{lens_result['final_next_token_text']} (id={lens_result['final_next_token_id']})"
    )

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        f"Model: {lens_result['model_name']}",
        f"Prompt: {lens_result['prompt']}",
        (
            "Final-layer next token: "
            f"{lens_result['final_next_token_text']} (id={lens_result['final_next_token_id']})"
        ),
        "",
        "Per-layer argmax next token:",
    ]
    for pred in lens_result["top_predictions"]:
        summary_lines.append(
            f"Layer {pred['layer']:>2}: token='{pred['token']}' id={pred['token_id']} prob={pred['prob']:.6f}"
        )
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Saved layer summary to {summary_path}")

    trained_transcoders_by_layer: dict[int, Transcoder] | None = None
    if args.run_transcoder_train:
        normalized_save_layer_pattern = _normalize_layer_pattern(
            args.save_layer_pattern,
            "--save-layer-pattern",
            allow_autofix_missing=True,
        )

        trained_transcoders_by_layer = {}
        for layer_idx in lens_result["layer_indices"]:
            # Placeholder dataset: replace with real MLP input/output activation pairs for this layer.
            x_in = torch.randn(args.num_samples, model_hidden_size)
            x_out = x_in.clone()
            dataset = TensorDataset(x_in, x_out)
            dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
            )

            trained_transcoder = Transcoder(input_dim=model_hidden_size, dict_size=args.dict_size).to(device)
            train_transcoder(
                transcoder=trained_transcoder,
                dataloader=dataloader,
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                l1_coefficient=args.l1_coefficient,
            )
            trained_transcoder.eval()
            trained_transcoders_by_layer[int(layer_idx)] = trained_transcoder

            if args.save_dir:
                save_dir = Path(args.save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / _render_layer_pattern(normalized_save_layer_pattern, layer_idx)
                torch.save(trained_transcoder.state_dict(), save_path)
                print(f"Saved layer {layer_idx} transcoder weights to {save_path}")

    if args.run_feature_lens:
        feature_plot_path = Path(args.feature_plot_path)
        feature_summary_path = Path(args.feature_summary_path)
        feature_layer_plot_path = Path(args.feature_layer_plot_path)
        feature_layer_summary_path = Path(args.feature_layer_summary_path)
        feature_layer_local_plot_path = Path(args.feature_layer_local_plot_path)
        feature_layer_local_summary_path = Path(args.feature_layer_local_summary_path)
        feature_tokenwise_output_dir = Path(args.feature_tokenwise_output_dir)
        feature_ckpt_dir = Path(args.feature_transcoder_dir) if args.feature_transcoder_dir else None

        feature_transcoders_by_layer: dict[int, Transcoder] | None = None

        if trained_transcoders_by_layer is not None:
            feature_transcoders_by_layer = trained_transcoders_by_layer
            print("Running feature-level lens with newly trained per-layer transcoders from this run.")
        elif feature_ckpt_dir is not None:
            feature_transcoders_by_layer = _load_transcoders_by_layer(
                checkpoint_dir=feature_ckpt_dir,
                layer_indices=lens_result["layer_indices"],
                layer_pattern=args.feature_transcoder_layer_pattern,
                device=device,
                expected_input_dim=model_hidden_size,
            )
            print(
                f"Loaded {len(feature_transcoders_by_layer)} per-layer transcoders from {feature_ckpt_dir}."
            )
        else:
            raise ValueError(
                "Per-layer transcoders are required. Provide --feature-transcoder-dir "
                "or run with --run-transcoder-train and --save-dir."
            )

        _run_and_save_feature_lens(
            model_name=args.model_name,
            prompt=sentence,
            device=device,
            transcoders_by_layer=feature_transcoders_by_layer,
            feature_top_n=args.feature_top_n,
            feature_layer=args.feature_layer,
            top_k=args.top_k,
            feature_target_token=args.target_token,
            feature_plot_path=feature_plot_path,
            feature_summary_path=feature_summary_path,
            feature_layer_plot_path=feature_layer_plot_path,
            feature_layer_summary_path=feature_layer_summary_path,
            feature_layer_local_plot_path=feature_layer_local_plot_path,
            feature_layer_local_summary_path=feature_layer_local_summary_path,
            feature_tokenwise_output_dir=feature_tokenwise_output_dir,
            feature_layer_sample_count=args.feature_layer_sample_count,
        )


if __name__ == "__main__":
    main()