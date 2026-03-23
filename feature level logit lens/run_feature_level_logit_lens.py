import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoModelForCausalLM

from logit_lens import (
    compute_feature_logit_lens_next_token,
    compute_layerwise_top_feature_token_effects,
    compute_feature_token_trajectory_by_layer,
    compute_logit_lens_next_token,
    save_feature_lens_heatmap,
    save_layerwise_top_feature_token_heatmap,
    save_feature_token_trajectory_heatmap,
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
        "--feature-transcoder-path",
        "--feature_transcoder_path",
        type=str,
        default="",
        help="Path to transcoder checkpoint (.pt) used for feature-level lens.",
    )
    parser.add_argument(
        "--feature-layer",
        type=int,
        default=-1,
        help="Transformer layer for feature lens (-1 means final layer, else 1..L).",
    )
    parser.add_argument("--feature-top-n", type=int, default=10, help="Number of top activated features to visualize.")
    parser.add_argument(
        "--feature-target-token",
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
    parser.add_argument("--save-path", type=str, default="", help="Optional path to save model weights.")
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
    transcoder: Transcoder,
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
) -> None:
    feature_result = compute_feature_logit_lens_next_token(
        model_name=model_name,
        prompt=prompt,
        device=device,
        transcoder=transcoder,
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

    # Additional view: for each top feature, show how it pushes a target token across layers.
    feature_layer_result = compute_feature_token_trajectory_by_layer(
        model_name=model_name,
        prompt=prompt,
        device=device,
        transcoder=transcoder,
        target_token=feature_target_token,
        feature_top_n=feature_top_n,
    )

    save_feature_token_trajectory_heatmap(
        feature_ids=feature_layer_result["feature_ids"],
        layer_indices=feature_layer_result["layer_indices"],
        token_prob_rows=feature_layer_result["token_prob_rows"],
        output_path=feature_layer_plot_path,
        title=(
            "Feature-by-Layer Token Trajectory | "
            f"{feature_layer_result['model_name']} | token='{feature_layer_result['target_token_label']}'"
        ),
        target_token_label=feature_layer_result["target_token_label"],
    )
    print(f"Saved feature-by-layer token trajectory plot to {feature_layer_plot_path}")

    feature_layer_summary_path.parent.mkdir(parents=True, exist_ok=True)
    layer_lines = [
        f"Model: {feature_layer_result['model_name']}",
        f"Prompt: {feature_layer_result['prompt']}",
        (
            f"Target token: '{feature_layer_result['target_token']}' "
            f"(resolved='{feature_layer_result['target_token_label']}', id={feature_layer_result['target_token_id']})"
        ),
        "",
        "Top features tracked across layers:",
    ]
    for item in feature_layer_result["feature_summaries"]:
        layer_lines.append(
            (
                f"Feature {item['feature_id']:>5} | peak_act={item['peak_activation']:.6f} "
                f"| best_layer={item['best_layer']} | best_prob={item['best_token_prob']:.6f}"
            )
        )
    feature_layer_summary_path.write_text("\n".join(layer_lines) + "\n", encoding="utf-8")
    print(f"Saved feature-by-layer token trajectory summary to {feature_layer_summary_path}")

    # Layer-local view: each layer uses its own top-N features.
    local_result = compute_layerwise_top_feature_token_effects(
        model_name=model_name,
        prompt=prompt,
        device=device,
        transcoder=transcoder,
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

    trained_transcoder: Transcoder | None = None
    if args.run_transcoder_train:
        # Placeholder dataset: replace with real MLP input/output activation pairs.
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

        if args.save_path:
            save_path = Path(args.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(trained_transcoder.state_dict(), save_path)
            print(f"Saved transcoder weights to {save_path}")

    if args.run_feature_lens:
        feature_plot_path = Path(args.feature_plot_path)
        feature_summary_path = Path(args.feature_summary_path)
        feature_layer_plot_path = Path(args.feature_layer_plot_path)
        feature_layer_summary_path = Path(args.feature_layer_summary_path)
        feature_layer_local_plot_path = Path(args.feature_layer_local_plot_path)
        feature_layer_local_summary_path = Path(args.feature_layer_local_summary_path)
        feature_ckpt_path = Path(args.feature_transcoder_path) if args.feature_transcoder_path else None

        feature_transcoder: Transcoder | None = None
        if feature_ckpt_path is not None and feature_ckpt_path.exists():
            candidate_transcoder = _load_transcoder_from_checkpoint(feature_ckpt_path, device)
            if candidate_transcoder.input_dim != model_hidden_size:
                print(
                    f"Ignoring checkpoint at {feature_ckpt_path}: input_dim={candidate_transcoder.input_dim} "
                    f"does not match model hidden size {model_hidden_size}."
                )
            else:
                feature_transcoder = candidate_transcoder
        elif trained_transcoder is not None:
            feature_transcoder = trained_transcoder
            feature_transcoder.eval()
            print("Running feature-level lens with newly trained transcoder from this run.")
        else:
            print(
                "Skipping feature-level lens: provide --feature-transcoder-path or run with --run-transcoder-train."
            )

        if feature_transcoder is not None:
            _run_and_save_feature_lens(
                model_name=args.model_name,
                prompt=sentence,
                device=device,
                transcoder=feature_transcoder,
                feature_top_n=args.feature_top_n,
                feature_layer=args.feature_layer,
                top_k=args.top_k,
                feature_target_token=args.feature_target_token,
                feature_plot_path=feature_plot_path,
                feature_summary_path=feature_summary_path,
                feature_layer_plot_path=feature_layer_plot_path,
                feature_layer_summary_path=feature_layer_summary_path,
                feature_layer_local_plot_path=feature_layer_local_plot_path,
                feature_layer_local_summary_path=feature_layer_local_summary_path,
            )


if __name__ == "__main__":
    main()