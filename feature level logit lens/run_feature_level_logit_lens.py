import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from logit_lens import (
    compute_feature_logit_lens_next_token,
    compute_logit_lens_next_token,
    save_feature_lens_heatmap,
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
        action="store_true",
        help="Run transcoder training step after logit lens analysis.",
    )
    parser.add_argument(
        "--run-feature-lens",
        action="store_true",
        help="Run feature-level logit lens using a transcoder checkpoint.",
    )
    parser.add_argument(
        "--feature-transcoder-path",
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
    parser.add_argument("--feature-top-n", type=int, default=16, help="Number of top activated features to visualize.")
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


def main() -> None:
    args = parse_args()
    set_environment(transformer_path=args.transformer_path, hf_token_path=None)
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

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda or args.device == "cpu" else "cpu")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")

    lens_result = compute_logit_lens_next_token(
        model_name=args.model_name,
        prompt=sentence,
        device=device,
        top_k=args.top_k,
    )
    plot_path = Path(args.plot_path)
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

    if args.run_feature_lens:
        feature_ckpt_path = Path(args.feature_transcoder_path) if args.feature_transcoder_path else None
        if feature_ckpt_path is None or not feature_ckpt_path.exists():
            print(
                "Skipping feature-level lens: provide --feature-transcoder-path to a trained transcoder checkpoint."
            )
        else:
            feature_transcoder = _load_transcoder_from_checkpoint(feature_ckpt_path, device)
            feature_result = compute_feature_logit_lens_next_token(
                model_name=args.model_name,
                prompt=sentence,
                device=device,
                transcoder=feature_transcoder,
                feature_top_n=args.feature_top_n,
                feature_layer=args.feature_layer,
                top_k=args.top_k,
            )

            feature_plot_path = Path(args.feature_plot_path)
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

            feature_summary_path = Path(args.feature_summary_path)
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

    if not args.run_transcoder_train:
        return

    # Placeholder dataset: replace with real MLP input/output activation pairs.
    x_in = torch.randn(args.num_samples, args.input_dim)
    x_out = x_in.clone()
    dataset = TensorDataset(x_in, x_out)
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    transcoder = Transcoder(input_dim=args.input_dim, dict_size=args.dict_size).to(device)
    train_transcoder(
        transcoder=transcoder,
        dataloader=dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        l1_coefficient=args.l1_coefficient,
    )

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(transcoder.state_dict(), save_path)
        print(f"Saved transcoder weights to {save_path}")


if __name__ == "__main__":
    main()