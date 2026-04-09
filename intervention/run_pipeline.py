"""
Main entry point for the uncertainty tracing pipeline.

Usage:
  # Full pipeline (all phases) with default model (pythia-1.4b):
  python run_pipeline.py

  # Full pipeline with Gemma-2 9B:
  python run_pipeline.py --model gemma-2-9b

  # Run specific phases only (e.g. re-run phase 3 after tweaking thresholds):
  python run_pipeline.py --phases 3 4

  # Force re-run even if checkpoint exists:
  python run_pipeline.py --force

Each phase saves a checkpoint before the next begins. If a checkpoint already
exists and --force is not set, that phase is skipped and its checkpoint is
loaded instead. This means a failed run on the school server can resume from
the last completed phase.
"""

import argparse
import torch
import os
import sys

from config import ExperimentConfig, ACTIVE_MODEL, device
from dataset import load_trivia_qa
from utils import load_checkpoint, checkpoint_exists

from phase1_correlational import run_phase1
from phase2_interventions import run_phase2
from phase3_calibration import run_phase3, summarize_phase3
from phase4_propagation import run_phase4, summarize_phase4


def load_model_and_saes(cfg: ExperimentConfig):
    """
    Load the HookedTransformer and all SAEs for the configured model.
    Separated into its own function so the model is loaded once regardless
    of which phases are being run.
    """
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    dev = device()
    print(f"[setup] device: {dev}")
    print(f"[setup] loading model: {cfg.model.model_name}")

    model = HookedTransformer.from_pretrained(
        cfg.model.model_name,
        device=dev,
    )
    model.eval()

    print(f"[setup] loading SAEs for layers: {cfg.model.layers_to_analyze}")
    saes: dict = {}
    hook_names: dict = {}
    for layer_idx in cfg.model.layers_to_analyze:
        sae_id = cfg.model.get_sae_id(layer_idx)
        sae = SAE.from_pretrained(
            release=cfg.model.sae_release,
            sae_id=sae_id,
            device=dev,
        )
        sae.eval()
        saes[layer_idx] = sae
        # Get the TransformerLens hook name directly from the SAE metadata so we
        # don't have to maintain a separate template for each model family.
        hook_name = sae.cfg.metadata["hook_name"]
        hook_names[layer_idx] = hook_name
        print(f"  layer {layer_idx}: {sae_id}  →  hook: {hook_name} ✓")

    return model, saes, hook_names


def main():
    parser = argparse.ArgumentParser(description="Uncertainty tracing pipeline")
    parser.add_argument(
        "--model",
        default=ACTIVE_MODEL,
        choices=["gpt2-small", "gemma-2-9b"],
        help="Which model config to use",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        choices=[1, 2, 3, 4],
        help="Which phases to run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run phases even if checkpoints already exist",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=None,
        help="Override cfg.n_prompts (useful for quick smoke tests)",
    )
    args = parser.parse_args()

    torch.manual_seed(42)

    cfg = ExperimentConfig(model_key=args.model)
    if args.n_prompts is not None:
        cfg.n_prompts = args.n_prompts

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Uncertainty Tracing Pipeline")
    print(f"  Model:   {cfg.model.model_name}")
    print(f"  Device:  {device()}")
    print(f"  Phases:  {args.phases}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    #  Load model + SAEs (shared across all phases)                       #
    # ------------------------------------------------------------------ #
    model, saes, hook_names = load_model_and_saes(cfg)

    # ------------------------------------------------------------------ #
    #  Load dataset (shared across all phases)                            #
    # ------------------------------------------------------------------ #
    print(f"[setup] loading dataset ({cfg.n_prompts} prompts)...")
    data = load_trivia_qa(
        model=model,
        n_prompts=cfg.n_prompts,
        device=device(),
        seed=cfg.seed,
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
    )
    print(f"[setup] dataset ready: {len(data.questions)} prompts\n")

    # ------------------------------------------------------------------ #
    #  Phase 1                                                            #
    # ------------------------------------------------------------------ #
    phase1_ckpt = f"{cfg.checkpoint_dir}/phase1_correlational.pt"
    phase1_results = None

    if 1 in args.phases:
        if checkpoint_exists(phase1_ckpt) and not args.force:
            print("[phase1] checkpoint found, loading...")
            phase1_results = load_checkpoint(phase1_ckpt)
        else:
            print("[phase1] running correlational analysis...")
            phase1_results = run_phase1(cfg, model, saes, hook_names, data)
        print()
    elif checkpoint_exists(phase1_ckpt):
        phase1_results = load_checkpoint(phase1_ckpt)

    # ------------------------------------------------------------------ #
    #  Phase 2                                                            #
    # ------------------------------------------------------------------ #
    phase2_ckpt = f"{cfg.checkpoint_dir}/phase2_interventions.pt"
    phase2_results = None

    if 2 in args.phases:
        if phase1_results is None:
            print("[error] Phase 2 requires Phase 1 results. Run phase 1 first.")
            sys.exit(1)
        if checkpoint_exists(phase2_ckpt) and not args.force:
            print("[phase2] checkpoint found, loading...")
            phase2_results = load_checkpoint(phase2_ckpt)
        else:
            print("[phase2] running causal interventions...")
            phase2_results = run_phase2(cfg, model, saes, hook_names, data, phase1_results)
        print()
    elif checkpoint_exists(phase2_ckpt):
        phase2_results = load_checkpoint(phase2_ckpt)

    # ------------------------------------------------------------------ #
    #  Phase 3                                                            #
    # ------------------------------------------------------------------ #
    phase3_ckpt = f"{cfg.checkpoint_dir}/phase3_calibration.pt"
    phase3_results = None

    if 3 in args.phases:
        if phase2_results is None:
            print("[error] Phase 3 requires Phase 2 results. Run phase 2 first.")
            sys.exit(1)
        if checkpoint_exists(phase3_ckpt) and not args.force:
            print("[phase3] checkpoint found, loading...")
            phase3_results = load_checkpoint(phase3_ckpt)
        else:
            print("[phase3] applying calibration-appropriate constraint...")
            phase3_results = run_phase3(cfg, model, saes, hook_names, data, phase2_results)
        summarize_phase3(phase3_results)
        print()
    elif checkpoint_exists(phase3_ckpt):
        phase3_results = load_checkpoint(phase3_ckpt)

    # ------------------------------------------------------------------ #
    #  Phase 4                                                            #
    # ------------------------------------------------------------------ #
    phase4_ckpt = f"{cfg.checkpoint_dir}/phase4_propagation.pt"

    if 4 in args.phases:
        if phase2_results is None or phase3_results is None:
            print("[error] Phase 4 requires Phase 2 and Phase 3 results.")
            sys.exit(1)
        if checkpoint_exists(phase4_ckpt) and not args.force:
            print("[phase4] checkpoint found, loading...")
            phase4_results = load_checkpoint(phase4_ckpt)
        else:
            print("[phase4] running propagation analysis...")
            phase4_results = run_phase4(
                cfg, model, saes, hook_names, data, phase2_results, phase3_results
            )
        summarize_phase4(phase4_results)
        print()

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
