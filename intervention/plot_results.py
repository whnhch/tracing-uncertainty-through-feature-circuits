"""
Visualize results from all four pipeline phases.

Run this locally after copying the checkpoints back from the server:

    python intervention/plot_results.py
    python intervention/plot_results.py --model gemma-2-9b
    python intervention/plot_results.py --checkpoint-dir /path/to/checkpoints

Figures are saved to intervention/figures/<model_key>/.
No GPU required — checkpoints load on CPU.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; works headless on servers too
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CALIBRATION_APPROPRIATE = "calibration_appropriate"
SEMANTIC_DRIFT          = "semantic_drift"
DEGRADATION             = "degradation"

LABEL_COLORS = {
    CALIBRATION_APPROPRIATE: "#2ecc71",
    SEMANTIC_DRIFT:          "#e67e22",
    DEGRADATION:             "#e74c3c",
}

LABEL_SHORT = {
    CALIBRATION_APPROPRIATE: "CA",
    SEMANTIC_DRIFT:          "Drift",
    DEGRADATION:             "Degrad.",
}


def load_checkpoint(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[warn] checkpoint not found: {path}")
        return {}
    return torch.load(path, map_location="cpu")


def save_fig(fig: plt.Figure, out_dir: Path, filename: str, dpi: int = 200) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# ---------------------------------------------------------------------------
# Figure 1 — Phase 1: Spearman ρ by layer
# ---------------------------------------------------------------------------

def plot_phase1_correlations(p1: dict, out_dir: Path, model_key: str) -> None:
    if not p1:
        return
    per_layer = p1.get("per_layer", {})
    if not per_layer:
        return

    layers = sorted(per_layer.keys())
    top_rho   = [float(per_layer[l]["top_correlations"][0]) for l in layers]
    mean_rho  = [float(per_layer[l]["correlations"].abs().mean()) for l in layers]

    fig, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(len(layers))
    bars = ax.bar(x, top_rho, color="#3498db", alpha=0.85, label="Top-1 |ρ|")
    ax.plot(x, mean_rho, "o--", color="#e74c3c", linewidth=1.5,
            markersize=5, label="Mean |ρ| across all features")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in layers], rotation=30, ha="right")
    ax.set_ylabel("Spearman ρ")
    ax.set_title(f"Phase 1 — Uncertainty Feature Correlations by Layer\n({model_key})")
    ax.set_ylim(0, max(top_rho) * 1.25)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="y", alpha=0.3)

    # Annotate bar tops
    for bar, rho in zip(bars, top_rho):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{rho:.3f}", ha="center", va="bottom", fontsize=8)

    # Summary annotation
    entropy_scores = p1.get("entropy_scores")
    correct_mask   = p1.get("correct_mask")
    if entropy_scores is not None and correct_mask is not None:
        n = len(entropy_scores)
        mean_h = float(entropy_scores.mean())
        correct_rate = float(correct_mask.float().mean())
        ax.annotate(
            f"n={n} prompts  |  mean entropy={mean_h:.2f}  |  correct@5={correct_rate:.1%}",
            xy=(0.01, 0.97), xycoords="axes fraction",
            va="top", ha="left", fontsize=8, color="#555555",
        )

    fig.tight_layout()
    save_fig(fig, out_dir, "fig1_phase1_correlations.png")


# ---------------------------------------------------------------------------
# Figure 2 — Phase 1: Entropy distribution across prompts
# ---------------------------------------------------------------------------

def plot_phase1_entropy_dist(p1: dict, out_dir: Path, model_key: str,
                              entropy_threshold: float) -> None:
    if not p1:
        return
    entropy_scores = p1.get("entropy_scores")
    if entropy_scores is None:
        return

    h = entropy_scores.numpy()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(h, bins=60, color="#3498db", alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvline(entropy_threshold, color="#e74c3c", linewidth=1.8, linestyle="--",
               label=f"Confident threshold ({entropy_threshold})")
    ax.set_xlabel("Output logit entropy (nats)")
    ax.set_ylabel("Number of prompts")
    ax.set_title(f"Phase 1 — Output Entropy Distribution\n({model_key})")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    n_confident = int((entropy_scores < entropy_threshold).sum())
    ax.annotate(
        f"{n_confident} prompts below threshold ({n_confident/len(h):.1%})",
        xy=(entropy_threshold, ax.get_ylim()[1] * 0.9),
        xytext=(entropy_threshold + 0.3, ax.get_ylim()[1] * 0.9),
        arrowprops=dict(arrowstyle="->", color="#e74c3c"),
        fontsize=8, color="#e74c3c",
    )

    fig.tight_layout()
    save_fig(fig, out_dir, "fig2_phase1_entropy_dist.png")


# ---------------------------------------------------------------------------
# Figure 3 — Phase 3: CA / Drift / Degradation breakdown per layer
# ---------------------------------------------------------------------------

def plot_phase3_label_breakdown(p3: dict, out_dir: Path, model_key: str) -> None:
    if not p3:
        return
    per_layer = p3.get("per_layer", {})
    if not per_layer:
        return

    layers = sorted(per_layer.keys())
    labels = [CALIBRATION_APPROPRIATE, SEMANTIC_DRIFT, DEGRADATION]

    # Collect counts: ablation only (primary evidence)
    abl_counts = {lab: [] for lab in labels}
    # Also collect amplification counts for supplementary subplot
    amp_counts  = {lab: [] for lab in labels}

    for layer in layers:
        layer_data = per_layer[layer]
        abl_tally = {l: 0 for l in labels}
        amp_tally  = {l: 0 for l in labels}
        for feat_data in layer_data.values():
            for key, result in feat_data.items():
                lab = result["label"]
                if lab not in abl_tally:
                    lab = DEGRADATION
                if key == "ablation":
                    abl_tally[lab] += 1
                else:
                    amp_tally[lab] += 1
        for lab in labels:
            abl_counts[lab].append(abl_tally[lab])
            amp_counts[lab].append(amp_tally[lab])

    x = np.arange(len(layers))
    width = 0.55

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    for ax, counts, title_tag in [
        (axes[0], abl_counts, "Ablation (primary)"),
        (axes[1], amp_counts, "Amplification (supplementary)"),
    ]:
        bottoms = np.zeros(len(layers))
        for lab in labels:
            vals = np.array(counts[lab], dtype=float)
            bars = ax.bar(x, vals, width, bottom=bottoms,
                          color=LABEL_COLORS[lab],
                          label=LABEL_SHORT[lab], alpha=0.88)
            # Annotate CA bars only
            if lab == CALIBRATION_APPROPRIATE:
                for bar, v, b in zip(bars, vals, bottoms):
                    if v > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                b + v / 2, str(int(v)),
                                ha="center", va="center", fontsize=8,
                                color="white", fontweight="bold")
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.set_ylabel("Number of interventions")
        ax.set_title(f"Phase 3 — Outcome Breakdown per Layer\n{title_tag}")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Model: {model_key}", fontsize=10, y=1.01)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig3_phase3_label_breakdown.png")


# ---------------------------------------------------------------------------
# Figure 4 — Phase 3: Δentropy distribution for CA ablations
# ---------------------------------------------------------------------------

def plot_phase3_entropy_delta_dist(p3: dict, out_dir: Path, model_key: str,
                                    min_entropy_delta: float) -> None:
    if not p3:
        return
    per_layer = p3.get("per_layer", {})
    if not per_layer:
        return

    # Collect Δentropy for ALL ablations (CA and non-CA) to show the filter effect
    all_deltas = []
    ca_deltas  = []

    for layer_data in per_layer.values():
        for feat_data in layer_data.values():
            abl = feat_data.get("ablation")
            if abl is None:
                continue
            delta = abl["entropy_delta"]
            all_deltas.append(delta)
            if abl["label"] == CALIBRATION_APPROPRIATE:
                ca_deltas.append(delta)

    if not all_deltas:
        return

    all_deltas = np.array(all_deltas)
    ca_deltas  = np.array(ca_deltas)

    fig, ax = plt.subplots(figsize=(6, 3.8))

    bins = np.linspace(all_deltas.min() - 0.05, max(all_deltas.max(), 0.5) + 0.05, 60)
    ax.hist(all_deltas, bins=bins, color="#95a5a6", alpha=0.6,
            edgecolor="white", linewidth=0.3, label="All ablations")
    if len(ca_deltas):
        ax.hist(ca_deltas, bins=bins, color="#2ecc71", alpha=0.85,
                edgecolor="white", linewidth=0.3, label="CA ablations")

    ax.axvline(min_entropy_delta, color="#e74c3c", linewidth=1.8, linestyle="--",
               label=f"min_entropy_delta = {min_entropy_delta}")
    ax.axvline(0, color="#333333", linewidth=1.0, linestyle=":")

    ax.set_xlabel("Mean Δentropy (post − pre)")
    ax.set_ylabel("Number of ablations")
    ax.set_title(f"Phase 3 — Δentropy Distribution (ablations)\n({model_key})")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    med = float(np.median(all_deltas))
    ax.annotate(f"median={med:.4f}", xy=(med, ax.get_ylim()[1] * 0.5),
                xytext=(med + 0.08, ax.get_ylim()[1] * 0.6),
                arrowprops=dict(arrowstyle="->", color="#555"),
                fontsize=8, color="#555")

    fig.tight_layout()
    save_fig(fig, out_dir, "fig4_phase3_entropy_delta_dist.png")


# ---------------------------------------------------------------------------
# Figure 5 — Phase 3: Top CA ablation features (ranked bar)
# ---------------------------------------------------------------------------

def plot_phase3_top_ca_features(p3: dict, out_dir: Path, model_key: str,
                                 top_n: int = 20) -> None:
    if not p3:
        return
    per_layer = p3.get("per_layer", {})
    if not per_layer:
        return

    rows = []
    for layer_idx, layer_data in per_layer.items():
        for feat_idx, feat_data in layer_data.items():
            abl = feat_data.get("ablation")
            if abl and abl["label"] == CALIBRATION_APPROPRIATE:
                rows.append({
                    "layer": layer_idx,
                    "feature": feat_idx,
                    "delta": abl["entropy_delta"],
                    "coherence": abl["coherence_rate"],
                    "sem_sim": abl["semantic_sim"],
                    "hedging": abl["hedging_rate"],
                })

    if not rows:
        print("  [fig5] no CA ablations found — skipping")
        return

    rows.sort(key=lambda r: r["delta"], reverse=True)
    rows = rows[:top_n]

    labels = [f"L{r['layer']} f{r['feature']}" for r in rows]
    deltas = [r["delta"] for r in rows]
    hedgings = [r["hedging"] for r in rows]

    x = np.arange(len(rows))
    fig, ax1 = plt.subplots(figsize=(max(8, len(rows) * 0.55), 4.5))

    bars = ax1.bar(x, deltas, color="#2ecc71", alpha=0.85, label="Δentropy")
    ax1.set_ylabel("Mean Δentropy")
    ax1.set_ylim(0, max(deltas) * 1.3)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, hedgings, "o--", color="#e67e22", linewidth=1.5,
             markersize=5, label="Hedging rate")
    ax2.set_ylabel("Post-intervention hedging rate")
    ax2.set_ylim(0, 1.05)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_title(f"Phase 3 — Top {len(rows)} CA Ablation Features by Δentropy\n({model_key})")

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right")

    fig.tight_layout()
    save_fig(fig, out_dir, "fig5_phase3_top_ca_features.png")


# ---------------------------------------------------------------------------
# Figure 6 — Phase 4: Layer potency heatmap
# ---------------------------------------------------------------------------

def plot_phase4_potency_heatmap(p4: dict, out_dir: Path, model_key: str) -> None:
    if not p4:
        return
    layer_potency = p4.get("layer_potency", {})
    if not layer_potency:
        return

    # Build matrix: rows = home layer, cols = probe layer
    all_home_layers  = sorted(set(k[0] for k in layer_potency.keys()))
    all_probe_layers = sorted(set(
        probe for potency in layer_potency.values() for probe in potency.keys()
    ))

    if not all_home_layers or not all_probe_layers:
        return

    # Average Δentropy across features with the same home layer
    matrix = np.zeros((len(all_home_layers), len(all_probe_layers)))
    counts = np.zeros_like(matrix)

    for (home_layer, feat_idx), potency in layer_potency.items():
        row = all_home_layers.index(home_layer)
        for probe_layer, delta in potency.items():
            col = all_probe_layers.index(probe_layer)
            matrix[row, col] += delta
            counts[row, col] += 1

    with np.errstate(invalid="ignore"):
        matrix = np.where(counts > 0, matrix / counts, np.nan)

    fig, ax = plt.subplots(figsize=(max(6, len(all_probe_layers) * 0.9),
                                    max(4, len(all_home_layers) * 0.9)))

    vmax = np.nanmax(np.abs(matrix))
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto",
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(all_probe_layers)))
    ax.set_xticklabels([f"L{l}" for l in all_probe_layers])
    ax.set_yticks(range(len(all_home_layers)))
    ax.set_yticklabels([f"L{l}" for l in all_home_layers])
    ax.set_xlabel("Probe layer (where ablation is applied)")
    ax.set_ylabel("Home layer (where feature lives)")
    ax.set_title(f"Phase 4 — Layer Potency Heatmap (mean Δentropy)\n({model_key})")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean Δentropy")

    # Annotate cells
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            v = matrix[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:+.3f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(v) < vmax * 0.6 else "white")

    fig.tight_layout()
    save_fig(fig, out_dir, "fig6_phase4_potency_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 7 — Phase 4: Redundancy — entropy recovery after home ablation
# ---------------------------------------------------------------------------

def plot_phase4_redundancy(p4: dict, out_dir: Path, model_key: str) -> None:
    if not p4:
        return
    redundancy = p4.get("redundancy", {})
    if not redundancy:
        return

    baseline = p4.get("baseline_entropy", 0.0)

    fig, ax = plt.subplots(figsize=(8, 4))

    plotted = 0
    for (home_layer, feat_idx), redund in redundancy.items():
        if not redund:
            continue
        probe_layers = sorted(redund.keys())
        deltas = [redund[l] for l in probe_layers]
        ax.plot(probe_layers, deltas, "o-", linewidth=1.4, markersize=5, alpha=0.75,
                label=f"L{home_layer} f{feat_idx}")
        plotted += 1

    ax.axhline(0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.5,
               label="Baseline (no change)")
    ax.axhline(-0.05, color="#e74c3c", linewidth=1.0, linestyle=":",
               label="Recovery threshold (−0.05)")

    ax.set_xlabel("Probe layer")
    ax.set_ylabel("Δentropy vs baseline")
    ax.set_title(f"Phase 4 — Entropy Recovery After Home Ablation\n({model_key},"
                 f" baseline={baseline:.3f})")

    if plotted <= 12:
        ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    else:
        ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")

    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig7_phase4_redundancy.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pipeline results from checkpoints")
    parser.add_argument("--model", default="gpt2-small",
                        choices=["gpt2-small", "gemma-2-9b"],
                        help="Model key (used for titles and output subdirectory)")
    parser.add_argument("--checkpoint-dir", default="intervention/checkpoints",
                        help="Directory containing phase*.pt checkpoint files")
    parser.add_argument("--output-dir", default="intervention/figures",
                        help="Root directory to save figures (a subdir per model is created)")
    parser.add_argument("--dpi", type=int, default=200,
                        help="Figure resolution in DPI")
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir
    out_dir  = Path(args.output_dir) / args.model

    print(f"\nLoading checkpoints from: {ckpt_dir}")
    p1 = load_checkpoint(f"{ckpt_dir}/phase1_correlational.pt")
    p3 = load_checkpoint(f"{ckpt_dir}/phase3_calibration.pt")
    p4 = load_checkpoint(f"{ckpt_dir}/phase4_propagation.pt")

    # Pull config values that affect plot annotations
    # These match the ExperimentConfig defaults; override if you changed them.
    ENTROPY_THRESHOLDS = {"gpt2-small": 4.0, "gemma-2-9b": 2.0}
    MIN_ENTROPY_DELTAS = {"gpt2-small": 0.05, "gemma-2-9b": 0.05}

    entropy_threshold = ENTROPY_THRESHOLDS.get(args.model, 2.0)
    min_entropy_delta = MIN_ENTROPY_DELTAS.get(args.model, 0.05)

    print(f"Saving figures to: {out_dir}/\n")

    plot_phase1_correlations(p1, out_dir, args.model)
    plot_phase1_entropy_dist(p1, out_dir, args.model, entropy_threshold)
    plot_phase3_label_breakdown(p3, out_dir, args.model)
    plot_phase3_entropy_delta_dist(p3, out_dir, args.model, min_entropy_delta)
    plot_phase3_top_ca_features(p3, out_dir, args.model)
    plot_phase4_potency_heatmap(p4, out_dir, args.model)
    plot_phase4_redundancy(p4, out_dir, args.model)

    print(f"\nDone. {len(list(out_dir.glob('*.png')))} figures in {out_dir}/")


if __name__ == "__main__":
    main()
