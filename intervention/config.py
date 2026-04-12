"""
Single source of truth for all model-specific and experiment-wide configuration.
To switch models, change ACTIVE_MODEL at the bottom of this file.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional
import torch as _torch


@dataclass
class ModelConfig:
    # HuggingFace / TransformerLens model identifier
    model_name: str
    # SAELens pretrained release identifier
    sae_release: str
    # SAE ID template — format with layer index.  Used when sae_ids is None.
    # NOTE: this is the SAE identifier passed to SAE.from_pretrained(), NOT the
    # TransformerLens hook name.  Hook names are read from sae.cfg.metadata
    # after loading, so they're always correct regardless of model family.
    sae_id_template: str
    # Per-layer SAE ID overrides.  When set, takes precedence over sae_id_template.
    # Required for Gemma-scope where each layer has a different L0 variant.
    sae_ids: Optional[dict[int, str]]
    # Total number of transformer layers
    n_layers: int
    # Residual stream dimension
    d_model: int
    # Layers to extract SAEs for (subset for efficiency)
    layers_to_analyze: list[int]
    # Batch size tuned for this model on the target device
    batch_size_local: int   # MPS / CPU
    batch_size_server: int  # CUDA
    # dtype to load the model in. None = TransformerLens default (float32).
    # Use torch.bfloat16 for large models to avoid OOM.
    dtype: Optional[_torch.dtype]
    # Entropy threshold for "confident" prompt selection.
    # Tune per-model: GPT-2 Small has high entropy (~4.8 mean on TriviaQA);
    # Gemma-2 9B is much more accurate so a tighter threshold applies.
    entropy_confident_threshold: float

    def get_sae_id(self, layer: int) -> str:
        if self.sae_ids is not None:
            return self.sae_ids[layer]
        return self.sae_id_template.format(layer=layer)


MODEL_CONFIGS: dict[str, ModelConfig] = {
    # -------------------------------------------------------------------------
    # Development model: GPT-2 Small
    # SAE release: gpt2-small-res-jb
    # All layers 0-10 have resid_pre SAEs; layer 11 also has resid_pre.
    # Adding layers 10 and 11 to see whether the layer-9 convergence signal
    # continues or drops off in the final two layers.
    # -------------------------------------------------------------------------
    "gpt2-small": ModelConfig(
        model_name="gpt2",
        sae_release="gpt2-small-res-jb",
        sae_id_template="blocks.{layer}.hook_resid_pre",
        sae_ids=None,
        n_layers=12,
        d_model=768,
        layers_to_analyze=[0, 3, 6, 9, 10, 11],
        batch_size_local=16,
        batch_size_server=64,
        dtype=None,  # float32 fine for 117M params
        entropy_confident_threshold=4.0,  # GPT-2 Small mean entropy on TriviaQA ~4.8
    ),
    # -------------------------------------------------------------------------
    # Full-experiment model: Gemma-2 9B
    # SAE release: gemma-scope-9b-pt-res
    # Each layer has different available L0 variants; sae_ids maps each target
    # layer to a verified, existing SAE ID (medium sparsity, width_16k).
    # Run on school GPU servers (fits in 32 GB at float16, ~18 GB).
    # -------------------------------------------------------------------------
    "gemma-2-9b": ModelConfig(
        model_name="gemma-2-9b",  # TransformerLens name — NOT the HuggingFace path
        sae_release="gemma-scope-9b-pt-res",
        sae_id_template="",  # not used — sae_ids takes precedence
        sae_ids={
            8:  "layer_8/width_16k/average_l0_51",
            16: "layer_16/width_16k/average_l0_75",
            24: "layer_24/width_16k/average_l0_61",
            32: "layer_32/width_16k/average_l0_61",
            40: "layer_40/width_16k/average_l0_61",
        },
        n_layers=42,
        d_model=3584,
        layers_to_analyze=[8, 16, 24, 32, 40],
        batch_size_local=2,
        batch_size_server=16,
        dtype=_torch.bfloat16,  # float32 = ~36GB (OOM); bfloat16 = ~18GB
        entropy_confident_threshold=2.0,  # Gemma-2 9B is accurate; expect low entropy
    ),
}


@dataclass
class ExperimentConfig:
    # ---- model choice -------------------------------------------------------
    model_key: str = "gpt2-small"

    # ---- dataset ------------------------------------------------------------
    dataset_name: str = "trivia_qa"
    dataset_config: str = "rc.nocontext"
    n_prompts: int = 3000           # prompts used in Phase 1 correlational pass
    n_confident_prompts: int = 200  # confident+correct prompts used in Phase 2

    # ---- phase 1 ------------------------------------------------------------
    # Number of top correlated features per layer to carry into Phase 2
    top_k_features: int = 50

    # ---- phase 2 ------------------------------------------------------------
    amplification_factors: list[float] = field(default_factory=lambda: [2.0, 5.0])

    # ---- phase 3 calibration ------------------------------------------------
    # Top-k for coherence check: correct answer must remain in top-k after intervention
    coherence_top_k: int = 5
    # Minimum cosine similarity between pre/post outputs to avoid semantic-drift flag
    semantic_similarity_threshold: float = 0.6
    # Minimum mean Δentropy for a calibration-appropriate classification.
    # Filters out interventions that technically pass coherence/semantic checks but
    # produce no meaningful shift in confidence (Δentropy ≈ 0.000).
    # GPT-2 Small analysis: median CA Δentropy was 0.003; 0.05 retains ~13% of CA results
    # but those are the ones with genuine signal.
    min_entropy_delta: float = 0.05
    # Sentence encoder model for semantic similarity
    sentence_encoder_model: str = "all-MiniLM-L6-v2"
    # Max tokens to generate for hedging / calibration checks.
    # 15 is enough for semantic similarity and hedging detection; 50 causes
    # ~27 hrs of generation on an L40S due to no KV cache in run_with_hooks.
    generation_max_tokens: int = 15
    # Only run generation-based classification for ablations.
    # Amplification results are supplementary; their entropy/coherence data
    # already exists in the Phase 2 checkpoint. Skipping amp generation gives
    # a 3× speedup in Phase 3 with no loss to the primary ablation results.
    phase3_ablation_only: bool = True
    # How many top features per layer to classify in Phase 3.
    # Phase 2 stores top_k_features (50); we only need the strongest signal.
    # Reducing to 25 gives a 2× speedup with minimal loss of interesting results.
    phase3_top_k: int = 25

    # ---- checkpointing ------------------------------------------------------
    checkpoint_dir: str = "intervention/checkpoints"

    # ---- runtime ------------------------------------------------------------
    seed: int = 42

    @property
    def model(self) -> ModelConfig:
        return MODEL_CONFIGS[self.model_key]

    @property
    def batch_size(self) -> int:
        return self.model.batch_size_server if device() == "cuda" else self.model.batch_size_local

    @property
    def entropy_confident_threshold(self) -> float:
        return self.model.entropy_confident_threshold


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Change this to switch between development and full-experiment runs.
# ---------------------------------------------------------------------------
ACTIVE_MODEL: str = "gpt2-small"
