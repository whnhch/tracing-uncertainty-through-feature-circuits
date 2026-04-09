# tracing-uncertainty-through-feature-circuits

University of Utah — Spring 2026 CS 6966 Group Project

Tracing uncertainty through LLM features using Sparse Autoencoders (SAEs).
We identify SAE features correlated with model uncertainty, intervene on them
causally, and verify that the resulting confidence shifts are calibration-appropriate
(not semantic drift or processing degradation).

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Mac / Linux
# .venv\Scripts\activate         # Windows
```

### 2. Install PyTorch

PyTorch must be installed separately because the right build depends on your hardware.

**Local (M2 Mac — MPS):**
```bash
pip install torch
```

**School servers (CUDA) — match the server's CUDA version:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

To check the CUDA version on a school server: `nvcc --version` or `nvidia-smi`.

### 3. Install all other dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify your device

```python
import torch
print(torch.cuda.is_available())        # True on school servers
print(torch.backends.mps.is_available()) # True on M2 Mac
```

---

## Running the pipeline

All intervention code lives in `intervention/`. The entry point is `run_pipeline.py`.

```bash
cd intervention

# Full pipeline with the default development model (Pythia-1.4B):
python run_pipeline.py

# Full pipeline with Gemma-2 9B (run this on school servers):
python run_pipeline.py --model gemma-2-9b

# Quick smoke test on a small number of prompts:
python run_pipeline.py --n-prompts 50

# Run specific phases only (e.g. re-run Phase 3 after changing thresholds):
python run_pipeline.py --phases 3 4

# Force re-run even if a checkpoint already exists:
python run_pipeline.py --force
```

Each phase saves a checkpoint to `intervention/checkpoints/` before the next
phase begins. If a run fails on the school server, re-running the same command
will resume from the last completed phase automatically.

---

## Model selection

To switch models, either pass `--model` on the command line or change
`ACTIVE_MODEL` in `intervention/config.py`.

| Model | Use case | Fits in 32 GB? | Est. full pipeline |
|---|---|---|---|
| `pythia-1.4b` | Local development, fast iteration | Yes | ~3–6 hrs (M2 Max) |
| `gemma-2-9b` | Full experiment results | Yes (~18 GB) | ~1–3 hrs (A100) |
