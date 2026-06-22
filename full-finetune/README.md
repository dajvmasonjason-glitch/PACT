# PACT: Full Fine-Tuning Model Merging

This folder contains the **full fine-tuning (full-finetune)** branch of the **PACT** project, which performs full-parameter merging of vision models (ViT-B-16, ViT-B-32, ViT-L-14) across 8 tasks.

The codebase follows the structure of **[Iso-Merging](https://github.com/danielm1405/iso-merging)** (ICML 2025), with PACT-specific additions listed below.

## 🆕 PACT Additions

### Merging Methods
PACT introduces new merging methods alongside the original Iso-C / Iso-CTS baselines:

| Method | Description |
|--------|-------------|
| `PACT` | Preserving Anchored Cores in Task-vectors — a pre-processing step that identifies LBW dimensions, aligns task vector orthogonal complements with the pre-trained weight subspace, and removes aligned components before merging. Can be combined with any merging backbone. |
| `SIFT` | Static-PACT (Appendix) — a simplified variant that directly removes task vector projections onto the pre-trained core space, without extracting per-task top-k directions.|

Available backbone combinations:
- `pact_ta` / `pact_ta_rsvd` — PACT + Task Arithmetic
- `pact_isoc` / `pact_isoc_rsvd` — PACT + Iso-C
- `pact_isocts` — PACT + Iso-CTS
- `pact_tsvm` / `pact_tsvm_rsvd` — PACT + TSVM
- `sift_ta` / `sift_isoc` — SIFT + TA / SIFT + Iso-C

Configs with `_rsvd` suffix use randomized SVD for improved scalability on larger models.

Key implementation files:
- `src/utils/pact_utils.py` — Core PACT logic (LBW subspace alignment, task vector filtering)
- `src/utils/sift_utils.py` — SIFT (static-PACT) filtering
- `config/method/pact_*.yaml` — PACT method configs for each merging backbone

### Hyperparameters
All PACT hyperparameters are configured in `config/method/pact_*.yaml`:

**Rank selection** (`rank_selection`): controls how the pre-trained core space dimension is determined.
- `fixed` — uses a fixed ratio of the total dimension. Set `K_ratio` (pre-trained core ratio, default 0.8) and `k_per_task` (dimensions per task, default 10).
- `adaptive` — uses energy-based thresholds on the singular value spectrum. Set `tau_pre` (pre-trained energy threshold, default 0.85) and `tau_task` (task energy threshold, default 0.95).

**Layer ratio** (`pact_layer_ratio`): controls which fraction of 2D weight matrices PACT is applied to, counting from shallow to deep layers. Default is `1.0` (all layers). For example, `0.5` applies PACT only to the first half of layers. The merging backbone (TA, Iso-C, etc.) still operates on all layers regardless.

**Layer type filtering** (`pact_include_patterns` / `pact_exclude_patterns`): selectively apply PACT filtering to specific layer types by name pattern (e.g., `["attn.in_proj", "mlp.c_fc"]`). Set to `null` (default) to apply to all 2D layers.

### Motivation Analysis
The `analysis/` folder contains code to reproduce all motivation experiments from the PACT paper:

| Directory | Content |
|-----------|---------|
| `analysis/lbw_parameters/` | Scalar LBW analysis pipeline (M0-M4 experiments, merge recovery, layer-wise diagnostics) |
| `analysis/lbw_subspaces/` | Subspace-level LBW analysis (subspace ablation, Intrusion Energy E_in, Sacred Similarity & Hidden Interference heatmaps) |

See `analysis/lbw_parameters/README_PACT_INSIGHT.md` and `analysis/lbw_parameters/README_NEW_EXPERIMENTS.md` for detailed usage.

## 🚀 Setup

### Download fine-tuned checkpoints
Use the checkpoints provided by [Task Singular Vectors](https://drive.google.com/drive/folders/1UEM1Thcz1c7dc1nji1i5uTN53Kf6G3-e?usp=sharing) (which are the same as provided by [Tall Masks](https://drive.google.com/drive/folders/15ParSng4d5xSdaWdBFsg1617zPXT8Dae)).

### Download the datasets
Most datasets being used should be downloaded automatically with `torchvision` or `huggingface`. For the datasets requiring manual preparation (like Cars, DTD, EuroSAT, SUN397), please follow the instructions in [this issue](https://github.com/mlfoundations/task_vectors/issues/1).

### Set data and models locations
Modify `model_location` and `data_location` in `config/config.yaml` before evaluation.

### Prepare the environment
This project follows the Iso-Merging dependency stack (PyTorch, Hydra, wandb, timm). See the Iso-Merging repository for full environment setup instructions.

## 🔄 Merging methods

### Original Iso-Merging Methods
- **Iso-C**: Isotropic Merging in Common Subspace — merge by Task Arithmetic and make the singular value spectrum uniform.
- **Iso-CTS**: Isotropic Merging in Common and Task-Specific Subspaces — merge in common subspace, replace least significant singular vectors with task-specific ones.

### PACT Method
- **PACT**: Preserving Anchored Cores in Task-vectors — a pre-processing step that identifies Load-Bearing Wall (LBW) dimensions (task-critical knowledge embedded in pre-trained weights), aligns their orthogonal complements with the pre-trained weight subspace, and removes the aligned components from task vectors. The filtered task vectors can then be merged with any existing method. PACT variants are named `pact_<backbone>` (e.g., `pact_ta`, `pact_isoc`, `pact_isocts`, `pact_tsvm`), with `_rsvd` suffixes for the randomized SVD variant.

## 🧪 Merge and eval
```bash
model=ViT-B-16
num_tasks=8

# Original Iso-Merging methods
python main.py method="iso_c" model=${model} num_tasks=${num_tasks}
python main.py method="iso_cts" model=${model} num_tasks=${num_tasks} method.common_space_fraction=0.8

# PACT + Task Arithmetic
python main.py method="pact_ta" model=${model} num_tasks=${num_tasks}
python main.py method="pact_ta_rsvd" model=${model} num_tasks=${num_tasks}

# PACT + Iso-C
python main.py method="pact_isoc" model=${model} num_tasks=${num_tasks}
python main.py method="pact_isoc_rsvd" model=${model} num_tasks=${num_tasks}

# PACT + Iso-CTS
python main.py method="pact_isocts" model=${model} num_tasks=${num_tasks} method.common_space_fraction=0.8

# PACT + TSVM
python main.py method="pact_tsvm" model=${model} num_tasks=${num_tasks}
python main.py method="pact_tsvm_rsvd" model=${model} num_tasks=${num_tasks}
```

## 🤝 Acknowledgements

This codebase follows the structure of **[Iso-Merging](https://github.com/danielm1405/iso-merging)** (ICML 2025), which itself builds on [Task Singular Vectors](https://github.com/AntoAndGar/task_singular_vectors) and [Tall Masks](https://github.com/nik-dim/tall_masks).

