# PACT: Full Fine-Tuning Model Merging

This folder contains the **full fine-tuning (full-finetune)** branch of the **PACT** project, which performs full-parameter merging of vision models (ViT-B-16, ViT-L-14) across 8 tasks.

The codebase follows the structure of **[Iso-Merging](https://github.com/danielm1405/iso-merging)** (ICML 2025), with PACT-specific additions listed below.

## 🆕 PACT Additions

### Merging Methods
PACT introduces new merging methods alongside the original Iso-C / Iso-CTS baselines:

| Method | Description |
|--------|-------------|
| `PACT` | Principal-Axis Composition Task merging — uses singular value decomposition guided by Fisher information to select and compose task-specific subspaces |

Key implementation files:
- `src/utils/pact_utils.py` — Core PACT merging utilities (Fisher-guided SVD composition)
- `src/merging/pact_merge.py` — PACT merging entry point

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
- **PACT**: Principal-Axis Composition Task merging — Fisher-guided SVD composition that selects and aligns task-specific principal axes, then merges them through an optimized subspace composition.

## 🧪 Merge and eval
```bash
model=ViT-B-16
num_tasks=8

# Original Iso-Merging methods
python main.py method="iso_c" model=${model} num_tasks=${num_tasks}
python main.py method="iso_cts" model=${model} num_tasks=${num_tasks} method.common_space_fraction=0.8

# PACT method
python main.py method="pact" model=${model} num_tasks=${num_tasks}
```

## 🤝 Acknowledgements

This codebase follows the structure of **[Iso-Merging](https://github.com/danielm1405/iso-merging)** (ICML 2025), which itself builds on [Task Singular Vectors](https://github.com/AntoAndGar/task_singular_vectors) and [Tall Masks](https://github.com/nik-dim/tall_masks).

