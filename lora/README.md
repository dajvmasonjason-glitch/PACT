# PACT: LoRA Model Merging

This folder contains the **LoRA** branch of the **PACT** project, which performs LoRA-based model merging for vision (ViT-B/32, ViT-L/14) and language (Llama3-8B) models.

The codebase follows the structure of **[KnOTS](https://github.com/gstoica27/KnOTS)** (ICLR 2025), with PACT-specific additions listed below.

## 🆕 PACT Additions

### Merging Methods
PACT introduces new LoRA merging methods that use Fisher-guided SVD composition:

| Method | Config | Description |
|--------|--------|-------------|
| `PACT-TA` | `vitB_r16_pact_ta.py` | PACT with Task Arithmetic — Fisher-guided principal axis composition followed by task arithmetic merging |
| `PACT-TA-RSVD` | `vitB_r16_pact_ta_rsvd.py` | PACT-TA with randomized SVD for efficiency |
| `PACT-IsoC` | `vitB_r16_pact_isoc.py` | PACT with Isotropic merging in common subspace |
| `PACT-IsoC-RSVD` | `vitB_r16_pact_isoc_rsvd.py` | PACT-IsoC with randomized SVD for efficiency |

Key implementation files:
- `pact_ta_rsvd.py` — PACT-TA with randomized SVD
- `pact_isoc_rsvd.py` — PACT-IsoC with randomized SVD
- `task_merger.py` — Updated merger with PACT handler for LoRA merging

### Baseline Methods
The folder also includes KnOTS baseline configs for comparison:
- `vitB_r16_knots_ties.py` / `vitB_r16_knots_dare_ties.py` — KnOTS-TIES / KnOTS-DARE-TIES
- `vitL_r16_knots_ties.py` / `vitL_r16_knots_dare_ties.py` — ViT-L variants
- Standard baselines: `ties`, `dare_ties`, `tv`, `iso_c`, `iso_cts`, `isoc_ns`

## Getting Started
The codebase is built on Python 3 with dependencies defined in the KnOTS environment. 
KnOTS relies on pretrained LoRA checkpoints to perform merging. We release model checkpoints on [HuggingFace](https://huggingface.co/collections/hoffman-lab/knots-model-merging-with-svd-672d3b5fabf766c22989e760).

See the [KnOTS repository](https://github.com/gstoica27/KnOTS) for full environment setup and dependency installation instructions.

## Experiment Pipeline
This repository can train and merge arbitrary LoRA finetuned models based on the [Hugging Face](https://huggingface.co/) library.

### Currently Supported Applications
Supported settings include ViT-B/32, ViT-L/14, and Llama3-8B models, with both per-task and joint evaluation settings.

#### Experiment config
The `configs/` directory contains all experiment configurations. Each config is a Python dict describing the model, dataset, LoRA setup, and merging method. Configs follow the naming convention `<model>_r<rank>_<method>.py`. See `configs/README.md` for details.

#### Training
Training scripts are provided in the `training_scripts/` directory:
- `8vision_training.py` — Vision model fine-tuning
- `nli_training.py` — Language model fine-tuning

#### Evaluation
Evaluation scripts are in `eval_scripts/`:
- `8vision_joint_linearsearch.py` / `8vision_pertask_linearsearch.py` — Vision linear search evaluation
- `8vision_joint.py` / `8vision_pertask.py` — Vision evaluation with fixed hyperparameters
- `nli_pertask_linearsearch.py` — NLI linear search evaluation

## Extending the Codebase
- **New Config**: See [`configs/README.md`](configs/README.md) for instructions on adding new configs.
- **New Model**: See [`models/README.md`](models/README.md) for instructions on adding a new model.
- **New Dataset**: See [`dataset/README.md`](dataset/README.md) for instructions on adding a new dataset.

## 🤝 Acknowledgements
This codebase follows the structure of **[KnOTS](https://github.com/gstoica27/KnOTS)** (ICLR 2025). The PACT-specific merging methods (`pact_ta_rsvd.py`, `pact_isoc_rsvd.py`) and their configs are our additions.
