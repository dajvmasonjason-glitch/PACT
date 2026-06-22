# Configs

Configs define our experimental test suites. The name of each config file describes the (1) model, (2) LoRA rank, (3) merging method used in an experiment, following the convention `<model>_r<rank>_<method>.py`.

This structure follows the KnOTS repo. PACT adds new configs for its merging methods (prefixed with `pact_`).

## Fields
Each config looks like this: 
```python
import os

VIT_ARCH = 'ViT-B-32-CLIP'                               # Model Architecture
MODEL_DIR = ''                                           # Model Directory
CACHE_DIR = ''                                           # Where to cache HF pretrained checkpoints
HEAD_DIR = ''                                            # CLIP Head Directory

config = {
    'dataset': [{                                        # Specifies datasets used
        'name': "<DATASET_NAME>",                        # name of the dataset
        'shuffle_train': True,                           # Whether to shuffle train set
        'crop_ratio': 1.0                                # Image crop ratio
        'clip_encodings': ""                             # Path to CLIP head
        'val_fraction': .2                               # Proportion of test set for validation
        'batch_size': 32,                                # Batch size
        'num_workers': 16,                               # Number of workers
    }, ...],
    'model': {                                           # Specifies types of models used
        'name': 'hf_clip',                               # Model type
        'base_type': "openai/clip-vit-base-patch32",     # HF name
        'cachedir': CACHE_DIR,
        'bases': [...],                                  # Paths to LoRA models
    },
    'ft_config': {                                        # Specifies FT setup
        'type': 'lora',                                   # FT type
        'r': 16,                                          # LoRA Rank
        'lora_alpha': 16,
        'target_modules': ["q_proj", "k_proj", "v_proj", "out_proj"],
        'lora_dropout': 0.1,
        'bias': "none",
    },
    'task_merge_config': {                                # Merging method configuration
        'representation': 'svd-vector',                   # "vector" (TA/TIES/DARE) or "svd-vector" (KnOTS/PACT)
        'sign_resolve_mode': 'sum_of_values',
        'scaling_coeffs': .6,
        'topK': 20,
        'merge_method': 'ties',                           # e.g., ties, dare_ties, pact_ta, pact_isoc, etc.
        'merging_type': 'mean',
        'concat_across_output': True,
        'dare' : False,
        'dare_pruning_coeffs' : 0.0,
    },
    'eval_type': 'clip'
}
```

## PACT Configs
PACT-specific configs (e.g., `vitB_r16_pact_ta.py`, `vitB_r16_pact_isoc_rsvd.py`) use `merge_method: 'pact_ta'` or `'pact_isoc'` in `task_merge_config`, which routes to the PACT merging handler in `task_merger.py`.

## New Configs
Adding new configs can be done simply by following the above format.
