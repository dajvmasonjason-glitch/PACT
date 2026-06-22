# Datasets

This structure follows the KnOTS repo. Instructions for adding datasets:

## Adding Datasets
Adding a new dataset requires three (optionally four) parts.

#### Dataset Definition
Define your new dataset following the same pattern as existing dataset files: write `prepare_train_loaders` and `prepare_test_loaders` functions. Each prepares (1) a dataloader for the full dataset, and (2) optionally dataloaders for subdatasets.

#### Add Dataset Config
Add a succinct dictionary to `configs.py` in this directory. Example:
```python
stanford_cars = {
    'wrapper': Cars,
    'batch_size': 128,
    'res': 224,
    'type': 'stanford_cars',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/stanford_cars'
}
```
The variable name must match the dataset name in the experiment config.

#### (Optional) Extract CLIP Encodings
For CLIP-style models, run `parsing/generate_clip_heads.py` and add label templates in `templates.py` if needed.

#### Add Dataset Parser
Add an `elif` branch in `utils.py` (in `prepare_data`) that checks for your dataset name and loads the train/test loaders.
