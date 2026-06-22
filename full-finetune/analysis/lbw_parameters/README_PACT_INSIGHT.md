# PACT Insight Experiment Pipeline

This folder contains a standalone implementation of the PACT insight experiment.
It only reads code from `src/`; all new experiment logic and outputs live under
`analysis/`.

## Files

- `pact_insight_common.py`: shared checkpoint, dataloader, Fisher, mask, and evaluation helpers.
- `pact_insight_compute.py`: module 1, computes `tau_A`, `tau_B`, `F_A`, and auxiliary scores.
- `pact_insight_masks.py`: module 2, scans layers, generates masks, and writes diagnostic plots.
- `pact_insight_ablation.py`: module 3, runs G0-G6 destructive ablations on task A.
- `pact_insight_batch.py`: batch runner for module 1 -> 2 -> 3.
- `pact_insight_aggregate.py`: summary plots/tables across task pairs.

## Expected Checkpoints

The scripts use the existing repository path convention:

```text
{model_location}/{model}/{TaskVal}/nonlinear_finetuned.pt
{model_location}/{model}/MNISTVal/nonlinear_zeroshot.pt
{model_location}/{model}/head_{TaskVal}.pt
```

If a classification head is missing, the existing `src.models.heads` helper will
try to build and save one under `{model_location}/{model}`.

## Single Pair Run

Run commands from the repository root:

```powershell
cd "D:\pythonproject\model merge\pact-main\full-finetune"
conda activate pact_full
```

Module 1:

```powershell
python -m analysis.lbw_parameters.pact_insight_compute `
  --task-a SUN397 `
  --task-b Cars `
  --model ViT-B-16 `
  --model-location models/ckpts `
  --data-location datasets `
  --device cuda `
  --batch-size 32 `
  --fisher-type true `
  --num-mc-samples 1 `
  --fisher-samples 1000
```

Module 2:

```powershell
python -m analysis.lbw_parameters.pact_insight_masks `
  --mod1-path analysis/outputs/SUN397-Cars/mod1_tensors.pt `
  --n-permutations 1000
```

Module 3:

```powershell
python -m analysis.lbw_parameters.pact_insight_ablation `
  --task-a SUN397 `
  --task-b Cars `
  --model ViT-B-16 `
  --model-location models/ckpts `
  --data-location datasets `
  --mod1-path analysis/outputs/SUN397-Cars/mod1_tensors.pt `
  --mod2-path analysis/outputs/SUN397-Cars/mod2_masks.pt `
  --device cuda `
  --batch-size 32 `
  --num-seeds 3
```

## Batch Run

```powershell
python -m analysis.lbw_parameters.pact_insight_batch `
  --task-pairs "SUN397:Cars,EuroSAT:SVHN,GTSRB:MNIST,DTD:RESISC45,MNIST:Cars" `
  --model ViT-B-16 `
  --model-location models/ckpts `
  --data-location datasets `
  --device cuda `
  --batch-size 32 `
  --fisher-samples 1000 `
  --n-permutations 1000 `
  --num-seeds 3
```

Aggregate:

```powershell
python -m analysis.lbw_parameters.pact_insight_aggregate --output-root analysis/outputs
```

## Main Outputs

For each pair `{A}-{B}`:

- `mod1_tensors.pt`
- `mod2a_layer_scan.csv`
- `mod2b_threshold_sweep.csv`
- `mod2_masks.pt`
- `mod2_layer_bar.png`
- `mod2_insight_plot_*.png`
- `mod3_results.csv`
- `mod3_results.pt`
- `mod3_bar_*.png`
- `mod3_layer_heatmap.png`

Summary:

- `analysis/outputs/summary/all_layer_scan.csv`
- `analysis/outputs/summary/all_results.csv`
- `analysis/outputs/summary/fig1_enrichment_heatmap.png`
- `analysis/outputs/summary/fig2_dropin_boxplot.png`

## Notes

- Fisher is computed on a random subset of task A training samples. Increase
  `--fisher-samples` for stronger estimates.
- `--fisher-type true --num-mc-samples 1` is the default main setting.
- Module 3 evaluates on task A validation/test loader using the task A head.
- All masks are stored in original tensor shape and applied directly with
  `tensor[mask]`.
