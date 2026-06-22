"""Module 3: destructive ablation experiments for PACT insight masks."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from analysis.lbw_parameters.pact_insight_common import (
    build_args,
    equalize_masks,
    evaluate_state_dict,
    git_commit,
    load_task_checkpoints,
    random_mask_like,
)


GROUPS = {
    "G0_base": (None, "base"),
    "G1_crucial_to_pre": ("mask_crucial", "to_pre"),
    "G2_crucial_to_pre_plus_tauB": ("mask_crucial", "to_pre_plus_tauB"),
    "G3_safe_to_pre_plus_tauB": ("mask_safe", "to_pre_plus_tauB"),
    "G6_random_to_pre_plus_tauB": ("mask_random", "to_pre_plus_tauB"),
}

# Groups that support alpha scaling
ALPHA_GROUPS = ["G2_crucial_to_pre_plus_tauB", "G3_safe_to_pre_plus_tauB", "G6_random_to_pre_plus_tauB"]

# Noise-based groups
NOISE_GROUPS = {
    "G7_crucial_noise": ("mask_crucial", "fixed_noise"),
    "G8_safe_noise": ("mask_safe", "fixed_noise"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PACT insight ablations.")
    p.add_argument("--task-a", required=True)
    p.add_argument("--task-b", required=True)
    p.add_argument("--model", default="ViT-B-16")
    p.add_argument("--model-location", default="models/ckpts")
    p.add_argument("--data-location", default="datasets")
    p.add_argument("--mod1-path", required=True)
    p.add_argument("--mod2-path", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layers", default=None, help="Comma-separated override; defaults to mod2 selected_layers.")
    p.add_argument("--skip-threshold-sweep", action="store_true")
    p.add_argument("--alphas", default="1.0,5.0,10.0,20.0,30.0", help="Comma-separated alpha values for tau_B scaling")
    p.add_argument("--noise-std", type=float, default=0.005, help="Fixed standard deviation for Gaussian noise ablation")
    return p.parse_args()


def masks_for_seed(mod2: Mapping, layer: str, seed: int) -> Dict[str, torch.Tensor]:
    if "raw_masks" in mod2 and layer in mod2["raw_masks"]:
        raw = mod2["raw_masks"][layer]
        eq = equalize_masks(raw, seed)
        k = int(next(iter(eq.values())).sum().item()) if eq else 0
        shape = next(iter(raw.values())).shape
        eq["mask_random"] = random_mask_like(shape, k, seed + 1000003)
        eq["k"] = k
        return eq
    return mod2["masks"][layer]


def apply_multi_layer_ablation(
    state_a: Mapping[str, torch.Tensor],
    tau_a: Mapping[str, torch.Tensor],
    tau_b: Mapping[str, torch.Tensor],
    layers: List[str],
    masks_by_layer: Dict[str, Dict[str, torch.Tensor]],
    mask_name: str,
    mode: str,
    alpha: float = 1.0,
    noise_std: float = 0.005,
    seed: int = 0,
) -> tuple[Dict[str, torch.Tensor], int]:
    """Apply ablation across multiple layers simultaneously.

    Args:
        alpha: Scaling factor for tau_B (only used when mode='to_pre_plus_tauB')
        noise_std: Fixed standard deviation for Gaussian noise (only used when mode='fixed_noise')
        seed: Random seed for noise generation
    """
    edited = {k: v.detach().clone() for k, v in state_a.items()}
    if mode == "base":
        return edited, 0

    total_modified = 0
    for layer in layers:
        if layer not in edited:
            continue
        if layer not in masks_by_layer or mask_name not in masks_by_layer[layer]:
            continue

        mask = masks_by_layer[layer][mask_name]
        layer_tensor = edited[layer].detach().cpu()
        layer_mask = mask.to(dtype=torch.bool, device="cpu")

        if mode == "to_pre":
            w_pre = state_a[layer].detach().cpu() - tau_a[layer].detach().cpu()
            layer_tensor[layer_mask] = w_pre[layer_mask].to(layer_tensor.dtype)
        elif mode == "to_pre_plus_tauB":
            w_pre = state_a[layer].detach().cpu() - tau_a[layer].detach().cpu()
            replacement = w_pre + alpha * tau_b[layer].detach().cpu()
            layer_tensor[layer_mask] = replacement[layer_mask].to(layer_tensor.dtype)
        elif mode == "fixed_noise":
            # Fixed Gaussian noise: W_pre + N(0, noise_std)
            w_pre = state_a[layer].detach().cpu() - tau_a[layer].detach().cpu()
            gen = torch.Generator().manual_seed(seed + hash(layer) % 1000000)
            noise = torch.randn(layer_tensor[layer_mask].shape, generator=gen, dtype=layer_tensor.dtype) * noise_std
            layer_tensor[layer_mask] = (w_pre[layer_mask] + noise).to(layer_tensor.dtype)
        else:
            raise ValueError(f"Unknown ablation mode: {mode}")

        edited[layer] = layer_tensor
        total_modified += int(layer_mask.sum().item())

    return edited, total_modified


def plot_summary_bar(df: pd.DataFrame, out_path: Path) -> None:
    """Plot bar chart comparing all groups."""
    summary = df.groupby("group")["accuracy"].agg(["mean", "std"]).reindex(GROUPS.keys())
    plt.figure(figsize=(10, 6))
    x_pos = range(len(summary))
    plt.bar(x_pos, summary["mean"], yerr=summary["std"].fillna(0.0), capsize=5, alpha=0.8)
    plt.xticks(x_pos, summary.index, rotation=25, ha="right")
    plt.ylabel("Task A Accuracy", fontsize=12)
    plt.xlabel("Experimental Group", fontsize=12)
    plt.title("Multi-Layer Simultaneous Ablation Results", fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"Saved summary bar chart: {out_path}")


def plot_alpha_sweep(df: pd.DataFrame, out_path: Path) -> None:
    """Plot line chart showing accuracy vs alpha for G2, G3, G6."""
    alpha_groups = ["G2_crucial_to_pre_plus_tauB", "G3_safe_to_pre_plus_tauB", "G6_random_to_pre_plus_tauB"]
    alpha_df = df[df["group"].isin(alpha_groups)]

    if alpha_df.empty:
        print("⚠️  No alpha sweep data found, skipping alpha sweep plot")
        return

    plt.figure(figsize=(10, 6))

    for group in alpha_groups:
        group_data = alpha_df[alpha_df["group"] == group]
        summary = group_data.groupby("alpha")["accuracy"].agg(["mean", "std"])
        alphas = summary.index.tolist()
        means = summary["mean"].tolist()
        stds = summary["std"].fillna(0.0).tolist()

        plt.plot(alphas, means, marker='o', linewidth=2, label=group, markersize=8)
        plt.fill_between(alphas,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2)

    plt.xlabel("Alpha (τ_B scaling factor)", fontsize=12)
    plt.ylabel("Task A Accuracy", fontsize=12)
    plt.title("Alpha Sweep: Impact of τ_B Scaling on Accuracy", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"Saved alpha sweep plot: {out_path}")



def main() -> None:
    cli = parse_args()
    args = build_args(cli.model, cli.model_location, cli.data_location, cli.batch_size, cli.device, cli.num_workers)
    _, state_a, _, paths = load_task_checkpoints(cli.model_location, cli.model, cli.task_a, cli.task_b)
    mod1 = torch.load(cli.mod1_path, map_location="cpu")
    mod2 = torch.load(cli.mod2_path, map_location="cpu")
    tau_a = mod1["tau_A"]
    tau_b = mod1["tau_B"]
    out_dir = Path(cli.output_dir) if cli.output_dir else Path(cli.mod2_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse alpha values
    alphas = [float(x.strip()) for x in cli.alphas.split(",")]
    print(f"Alpha values for sweep: {alphas}")

    # Use layers from mod2 selected_layers
    significant_layers = list(mod2["selected_layers"])

    if not significant_layers:
        raise ValueError("No layers found in mod2['selected_layers']. Please re-run Module 2.")

    print(f"\n{'='*60}")
    print(f"Target layers for simultaneous ablation: {len(significant_layers)}")
    print(f"{'='*60}")
    for layer in significant_layers:
        print(f"  - {layer}")
    print(f"{'='*60}\n")

    # Prepare masks for all target layers
    masks_by_layer = {}
    for layer in significant_layers:
        masks_by_layer[layer] = masks_for_seed(mod2, layer, cli.seed)

    # Run multi-layer ablation experiments
    rows = []
    alpha_rows = []

    for seed_offset in range(cli.num_seeds):
        seed = cli.seed + seed_offset
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        # Regenerate masks for this seed
        for layer in significant_layers:
            masks_by_layer[layer] = masks_for_seed(mod2, layer, seed)

        # 1. Run baseline groups (G0, G1) with alpha=1.0
        print("\n[Baseline Groups]")
        for group in ["G0_base", "G1_crucial_to_pre"]:
            mask_name, mode = GROUPS[group]
            edited, total_modified = apply_multi_layer_ablation(
                state_a, tau_a, tau_b, significant_layers, masks_by_layer, mask_name, mode,
                alpha=1.0, noise_std=cli.noise_std, seed=seed
            )
            acc = evaluate_state_dict(edited, cli.task_a, args)
            rows.append({
                "seed": seed,
                "group": group,
                "alpha": 1.0,
                "accuracy": acc,
                "num_layers": len(significant_layers),
                "total_params_modified": total_modified
            })
            print(f"  {group}: accuracy={acc:.4f}, modified_params={total_modified:,}")

        # 2. Run alpha sweep for G2, G3, G6
        print("\n[Alpha Sweep Groups]")
        for alpha in alphas:
            print(f"\n  Alpha = {alpha}")
            for group in ALPHA_GROUPS:
                mask_name, mode = GROUPS[group]
                edited, total_modified = apply_multi_layer_ablation(
                    state_a, tau_a, tau_b, significant_layers, masks_by_layer, mask_name, mode,
                    alpha=alpha, noise_std=cli.noise_std, seed=seed
                )
                acc = evaluate_state_dict(edited, cli.task_a, args)
                row_data = {
                    "seed": seed,
                    "group": group,
                    "alpha": alpha,
                    "accuracy": acc,
                    "num_layers": len(significant_layers),
                    "total_params_modified": total_modified
                }
                rows.append(row_data)
                alpha_rows.append(row_data)
                print(f"    {group}: accuracy={acc:.4f}")

        # 3. Run Fixed Gaussian noise groups (G7, G8)
        print(f"\n[Fixed Gaussian Noise Groups] (noise_std={cli.noise_std})")
        for group, (mask_name, mode) in NOISE_GROUPS.items():
            edited, total_modified = apply_multi_layer_ablation(
                state_a, tau_a, tau_b, significant_layers, masks_by_layer, mask_name, mode,
                alpha=1.0, noise_std=cli.noise_std, seed=seed
            )
            acc = evaluate_state_dict(edited, cli.task_a, args)
            rows.append({
                "seed": seed,
                "group": group,
                "alpha": 1.0,
                "accuracy": acc,
                "num_layers": len(significant_layers),
                "total_params_modified": total_modified
            })
            print(f"  {group}: accuracy={acc:.4f}, modified_params={total_modified:,}")

    # Save all results
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "mod3_results.csv", index=False)

    # Calculate and print summary
    print(f"\n{'='*60}")
    print(f"Multi-Layer Ablation Summary")
    print(f"{'='*60}")
    print(f"Layers involved: {len(significant_layers)}")

    # Summary for baseline groups (alpha=1.0)
    baseline_df = df[df["alpha"] == 1.0]
    baseline_summary = baseline_df.groupby("group")["accuracy"].agg(["mean", "std"])
    print("\n[Baseline Results (alpha=1.0)]")
    print(baseline_summary.to_string())

    # Summary for noise groups
    noise_df = df[df["group"].isin(NOISE_GROUPS.keys())]
    if not noise_df.empty:
        noise_summary = noise_df.groupby("group")["accuracy"].agg(["mean", "std"])
        print("\n[Gaussian Noise Results]")
        print(noise_summary.to_string())

    # Plot baseline bar chart (alpha=1.0 only)
    baseline_groups = list(GROUPS.keys())
    baseline_plot_df = df[(df["alpha"] == 1.0) & (df["group"].isin(baseline_groups))]
    if not baseline_plot_df.empty:
        plot_summary_bar(baseline_plot_df, out_dir / "mod3_summary_bar.png")

    # Plot alpha sweep line chart
    alpha_df = pd.DataFrame(alpha_rows)
    if not alpha_df.empty:
        plot_alpha_sweep(alpha_df, out_dir / "mod3_alpha_sweep.png")

    # ========================================================================
    # Model Merging Real-world Verification
    # ========================================================================
    print(f"\n{'='*60}")
    print("Model Merging Real-world Verification")
    print(f"{'='*60}")

    # Step 1: Standard Task Arithmetic Merge (W_pre + tau_A + tau_B)
    print("\n[Step 1] Standard Task Arithmetic Merge...")
    state_merged = {k: v.detach().clone() for k, v in state_a.items()}
    for key in state_merged.keys():
        if key in tau_a and key in tau_b:
            # W_merged = W_pre + tau_A + tau_B
            # Since state_a = W_pre + tau_A, we have:
            # W_merged = state_a + tau_B
            state_merged[key] = state_a[key].detach().cpu() + tau_b[key].detach().cpu()

    acc_standard_merge = evaluate_state_dict(state_merged, cli.task_a, args)
    print(f"  Task A Accuracy (Standard Merge): {acc_standard_merge:.4f}")

    # Step 2: Surgical Protection (Restore crucial parameters to W_pre)
    print("\n[Step 2] Surgical Protection (Restore crucial to W_pre)...")
    state_protected = {k: v.detach().clone() for k, v in state_merged.items()}
    total_protected_params = 0

    for layer in significant_layers:
        if layer not in state_protected:
            continue
        if layer not in masks_by_layer or "mask_crucial" not in masks_by_layer[layer]:
            continue

        mask = masks_by_layer[layer]["mask_crucial"]
        layer_mask = mask.to(dtype=torch.bool, device="cpu")

        # Restore crucial positions to W_pre
        w_pre = state_a[layer].detach().cpu() - tau_a[layer].detach().cpu()
        layer_tensor = state_protected[layer].detach().cpu()
        layer_tensor[layer_mask] = w_pre[layer_mask].to(layer_tensor.dtype)
        state_protected[layer] = layer_tensor
        total_protected_params += int(layer_mask.sum().item())

    acc_protected_merge = evaluate_state_dict(state_protected, cli.task_a, args)
    print(f"  Task A Accuracy (Protected Merge): {acc_protected_merge:.4f}")
    print(f"  Protected parameters: {total_protected_params:,}")

    # Calculate improvement
    improvement = acc_protected_merge - acc_standard_merge
    print(f"\n{'='*60}")
    print("Model Merging Summary")
    print(f"{'='*60}")
    print(f"Task A Acc (Standard W_pre + tau_A + tau_B): {acc_standard_merge:.4f}")
    print(f"Task A Acc (Crucial Protected, restored to pre): {acc_protected_merge:.4f}")
    print(f"Performance Recovery: {improvement:+.4f} ({improvement*100:+.2f}%)")
    print(f"{'='*60}\n")

    # Save detailed results
    payload = {
        "baseline_results": {
            group: {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "per_seed": [float(x) for x in vals.tolist()],
            }
            for group, vals in baseline_df.groupby("group")["accuracy"]
        },
        "alpha_sweep_results": alpha_df.to_dict(orient="records") if not alpha_df.empty else [],
        "noise_results": {
            group: {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "per_seed": [float(x) for x in vals.tolist()],
            }
            for group, vals in noise_df.groupby("group")["accuracy"]
        } if not noise_df.empty else {},
        "merging_results": {
            "standard_merge_acc": float(acc_standard_merge),
            "protected_merge_acc": float(acc_protected_merge),
            "improvement": float(improvement),
            "protected_params": int(total_protected_params),
        },
        "meta": {
            "module": "mod3_ablation_multi_layer",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "task_A": cli.task_a,
            "task_B": cli.task_b,
            "model": cli.model,
            "paths": paths,
            "mod1_path": cli.mod1_path,
            "mod2_path": cli.mod2_path,
            "num_seeds": cli.num_seeds,
            "seed": cli.seed,
            "target_layers": significant_layers,
            "num_target_layers": len(significant_layers),
            "alphas": alphas,
            "noise_std": cli.noise_std,
        },
    }
    torch.save(payload, out_dir / "mod3_results.pt")
    print(f"\n✅ Saved results to {out_dir / 'mod3_results.pt'}")
    print(f"✅ Saved CSV to {out_dir / 'mod3_results.csv'}")
    print(f"✅ Saved baseline plot to {out_dir / 'mod3_summary_bar.png'}")
    if not alpha_df.empty:
        print(f"✅ Saved alpha sweep plot to {out_dir / 'mod3_alpha_sweep.png'}")
    print(f"{'='*60}\n")



if __name__ == "__main__":
    main()
