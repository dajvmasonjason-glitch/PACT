"""Module 2: layer scan, mask generation, and diagnostic plots."""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from analysis.lbw_parameters.pact_insight_common import (
    analyzable_keys,
    equalize_masks,
    flatten,
    git_commit,
    quantile,
    random_mask_like,
)


MASK_NAMES = ["mask_crucial", "mask_safe"]


def parse_sweep(value: str) -> List[Tuple[float, float]]:
    pairs = []
    for item in value.split(","):
        if not item.strip():
            continue
        left, right = item.split(":")
        pairs.append((float(left), float(right)))
    return pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate PACT insight masks.")
    p.add_argument("--mod1-path", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--q-a-zero", type=float, default=0.30)
    p.add_argument("--q-a-sensitive", type=float, default=0.70)
    p.add_argument("--q-sweep", default="0.10:0.90,0.20:0.80,0.30:0.70,0.40:0.60")
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-k-layers", type=int, default=3)
    p.add_argument("--layer-regex", default=None)
    p.add_argument("--fisher-key", default="F_A")
    return p.parse_args()


def enrichment_and_stats(mask: torch.Tensor, tau_b: torch.Tensor, n_permutations: int, seed: int) -> Dict[str, float]:
    m = flatten(mask).bool()
    b = flatten(tau_b).pow(2)
    n = int(m.sum().item())
    d = int(m.numel())
    total = float(b.sum().item())
    if n == 0 or d == 0 or total == 0.0:
        return {"n_crucial": n, "enrichment": float("nan"), "p_value": 1.0, "cohens_d": float("nan")}
    observed = float(b[m].sum().item() / total) / (n / d)
    gen = torch.Generator().manual_seed(seed)
    ge = 0
    for _ in range(n_permutations):
        idx = torch.randperm(d, generator=gen)[:n]
        val = float(b[idx].sum().item() / total) / (n / d)
        ge += int(val >= observed)
    inside = flatten(tau_b).abs()[m]
    outside = flatten(tau_b).abs()[~m]
    pooled = math.sqrt((float(inside.var(unbiased=False)) + float(outside.var(unbiased=False))) / 2.0 + 1e-12)
    cohen = (float(inside.mean()) - float(outside.mean())) / pooled
    return {
        "n_crucial": n,
        "enrichment": observed,
        "p_value": (ge + 1) / (n_permutations + 1),
        "cohens_d": cohen,
    }


def raw_masks_for_layer(tau_a: torch.Tensor, fisher: torch.Tensor, q_zero: float, q_sens: float) -> Dict[str, torch.Tensor]:
    abs_a = tau_a.abs()
    mask_a_low = abs_a < quantile(abs_a, q_zero)
    mask_f_high = fisher > quantile(fisher, q_sens)
    mask_f_low = fisher < quantile(fisher, 1.0 - q_sens)
    return {
        "mask_crucial": mask_a_low & mask_f_high,
        "mask_safe": mask_a_low & mask_f_low,
    }


def equalized_with_random(raw: Dict[str, torch.Tensor], seed: int) -> Dict[str, torch.Tensor]:
    eq = equalize_masks(raw, seed)
    k = int(next(iter(eq.values())).sum().item()) if eq else 0
    shape = next(iter(raw.values())).shape
    eq["mask_random"] = random_mask_like(shape, k, seed + 1000003)
    eq["k"] = k
    return eq


def plot_layer_bar(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    plot_df = df.sort_values(["p_value", "enrichment"]).head(30)
    plt.figure(figsize=(12, max(4, 0.28 * len(plot_df))))
    plt.barh(plot_df["layer"], plot_df["enrichment"])
    plt.axvline(1.0, color="black", linewidth=1)
    plt.xlabel("Enrichment ratio")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_insight(tau_a: torch.Tensor, tau_b: torch.Tensor, fisher: torch.Tensor, masks: Dict[str, torch.Tensor], path: Path) -> None:
    low = flatten(tau_a.abs()) < quantile(tau_a.abs(), 0.2)
    x = flatten(tau_b.abs())[low]
    y = flatten(fisher)[low]
    if x.numel() > 10000:
        gen = torch.Generator().manual_seed(0)
        idx = torch.randperm(x.numel(), generator=gen)[:10000]
        x, y = x[idx], y[idx]
    plt.figure(figsize=(7, 5))
    plt.hexbin(x.numpy(), y.numpy(), gridsize=60, mincnt=1, bins="log", cmap="Blues")
    for name, color in [("mask_crucial", "red"), ("mask_safe", "green")]:
        m = flatten(masks[name]).bool()
        plt.scatter(flatten(tau_b.abs())[m].numpy(), flatten(fisher)[m].numpy(), s=5, c=color, label=name, alpha=0.7)
    plt.xlabel("|tau_B|")
    plt.ylabel("F_A")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main() -> None:
    cli = parse_args()
    mod1_path = Path(cli.mod1_path)
    out_dir = Path(cli.output_dir) if cli.output_dir else mod1_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    mod1 = torch.load(mod1_path, map_location="cpu")
    tau_a = mod1["tau_A"]
    tau_b = mod1["tau_B"]
    fisher = mod1[cli.fisher_key]

    rows = []
    raw_by_layer = {}
    for key in analyzable_keys(tau_a, cli.layer_regex):
        if key not in tau_b or key not in fisher:
            continue
        raw = raw_masks_for_layer(tau_a[key], fisher[key], cli.q_a_zero, cli.q_a_sensitive)
        raw_by_layer[key] = raw
        stats = enrichment_and_stats(raw["mask_crucial"], tau_b[key], cli.n_permutations, cli.seed)
        rows.append({"layer": key, **stats})

    scan = pd.DataFrame(rows).sort_values(["p_value", "enrichment"], ascending=[True, False])
    scan.to_csv(out_dir / "mod2a_layer_scan.csv", index=False)
    plot_layer_bar(scan, out_dir / "mod2_layer_bar.png")

    # Select layers based on enrichment > 1.0 and p_value < 0.05 (consistent with Module 3)
    significant = scan[(scan["p_value"] < 0.05) & (scan["enrichment"] > 1.0)]
    selected = significant["layer"].tolist()

    # Fallback to top-k if no significant layers found
    if len(selected) == 0:
        print(f"⚠️  No layers meet criteria (p_value < 0.05 and enrichment > 1.0)")
        print(f"Falling back to top {cli.top_k_layers} layers")
        selected = scan.head(cli.top_k_layers)["layer"].tolist()
    else:
        print(f"✓ Selected {len(selected)} layers with enrichment > 1.0 and p_value < 0.05")

    masks = {}
    raw_selected = {}
    for layer in selected:
        raw = raw_by_layer[layer]
        raw_selected[layer] = raw
        masks[layer] = equalized_with_random(raw, cli.seed)
        plot_insight(tau_a[layer], tau_b[layer], fisher[layer], masks[layer], out_dir / f"mod2_insight_plot_{layer.replace('/', '_').replace('.', '_')}.png")

    sweep_rows = []
    sweep_masks = {}
    for q_zero, q_sens in parse_sweep(cli.q_sweep):
        sweep_key = f"{q_zero:.4f}_{q_sens:.4f}"
        sweep_masks[sweep_key] = {}
        for layer in selected:
            raw = raw_masks_for_layer(tau_a[layer], fisher[layer], q_zero, q_sens)
            eq = equalized_with_random(raw, cli.seed)
            sweep_masks[sweep_key][layer] = eq
            stats = enrichment_and_stats(raw["mask_crucial"], tau_b[layer], cli.n_permutations, cli.seed)
            sweep_rows.append({"q_zero": q_zero, "q_sens": q_sens, "layer": layer, "k": eq["k"], **stats})
    pd.DataFrame(sweep_rows).to_csv(out_dir / "mod2b_threshold_sweep.csv", index=False)

    payload = {
        "selected_layers": selected,
        "masks": masks,
        "raw_masks": raw_selected,
        "threshold_sweep_masks": sweep_masks,
        "meta": {
            "module": "mod2_masks",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "mod1_path": str(mod1_path),
            "q_A_zero": cli.q_a_zero,
            "q_A_sensitive": cli.q_a_sensitive,
            "q_sweep": cli.q_sweep,
            "n_permutations": cli.n_permutations,
            "seed": cli.seed,
            "fisher_key": cli.fisher_key,
        },
    }
    torch.save(payload, out_dir / "mod2_masks.pt")
    print(f"Saved {out_dir / 'mod2_masks.pt'}")


if __name__ == "__main__":
    main()
