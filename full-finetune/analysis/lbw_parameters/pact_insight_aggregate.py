"""Module 4: aggregate PACT insight outputs across task pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate PACT insight results.")
    p.add_argument("--output-root", default="analysis/outputs")
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    root = Path(cli.output_root)
    summary = root / "summary"
    summary.mkdir(parents=True, exist_ok=True)

    scan_rows = []
    result_rows = []
    for pair_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "summary"):
        scan_path = pair_dir / "mod2a_layer_scan.csv"
        if scan_path.exists():
            scan = pd.read_csv(scan_path)
            scan["task_pair"] = pair_dir.name
            scan_rows.append(scan)
        result_path = pair_dir / "mod3_results.csv"
        if result_path.exists():
            res = pd.read_csv(result_path)
            res["task_pair"] = pair_dir.name
            result_rows.append(res)

    if scan_rows:
        scans = pd.concat(scan_rows, ignore_index=True)
        scans.to_csv(summary / "all_layer_scan.csv", index=False)
        heat = scans.groupby("task_pair")["enrichment"].mean().to_frame("mean_enrichment")
        heat.to_csv(summary / "fig1_enrichment_values.csv")
        plt.figure(figsize=(8, max(3, 0.35 * len(heat))))
        plt.imshow(heat.values, aspect="auto", cmap="magma")
        plt.colorbar(label="Mean enrichment")
        plt.yticks(range(len(heat.index)), heat.index)
        plt.xticks([0], ["mean"])
        plt.tight_layout()
        plt.savefig(summary / "fig1_enrichment_heatmap.png", dpi=220)
        plt.close()

    if result_rows:
        results = pd.concat(result_rows, ignore_index=True)
        results.to_csv(summary / "all_results.csv", index=False)
        base = results[results["group"] == "G0_base"].groupby(["task_pair", "layer"])["accuracy"].mean()
        drops = []
        for _, row in results.iterrows():
            b = base.get((row["task_pair"], row["layer"]), float("nan"))
            drops.append((b - row["accuracy"]) / b if b and b == b else float("nan"))
        results["relative_drop"] = drops
        plt.figure(figsize=(12, 5))
        groups = list(results["group"].drop_duplicates())
        data = [results[results["group"] == group]["relative_drop"].dropna().values for group in groups]
        plt.boxplot(data, labels=groups, showfliers=False)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Relative accuracy drop")
        plt.tight_layout()
        plt.savefig(summary / "fig2_dropin_boxplot.png", dpi=220)
        plt.close()

        teaser = results.groupby(["task_pair", "group"])["relative_drop"].mean().unstack()
        teaser.to_csv(summary / "fig4_teaser_values.csv")

    print(f"Saved summary outputs to {summary}")


if __name__ == "__main__":
    main()
