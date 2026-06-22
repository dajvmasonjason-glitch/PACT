"""独立绘制 axis_profile_heatmap: 直接读取 axis_profile_hot.csv, 不重跑 SVD.

每个 hot layer 一张 T × K 热力图, 显示 V₀,K 内归一化分布 p_i(t, ℓ).
相比 analyze_axis_profile.py 内的版本: 去掉了整图标题 (suptitle).

用法:
    python -m analysis.lbw_subspaces.plot_axis_profile_heatmap \
        --input_csv analysis/lbw_subspaces/outputs/axis_profile_hot.csv \
        --output_dir analysis/lbw_subspaces/outputs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_axis_profile_csv(
    path: str,
) -> Tuple[List[str], List[str], Dict[str, np.ndarray]]:
    """读取 axis_profile_hot.csv.

    返回 (layer_order, task_order, {layer: matrix[T, K]}).
    空单元 (例如某些层 K_eff < 15) 解析为 NaN.
    layer / task 顺序均按文件中首次出现的顺序保留 (= 原始绘图顺序).
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    header = lines[0].split(",")
    n_axes = len(header) - 2                       # 去掉 layer, task 两列

    layer_order: List[str] = []
    task_order: List[str] = []
    rows_by_layer: Dict[str, List[np.ndarray]] = {}

    for ln in lines[1:]:
        parts = ln.split(",")
        layer, task = parts[0], parts[1]
        vals = np.array(
            [float(x) if x != "" else np.nan for x in parts[2:2 + n_axes]],
            dtype=np.float64,
        )
        if layer not in rows_by_layer:
            rows_by_layer[layer] = []
            layer_order.append(layer)
        rows_by_layer[layer].append(vals)
        if task not in task_order:
            task_order.append(task)

    matrices = {ly: np.stack(rows, axis=0) for ly, rows in rows_by_layer.items()}
    return layer_order, task_order, matrices


def plot_axis_profile_heatmap(
    layer_order: List[str],
    task_names: List[str],
    matrices: Dict[str, np.ndarray],
    save_path: str,
) -> None:
    """每个 hot layer 一张 T × K 热力图 (无整图标题)."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })

    n_hot = len(layer_order)
    n_cols = min(3, n_hot)
    n_rows = int(np.ceil(n_hot / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 2.7 * n_rows),
                             squeeze=False)
    cmap = sns.color_palette("rocket_r", as_cmap=True)

    for plot_idx, layer in enumerate(layer_order):
        r, c = divmod(plot_idx, n_cols)
        ax = axes[r][c]
        profile = matrices[layer]                    # (T, K)
        K_eff = int((~np.isnan(profile[0])).sum())
        profile = profile[:, :K_eff]
        sns.heatmap(
            profile, ax=ax,
            cmap=cmap, vmin=0.0,
            cbar=True, cbar_kws={"shrink": 0.7, "pad": 0.02},
            xticklabels=[f"v{i+1}" for i in range(K_eff)],
            yticklabels=task_names,
            linewidths=0.3, linecolor="white",
        )
        ax.set_title(layer, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(length=0)

    for k in range(n_hot, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r][c].axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    fig.savefig(os.path.splitext(save_path)[0] + ".pdf")
    plt.close(fig)


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot axis_profile_heatmap from CSV (no title)"
    )
    parser.add_argument(
        "--input_csv", type=str,
        default=str(here / "ouput_alignment" / "axis_profile_hot.csv"),
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(here / "ouput_alignment"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"CSV not found: {args.input_csv}")
    os.makedirs(args.output_dir, exist_ok=True)

    layer_order, task_order, matrices = load_axis_profile_csv(args.input_csv)
    print(f"[load] {args.input_csv}: "
          f"{len(layer_order)} layers × {len(task_order)} tasks")

    out_path = os.path.join(args.output_dir, "axis_profile_heatmap.png")
    plot_axis_profile_heatmap(layer_order, task_order, matrices, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
