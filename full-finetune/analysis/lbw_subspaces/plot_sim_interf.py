"""绘制 sacred_similarity.csv 与 hidden_interference.csv 两张 8×8 任务热力图.

样式与 analyze_hot_layers.py 的 plot_heatmap_clean (sim_*.png) 完全一致:
    无标题, 无 color bar, rocket 配色, 方格 + 数值标注, y 轴标签放右侧.

约定 (与原脚本一致):
    sacred_similarity     → vmin=0, vmax=1.0   (相似度天然落在 [0,1])
    hidden_interference   → vmin=0, vmax=None  (干扰量自动取最大值)

用法:
    python -m analysis.lbw_subspaces.plot_sim_interf \
        --output_dir analysis/lbw_subspaces/outputs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_matrix_csv(path: str) -> Tuple[List[str], np.ndarray]:
    """读取首行/首列为任务名的方阵 CSV, 返回 (task_names, matrix[T, T])."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    task_names = lines[0].split(",")[1:]
    rows: List[List[float]] = []
    for ln in lines[1:]:
        parts = ln.split(",")
        rows.append([float(x) for x in parts[1:]])
    return task_names, np.asarray(rows, dtype=np.float64)


def plot_heatmap_clean(
    matrix: np.ndarray,
    task_names: List[str],
    save_path: str,
    vmin: float = 0.0,
    vmax: float | None = None,
    cmap_name: str = "rocket",
    fmt: str = ".2f",
) -> None:
    """无标题、无 color bar 的方格热力图 (复刻 sim_b00.mlp.c_fc.png 样式)."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.linewidth": 0.0,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })

    if vmax is None:
        vmax = float(np.max(matrix))

    cmap = sns.color_palette(cmap_name, as_cmap=True)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    sns.heatmap(
        matrix, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
        annot=True, fmt=fmt, annot_kws={"size": 9, "weight": "bold"},
        xticklabels=task_names, yticklabels=task_names,
        cbar=False, linewidths=0.6, linecolor="white", square=True,
    )
    ax.set_xticklabels(task_names, rotation=45, ha="right", fontsize=10)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_yticklabels(task_names, rotation=0, fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    fig.savefig(os.path.splitext(save_path)[0] + ".pdf")
    plt.close(fig)


def parse_args():
    here = Path(__file__).resolve().parent
    out_default = here / "ouput_alignment"
    parser = argparse.ArgumentParser(
        description="Plot sacred_similarity & hidden_interference (clean style)"
    )
    parser.add_argument(
        "--sim_csv", type=str,
        default=str(out_default / "sacred_similarity.csv"),
    )
    parser.add_argument(
        "--interf_csv", type=str,
        default=str(out_default / "hidden_interference.csv"),
    )
    parser.add_argument("--output_dir", type=str, default=str(out_default))
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sim_tasks, sim_mat = load_matrix_csv(args.sim_csv)
    interf_tasks, interf_mat = load_matrix_csv(args.interf_csv)
    print(f"[load] sim    : {sim_mat.shape} from {args.sim_csv}")
    print(f"[load] interf : {interf_mat.shape} from {args.interf_csv}")

    sim_out = os.path.join(args.output_dir, "sacred_similarity.png")
    interf_out = os.path.join(args.output_dir, "hidden_interference.png")

    plot_heatmap_clean(sim_mat, sim_tasks, sim_out, vmin=0.0, vmax=1.0)
    plot_heatmap_clean(interf_mat, interf_tasks, interf_out,
                       vmin=0.0, vmax=None)

    print(f"Saved:\n  {sim_out}\n  {interf_out}")


if __name__ == "__main__":
    main()
