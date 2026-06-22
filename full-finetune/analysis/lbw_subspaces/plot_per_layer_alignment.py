"""逐层可视化 E_in(t, ℓ): θ₀ 主轴在不同层、不同任务上的能量占比.

读取 analyze_theta0_alignment.py 生成的 per_layer_e_in.csv, 不重新跑 SVD.

产出 3 张图:
  layer_depth_by_type.{png,pdf}
      4 子图 (attn.in_proj / attn.out_proj / mlp.c_fc / mlp.c_proj)
      x = block index (0..11), y = E_in, 每任务一条线.
      展示 depth × layer-type 的双重模式.

  layer_heatmap.{png,pdf}
      53 layers × 8 tasks 的热力图.
      行 = layer (按 state_dict 顺序), 列 = task, 颜色 = E_in.
      一眼看出"早期层 hot, 后期层 cold".

  layer_mean_band.{png,pdf}
      x = layer index, y = E_in 跨任务均值 + ±1 std 带.
      最干净的单线版本, 适合做正文主图.

用法:
    python -m analysis.lbw_subspaces.plot_per_layer_alignment \
        --input_csv analysis/lbw_subspaces/outputs/per_layer_e_in.csv \
        --output_dir analysis/lbw_subspaces/outputs
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# CSV 读取
# ---------------------------------------------------------------------------


def load_per_layer_csv(path: str) -> Tuple[List[str], List[str], np.ndarray]:
    """读取 per_layer_e_in.csv, 返回 (layer_names, task_names, matrix[L,T])."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    header = lines[0].split(",")
    task_names = header[1:]
    layer_names: List[str] = []
    rows: List[List[float]] = []
    for ln in lines[1:]:
        parts = ln.split(",")
        layer_names.append(parts[0])
        rows.append([float(x) for x in parts[1:]])
    return layer_names, task_names, np.asarray(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# 层类型解析: 把 model.visual.transformer.resblocks.N.<type> 抽出 (N, type)
# ---------------------------------------------------------------------------

LAYER_TYPES = [
    ("attn.in_proj_weight",   "attn.in_proj"),
    ("attn.out_proj.weight",  "attn.out_proj"),
    ("mlp.c_fc.weight",       "mlp.c_fc"),
    ("mlp.c_proj.weight",     "mlp.c_proj"),
]


def parse_block_layer(name: str) -> Tuple[int, str] | None:
    """匹配 model.visual.transformer.resblocks.N.<suffix>, 返回 (N, short_type)."""
    m = re.search(r"resblocks\.(\d+)\.(.+)$", name)
    if not m:
        return None
    block_idx = int(m.group(1))
    suffix = m.group(2)
    for full, short in LAYER_TYPES:
        if suffix == full:
            return block_idx, short
    return None


def group_by_layer_type(
    layer_names: List[str], matrix: np.ndarray
) -> Dict[str, Tuple[List[int], np.ndarray]]:
    """返回 {layer_type: (block_indices, sub_matrix[B, T])}."""
    buckets: Dict[str, Dict[int, np.ndarray]] = {short: {} for _, short in LAYER_TYPES}
    for i, name in enumerate(layer_names):
        parsed = parse_block_layer(name)
        if parsed is None:
            continue
        block, short = parsed
        buckets[short][block] = matrix[i]
    out = {}
    for short, blocks in buckets.items():
        if not blocks:
            continue
        idxs = sorted(blocks.keys())
        arr = np.stack([blocks[i] for i in idxs], axis=0)
        out[short] = (idxs, arr)
    return out


# ---------------------------------------------------------------------------
# 图 1: 按层类型分组的折线图
# ---------------------------------------------------------------------------


def plot_layer_depth_by_type(
    grouped: Dict[str, Tuple[List[int], np.ndarray]],
    task_names: List[str],
    save_path: str,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })

    order = ["attn.in_proj", "attn.out_proj", "mlp.c_fc", "mlp.c_proj"]
    order = [k for k in order if k in grouped]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.6), sharex=True)
    axes = axes.flatten()
    palette = sns.color_palette("tab10", len(task_names))

    y_max = 0.0
    for short in order:
        _, arr = grouped[short]
        y_max = max(y_max, float(arr.max()))
    y_max = min(1.0, y_max * 1.08)

    for ax, short in zip(axes, order):
        blocks, arr = grouped[short]
        for t_idx, t_name in enumerate(task_names):
            ax.plot(blocks, arr[:, t_idx],
                    marker="o", markersize=3.5, linewidth=1.4,
                    color=palette[t_idx], alpha=0.92, label=t_name)
        ax.set_title(short, fontweight="bold")
        ax.set_ylim(0, y_max)
        ax.set_xticks(blocks)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel(r"$E_{\mathrm{in}}(t, \ell)$")

    axes[-1].set_xlabel("transformer block index")
    axes[-2].set_xlabel("transformer block index")
    axes[0].legend(loc="upper right", frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    fig.savefig(os.path.splitext(save_path)[0] + ".pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 2: 53 × 8 全层热力图
# ---------------------------------------------------------------------------


def shorten_layer_name(name: str) -> str:
    """state_dict 名 → 短显示名."""
    if "resblocks" in name:
        m = re.search(r"resblocks\.(\d+)\.(.+)$", name)
        if m:
            block = int(m.group(1))
            suffix = m.group(2)
            for full, short in LAYER_TYPES:
                if suffix == full:
                    return f"b{block:02d}.{short}"
            return f"b{block:02d}.{suffix}"
    return name.replace("model.visual.", "").replace("model.", "")


def plot_layer_heatmap(
    layer_names: List[str],
    task_names: List[str],
    matrix: np.ndarray,
    save_path: str,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 7,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })

    short_names = [shorten_layer_name(n) for n in layer_names]
    n_layers, n_tasks = matrix.shape

    fig_h = max(8.0, 0.18 * n_layers + 1.2)
    fig, ax = plt.subplots(figsize=(6.4, fig_h))
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    sns.heatmap(
        matrix, ax=ax,
        cmap=cmap, vmin=0.0, vmax=float(matrix.max()),
        cbar=True,
        cbar_kws={"shrink": 0.55, "pad": 0.02,
                  "label": r"$E_{\mathrm{in}}$"},
        xticklabels=task_names,
        yticklabels=short_names,
        linewidths=0.0,
        square=False,
    )
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.set_yticklabels(short_names, rotation=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=0)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    fig.savefig(os.path.splitext(save_path)[0] + ".pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 3: 跨任务均值 ± std 折线 (干净主图)
# ---------------------------------------------------------------------------


def plot_layer_mean_band(
    layer_names: List[str],
    matrix: np.ndarray,
    save_path: str,
    only_transformer_blocks: bool = True,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })

    # 按 layer type 分四条线 + 阴影带
    grouped: Dict[str, List[Tuple[int, np.ndarray]]] = {
        short: [] for _, short in LAYER_TYPES
    }
    for i, n in enumerate(layer_names):
        p = parse_block_layer(n)
        if p is None and only_transformer_blocks:
            continue
        if p is None:
            continue
        block, short = p
        grouped[short].append((block, matrix[i]))

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    palette = sns.color_palette("tab10", 4)
    order = ["attn.in_proj", "attn.out_proj", "mlp.c_fc", "mlp.c_proj"]

    for color, short in zip(palette, order):
        items = sorted(grouped.get(short, []), key=lambda x: x[0])
        if not items:
            continue
        blocks = [b for b, _ in items]
        arr = np.stack([row for _, row in items], axis=0)   # (B, T)
        mean = arr.mean(axis=1)
        std = arr.std(axis=1)
        ax.plot(blocks, mean, marker="o", markersize=4, linewidth=1.6,
                color=color, label=short)
        ax.fill_between(blocks, mean - std, mean + std,
                        color=color, alpha=0.16, linewidth=0)

    ax.set_xlabel("transformer block index", fontweight="bold")
    ax.set_ylabel(r"$E_{\mathrm{in}}$", fontweight="bold")
    ax.set_xticks(sorted({b for items in grouped.values() for b, _ in items}))
    ax.legend(loc="upper right", frameon=False, ncol=2)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # 加粗 X/Y 轴
    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)
    ax.tick_params(width=1.6)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    fig.savefig(os.path.splitext(save_path)[0] + ".pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 文本摘要
# ---------------------------------------------------------------------------


def print_summary(
    layer_names: List[str],
    task_names: List[str],
    matrix: np.ndarray,
) -> None:
    print("\n[Per-layer summary]")
    print(f"  layers analyzed : {matrix.shape[0]}")
    print(f"  global max E_in : {matrix.max():.4f}  "
          f"(layer={layer_names[int(np.unravel_index(matrix.argmax(), matrix.shape)[0])]}, "
          f"task={task_names[int(np.unravel_index(matrix.argmax(), matrix.shape)[1])]})")
    print(f"  global min E_in : {matrix.min():.4f}  "
          f"(non-zero excluding embedding rows)")
    print(f"  layer-mean range: "
          f"{matrix.mean(axis=1).min():.4f} ~ {matrix.mean(axis=1).max():.4f}")

    # 按层类型聚合
    grouped = group_by_layer_type(layer_names, matrix)
    print("\n  Mean E_in (across all blocks & tasks) by layer type:")
    for short in ["attn.in_proj", "attn.out_proj", "mlp.c_fc", "mlp.c_proj"]:
        if short in grouped:
            _, arr = grouped[short]
            print(f"    {short:<16}: {arr.mean():.4f}   "
                  f"(block 0 mean: {arr[0].mean():.4f}, "
                  f"block -1 mean: {arr[-1].mean():.4f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Per-layer alignment plots")
    parser.add_argument(
        "--input_csv", type=str,
        default=str(here / "ouput_alignment" / "per_layer_e_in.csv"),
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(here / "ouput_alignment"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"CSV not found: {args.input_csv}\n"
                                f"先跑 analyze_theta0_alignment.py")
    os.makedirs(args.output_dir, exist_ok=True)

    layer_names, task_names, matrix = load_per_layer_csv(args.input_csv)
    print(f"[load] {args.input_csv}: "
          f"{len(layer_names)} layers × {len(task_names)} tasks")

    print_summary(layer_names, task_names, matrix)

    grouped = group_by_layer_type(layer_names, matrix)

    out_depth   = os.path.join(args.output_dir, "layer_depth_by_type.png")
    out_heatmap = os.path.join(args.output_dir, "layer_heatmap.png")
    out_band    = os.path.join(args.output_dir, "layer_mean_band.png")

    plot_layer_depth_by_type(grouped, task_names, out_depth)
    plot_layer_heatmap(layer_names, task_names, matrix, out_heatmap)
    plot_layer_mean_band(layer_names, matrix, out_band)

    print(f"\nSaved:")
    print(f"  {out_depth}")
    print(f"  {out_heatmap}")
    print(f"  {out_band}")


if __name__ == "__main__":
    main()
