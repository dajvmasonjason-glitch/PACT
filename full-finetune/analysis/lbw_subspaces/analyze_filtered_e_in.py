"""PACT 过滤后任务向量的 E_in 分析。

与 analyze_theta0_alignment.py 对照: 原脚本计算的是**原始**任务向量
    Δ_t = W_t - W_0
落在 θ₀ 核心子空间 V₀,K 内的能量占比 E_in。本脚本则先用 PACT 算法对任务
向量做**无干涉正交过滤**, 再计算过滤后 Δ_t^filtered 的 E_in, 用来展示
"过滤后任务更新对 θ₀ 承重墙的占用被显著削弱"这一效果。

PACT 过滤 (与 src/utils/pact_utils.py 完全一致, 固定维度版):
    对每层 2D 权重 W ∈ R^{d_out × d_in}:
        V_pre^K   = W_0 的 top-K 右奇异向量              (K = 15)
        V_t^k     = Δ_t 的 top-k 右奇异向量              (k = 8)
        V_rel^t   = Orth( (I - V_t^k V_t^{k,T}) V_pre^K )   # 任务 t 的隐式依赖空间
        V_protect^j = Orth( ∪_{i≠j} V_rel^i )               # 其余任务的保护空间并集
        Δ_j^filtered = Δ_j - Δ_j V_protect^j (V_protect^j)^T

E_in 指标 (与原脚本同一定义, 只是把 Δ_t 换成 Δ_t^filtered):
    E_in(t,ℓ) = ‖Δ_t^filtered · V_pre^K‖²_F / ‖Δ_t^filtered‖²_F  ∈ [0, 1]

产出 (output_pact_filtered/, 与原 E_in 实验完全一致的图, 均含 pdf):
    per_layer_e_in.{csv, npy}
    per_task_summary.csv
    energy_alignment_stacked.{pdf, png}
    energy_alignment_violin.{pdf, png}
    layer_depth_by_type.{pdf, png}
    layer_heatmap.{pdf, png}
    layer_mean_band.{pdf, png}

用法:
    cd <repo>
    python -m analysis.lbw_subspaces.analyze_filtered_e_in \
        --model ViT-B-16 --model_location models/ckpts --K 15 --k 8
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.variables_and_paths import DATASETS_8  # noqa: E402

from src.utils.pact_utils import (  # noqa: E402
    compute_implicit_reliance_space,
    orthogonal_core_filtering,
)

from analysis.lbw_subspaces.analyze_pact_motivation import (  # noqa: E402
    _is_2d_weight,
    extract_pretrained_core_basis,
    extract_task_explicit_basis,
    load_pretrained_and_task_states,
)

# 复用原 E_in 实验的全部绘图与 I/O 函数, 保证图片样式完全一致。
from analysis.lbw_subspaces.analyze_theta0_alignment import (  # noqa: E402
    plot_stacked_bar,
    plot_per_layer_violin,
    save_layer_matrix_csv,
    save_summary_csv,
)


# ---------------------------------------------------------------------------
# 逐层折线/热力图绘制 (内联自 plot_per_layer_alignment.py, 样式完全一致)
# ---------------------------------------------------------------------------

LAYER_TYPES = [
    ("attn.in_proj_weight",   "attn.in_proj"),
    ("attn.out_proj.weight",  "attn.out_proj"),
    ("mlp.c_fc.weight",       "mlp.c_fc"),
    ("mlp.c_proj.weight",     "mlp.c_proj"),
]


def parse_block_layer(name: str) -> Tuple[int, str] | None:
    """匹配 ...resblocks.N.<suffix>, 返回 (N, short_type)."""
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

    grouped: Dict[str, List[Tuple[int, np.ndarray]]] = {
        short: [] for _, short in LAYER_TYPES
    }
    for i, n in enumerate(layer_names):
        p = parse_block_layer(n)
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
    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)
    ax.tick_params(width=1.6)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    fig.savefig(os.path.splitext(save_path)[0] + ".pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 核心计算: PACT 过滤后逐层 E_in
# ---------------------------------------------------------------------------


def compute_filtered_energy_alignment(
    pretrained_state: Mapping[str, torch.Tensor],
    task_states: Mapping[str, Mapping[str, torch.Tensor]],
    task_order: List[str],
    K: int = 15,
    k: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """逐层先做 PACT 过滤, 再计算过滤后 Δ_t^filtered 的 E_in。

    Returns dict with:
        per_layer        : (L, T)  -- E_in(t, ℓ) on filtered Δ
        in_energy        : (L, T)  -- ‖Δ_t^filtered V_pre^K‖²_F
        delta_sq_norm    : (L, T)  -- ‖Δ_t^filtered‖²_F
        layer_names      : list[str]   长度 L
        layer_mean       : (T,)
        energy_weighted  : (T,)
    """
    n_tasks = len(task_order)
    e_in_layers: List[List[float]] = []
    in_energy_layers: List[List[float]] = []
    delta_sq_layers: List[List[float]] = []
    layer_names: List[str] = []

    for name, W_pre_cpu in pretrained_state.items():
        if not _is_2d_weight(name, W_pre_cpu):
            continue
        if any(name not in task_states[t] for t in task_order):
            continue

        W_pre = W_pre_cpu.to(device=device, dtype=torch.float32)
        V_pre_K = extract_pretrained_core_basis(W_pre, K=K)  # d_in × K_eff

        # --- 步骤 1~3: 每个任务的 Δ_t 与隐式依赖空间 V_rel^t ---
        deltas: List[torch.Tensor] = []
        rel_spaces: List[torch.Tensor] = []
        for t in task_order:
            W_t = task_states[t][name].to(device=device, dtype=torch.float32)
            Delta_t = W_t - W_pre
            V_t_k = extract_task_explicit_basis(Delta_t, k=k)
            V_rel_t = compute_implicit_reliance_space(V_pre_K, V_t_k)
            deltas.append(Delta_t)
            rel_spaces.append(V_rel_t)
            del W_t, V_t_k

        # --- 步骤 4: 对每个任务做无干涉正交过滤, 然后算过滤后的 E_in ---
        row_e_in: List[float] = []
        row_in_energy: List[float] = []
        row_delta_sq: List[float] = []

        for j in range(n_tasks):
            # 保护空间 = 其余所有任务的隐式依赖空间并集
            protect = [
                rel_spaces[i] for i in range(n_tasks)
                if i != j and rel_spaces[i].shape[1] > 0
            ]
            Delta_j = deltas[j]
            if protect:
                V_protect = torch.cat(protect, dim=1)
                V_protect, _ = torch.linalg.qr(V_protect)
                Delta_f = orthogonal_core_filtering(Delta_j, V_protect)
                del V_protect
            else:
                Delta_f = Delta_j

            delta_sq = float(torch.linalg.norm(Delta_f, ord="fro") ** 2)
            if delta_sq < 1e-12:
                row_e_in.append(0.0)
                row_in_energy.append(0.0)
                row_delta_sq.append(delta_sq)
                continue

            projected = Delta_f @ V_pre_K
            in_energy = float(torch.linalg.norm(projected, ord="fro") ** 2)
            e_in = in_energy / max(delta_sq, 1e-30)

            row_e_in.append(e_in)
            row_in_energy.append(in_energy)
            row_delta_sq.append(delta_sq)
            del projected, Delta_f

        e_in_layers.append(row_e_in)
        in_energy_layers.append(row_in_energy)
        delta_sq_layers.append(row_delta_sq)
        layer_names.append(name)

        if verbose and len(layer_names) % 10 == 0:
            print(f"  processed {len(layer_names)} layers (last: {name})")

        del W_pre, V_pre_K, deltas, rel_spaces
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if not layer_names:
        raise RuntimeError("没有可分析的 2D 权重层。")

    per_layer = np.array(e_in_layers, dtype=np.float64)
    in_energy = np.array(in_energy_layers, dtype=np.float64)
    delta_sq = np.array(delta_sq_layers, dtype=np.float64)

    layer_mean = per_layer.mean(axis=0)
    energy_weighted = in_energy.sum(axis=0) / np.maximum(delta_sq.sum(axis=0), 1e-30)

    return {
        "per_layer": per_layer,
        "in_energy": in_energy,
        "delta_sq_norm": delta_sq,
        "layer_names": layer_names,
        "layer_mean": layer_mean,
        "energy_weighted": energy_weighted,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="PACT 过滤后任务向量的 E_in 分析 (只算 E_in 一个指标)。"
    )
    parser.add_argument("--model", type=str, default="ViT-B-16")
    parser.add_argument(
        "--model_location",
        type=str,
        default=str(REPO_ROOT / "models_complete" / "models" / "checkpoints"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "output_pact_filtered"),
    )
    parser.add_argument("--K", type=int, default=15,
                        help="θ₀ 核心基底维度 (top-K 右奇异向量, 默认 15)")
    parser.add_argument("--k", type=int, default=8,
                        help="任务显式基底维度 (top-k 右奇异向量, 默认 8)")
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--tasks", type=str, nargs="+", default=DATASETS_8)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isabs(args.model_location):
        args.model_location = str((REPO_ROOT / args.model_location).resolve())
    if not os.path.isabs(args.output_dir):
        args.output_dir = str(Path(args.output_dir).resolve())
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 72)
    print("PACT-filtered E_in analysis  (Δ_t^filtered energy ∈ V₀,K)")
    print(f"  Model       : {args.model}")
    print(f"  Tasks ({len(args.tasks)}): {args.tasks}")
    print(f"  K (pretrain)= {args.K},  k (task) = {args.k}")
    print(f"  Device      : {args.device}")
    print(f"  Output dir  : {args.output_dir}")
    print("=" * 72)

    pre_state, task_states = load_pretrained_and_task_states(
        args.model, args.model_location, args.tasks
    )

    res = compute_filtered_energy_alignment(
        pretrained_state=pre_state,
        task_states=task_states,
        task_order=args.tasks,
        K=args.K,
        k=args.k,
        device=args.device,
    )

    n_layers = len(res["layer_names"])
    print(f"\n[Done] averaged over {n_layers} 2D weight layers (PACT filtered)")
    print("\nPer-task summary (filtered E_in):")
    print(f"  {'task':<12} {'E_in (mean)':>14} {'E_in (weighted)':>18}")
    for t, m, w in zip(args.tasks, res["layer_mean"], res["energy_weighted"]):
        print(f"  {t:<12} {m:>14.4f} {w:>18.4f}")

    # --- 数值结果 ---
    np.save(os.path.join(args.output_dir, "per_layer_e_in.npy"),
            res["per_layer"])
    save_layer_matrix_csv(
        res["per_layer"], args.tasks, res["layer_names"],
        os.path.join(args.output_dir, "per_layer_e_in.csv"),
    )
    # fold_enrichment 不再计算, 用 0 占位以复用 save_summary_csv 签名。
    save_summary_csv(
        res["layer_mean"], res["energy_weighted"],
        np.zeros_like(res["layer_mean"]),
        args.tasks,
        os.path.join(args.output_dir, "per_task_summary.csv"),
    )

    # --- 图 (与原 E_in 实验完全一致的样式) ---
    plot_stacked_bar(
        res["layer_mean"], res["energy_weighted"], args.tasks,
        os.path.join(args.output_dir, "energy_alignment_stacked.png"),
    )
    plot_per_layer_violin(
        res["per_layer"], args.tasks,
        os.path.join(args.output_dir, "energy_alignment_violin.png"),
    )

    grouped = group_by_layer_type(res["layer_names"], res["per_layer"])
    plot_layer_depth_by_type(
        grouped, args.tasks,
        os.path.join(args.output_dir, "layer_depth_by_type.png"),
    )
    plot_layer_heatmap(
        res["layer_names"], args.tasks, res["per_layer"],
        os.path.join(args.output_dir, "layer_heatmap.png"),
    )
    plot_layer_mean_band(
        res["layer_names"], res["per_layer"],
        os.path.join(args.output_dir, "layer_mean_band.png"),
    )

    print(f"\nSaved (png + pdf) to: {args.output_dir}")


if __name__ == "__main__":
    main()
