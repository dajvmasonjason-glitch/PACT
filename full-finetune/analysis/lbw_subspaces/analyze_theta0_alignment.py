"""θ₀-alignment 分析 (PACT motivation 必做图 1)。

目标:
    验证 θ₀ 对每个任务并非"公平的起点"。不同任务把多少 Δ 能量投到 V₀,K
    (θ₀ 的 top-K 右奇异子空间) 是任务特异的，从而说明 θ₀ 的核心结构对
    不同任务的重要性不同 —— 这是 PACT 把 V₀,K 视作"承重墙基底"的前提。

数学定义 (对每一层 2D 权重 W ∈ R^{d_out × d_in}):
    Δ_t        = W_t - W_0
    V₀,K       = W_0 的 top-K 右奇异向量, d_in × K
    E_in(t,ℓ)  = ‖Δ_t · V₀,K‖²_F / ‖Δ_t‖²_F  ∈ [0, 1]
    E_out      = 1 - E_in

层级聚合:
    layer_mean       : 不加权的层平均 E_in(t,ℓ)
    energy_weighted  : sum_ℓ ‖Δ_t,ℓ V₀,K‖² / sum_ℓ ‖Δ_t,ℓ‖²
                       (任务 t 全部 Δ 能量中, 落在 θ₀ 核心子空间的比例)

补充指标:
    fold_enrichment(t,ℓ) = E_in(t,ℓ) / (K / d_in(ℓ))
                           对随机基线的倍数, 用于跨层比较

输出:
    outputs_alignment/
        per_layer_e_in.{csv, npy}
        per_layer_fold_enrichment.{csv, npy}
        per_task_summary.csv
        energy_alignment_stacked.{pdf, png}
        energy_alignment_violin.{pdf, png}

用法:
    cd <repo>
    python -m analysis.lbw_subspaces.analyze_theta0_alignment \
        --model ViT-B-16 --model_location models/ckpts --K 15
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.variables_and_paths import DATASETS_8  # noqa: E402

from analysis.lbw_subspaces.analyze_pact_motivation import (  # noqa: E402
    _is_2d_weight,
    extract_pretrained_core_basis,
    load_pretrained_and_task_states,
)


TensorDict = Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# 核心计算
# ---------------------------------------------------------------------------


def compute_energy_alignment(
    pretrained_state: Mapping[str, torch.Tensor],
    task_states: Mapping[str, Mapping[str, torch.Tensor]],
    task_order: List[str],
    K: int = 15,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """逐层计算 E_in(t,ℓ) 与配套量。

    Returns dict with:
        per_layer        : (L, T)  -- E_in(t, ℓ)
        in_energy        : (L, T)  -- ‖Δ_t,ℓ V₀,K‖²_F
        delta_sq_norm    : (L, T)  -- ‖Δ_t,ℓ‖²_F
        fold_enrichment  : (L, T)  -- E_in / (K_eff / d_in)
        layer_names      : list[str]   长度 L
        layer_d_in       : (L,)        每层的 d_in
        layer_K_eff      : (L,)        实际取到的 K (=min(K, d_out, d_in))
        layer_mean       : (T,)
        energy_weighted  : (T,)
        fold_mean        : (T,)        per-layer fold_enrichment 的均值
    """
    n_tasks = len(task_order)
    e_in_layers: List[List[float]] = []
    in_energy_layers: List[List[float]] = []
    delta_sq_layers: List[List[float]] = []
    fold_layers: List[List[float]] = []
    layer_names: List[str] = []
    layer_d_in: List[int] = []
    layer_K_eff: List[int] = []

    for name, W_pre_cpu in pretrained_state.items():
        if not _is_2d_weight(name, W_pre_cpu):
            continue
        if any(name not in task_states[t] for t in task_order):
            continue

        W_pre = W_pre_cpu.to(device=device, dtype=torch.float32)
        d_out, d_in = W_pre.shape
        V_pre_K = extract_pretrained_core_basis(W_pre, K=K)  # d_in × K_eff
        K_eff = V_pre_K.shape[1]
        random_baseline = K_eff / d_in if d_in > 0 else 0.0

        row_e_in: List[float] = []
        row_in_energy: List[float] = []
        row_delta_sq: List[float] = []
        row_fold: List[float] = []

        for t in task_order:
            W_t = task_states[t][name].to(device=device, dtype=torch.float32)
            Delta_t = W_t - W_pre
            delta_sq = float(torch.linalg.norm(Delta_t, ord="fro") ** 2)

            if delta_sq < 1e-12:
                row_e_in.append(0.0)
                row_in_energy.append(0.0)
                row_delta_sq.append(delta_sq)
                row_fold.append(0.0)
                continue

            projected = Delta_t @ V_pre_K           # d_out × K_eff
            in_energy = float(torch.linalg.norm(projected, ord="fro") ** 2)
            e_in = in_energy / max(delta_sq, 1e-30)
            fold = e_in / random_baseline if random_baseline > 0 else 0.0

            row_e_in.append(e_in)
            row_in_energy.append(in_energy)
            row_delta_sq.append(delta_sq)
            row_fold.append(fold)

            del W_t, Delta_t, projected

        e_in_layers.append(row_e_in)
        in_energy_layers.append(row_in_energy)
        delta_sq_layers.append(row_delta_sq)
        fold_layers.append(row_fold)
        layer_names.append(name)
        layer_d_in.append(int(d_in))
        layer_K_eff.append(int(K_eff))

        if verbose and len(layer_names) % 10 == 0:
            print(f"  processed {len(layer_names)} layers (last: {name})")

        del W_pre, V_pre_K
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if not layer_names:
        raise RuntimeError("没有可分析的 2D 权重层。")

    per_layer = np.array(e_in_layers, dtype=np.float64)
    in_energy = np.array(in_energy_layers, dtype=np.float64)
    delta_sq = np.array(delta_sq_layers, dtype=np.float64)
    fold_enrichment = np.array(fold_layers, dtype=np.float64)

    layer_mean = per_layer.mean(axis=0)
    energy_weighted = in_energy.sum(axis=0) / np.maximum(delta_sq.sum(axis=0), 1e-30)
    fold_mean = fold_enrichment.mean(axis=0)

    return {
        "per_layer": per_layer,
        "in_energy": in_energy,
        "delta_sq_norm": delta_sq,
        "fold_enrichment": fold_enrichment,
        "layer_names": layer_names,
        "layer_d_in": np.array(layer_d_in),
        "layer_K_eff": np.array(layer_K_eff),
        "layer_mean": layer_mean,
        "energy_weighted": energy_weighted,
        "fold_mean": fold_mean,
    }


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------


def plot_stacked_bar(
    layer_mean: np.ndarray,
    energy_weighted: np.ndarray,
    task_names: List[str],
    save_path: str,
) -> None:
    """两子图 stacked bar: 左=层平均, 右=能量加权; in/out V₀,K."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })

    n = len(task_names)
    x = np.arange(n)
    width = 0.62
    color_in = "#3B6FB6"
    color_out = "#E2A23B"

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)

    for ax, vec, title in (
        (axes[0], layer_mean,      r"Layer-mean  $E_{\mathrm{in}}(t)$"),
        (axes[1], energy_weighted, r"Energy-weighted  $E_{\mathrm{in}}(t)$"),
    ):
        e_in = vec
        e_out = 1.0 - vec
        ax.bar(x, e_in,  width, label=r"in $V_{0,K}$",  color=color_in)
        ax.bar(x, e_out, width, bottom=e_in,
               label=r"out $V_{0,K}$", color=color_out)

        for xi, v in zip(x, e_in):
            if v > 0.05:
                ax.text(xi, v / 2, f"{v:.3f}", ha="center", va="center",
                        fontsize=9, color="white", weight="bold")
            else:
                ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=8, color=color_in, weight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(task_names, rotation=45, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_title(title, pad=8, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"fraction of $\Delta_t$ Frobenius energy")
    axes[1].legend(loc="upper right", frameon=False, fontsize=9)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    fig.savefig(os.path.splitext(save_path)[0] + ".pdf")
    plt.close(fig)


def plot_per_layer_violin(
    per_layer: np.ndarray,
    task_names: List[str],
    save_path: str,
    ylabel: str = r"$E_{\mathrm{in}}(t,\ell)$  per layer",
    title: str = r"Per-layer $\Delta_t$ energy aligned with $V_{0,K}$",
) -> None:
    """8 任务的 per-layer 小提琴 + 散点."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })

    n_tasks = len(task_names)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    data = [per_layer[:, j] for j in range(n_tasks)]

    parts = ax.violinplot(
        data,
        positions=np.arange(n_tasks),
        showmeans=False,
        showmedians=True,
        showextrema=False,
        widths=0.8,
    )
    palette = sns.color_palette("rocket", n_tasks)
    for body, c in zip(parts["bodies"], palette):
        body.set_facecolor(c)
        body.set_edgecolor("none")
        body.set_alpha(0.75)
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.2)

    rng = np.random.RandomState(0)
    for j in range(n_tasks):
        jitter = (rng.rand(per_layer.shape[0]) - 0.5) * 0.18
        ax.scatter(
            np.full(per_layer.shape[0], j) + jitter,
            per_layer[:, j],
            s=7, color="black", alpha=0.28, linewidths=0,
        )

    ax.set_xticks(np.arange(n_tasks))
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    fig.savefig(os.path.splitext(save_path)[0] + ".pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def save_layer_matrix_csv(matrix, task_names, layer_names, out_path):
    header = "layer," + ",".join(task_names)
    rows = [
        f"{layer_names[i]}," + ",".join(f"{v:.6f}" for v in matrix[i])
        for i in range(matrix.shape[0])
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")


def save_summary_csv(
    layer_mean, energy_weighted, fold_mean, task_names, out_path
):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("task,E_in_layer_mean,E_in_energy_weighted,fold_enrichment_mean\n")
        for t, m, w, fm in zip(task_names, layer_mean, energy_weighted, fold_mean):
            f.write(f"{t},{m:.6f},{w:.6f},{fm:.6f}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="θ₀-alignment analysis: how much of each task's "
                    "Δ energy lives in the top-K subspace of θ₀."
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
        default=str(Path(__file__).resolve().parent / "outputs_alignment"),
    )
    parser.add_argument("--K", type=int, default=15,
                        help="top-K right singular vectors of θ₀ (default: 15)")
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=DATASETS_8,
        help="任务列表, 顺序决定柱图与小提琴的顺序",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isabs(args.model_location):
        args.model_location = str((REPO_ROOT / args.model_location).resolve())
    if not os.path.isabs(args.output_dir):
        args.output_dir = str(Path(args.output_dir).resolve())
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 72)
    print("θ₀-alignment analysis  (Δ_t energy ∈ V₀,K)")
    print(f"  Model       : {args.model}")
    print(f"  Tasks ({len(args.tasks)}): {args.tasks}")
    print(f"  K           : {args.K}")
    print(f"  Device      : {args.device}")
    print(f"  Output dir  : {args.output_dir}")
    print("=" * 72)

    pre_state, task_states = load_pretrained_and_task_states(
        args.model, args.model_location, args.tasks
    )

    res = compute_energy_alignment(
        pretrained_state=pre_state,
        task_states=task_states,
        task_order=args.tasks,
        K=args.K,
        device=args.device,
    )

    n_layers = len(res["layer_names"])
    avg_d_in = float(np.mean(res["layer_d_in"]))
    avg_baseline = float(np.mean(res["layer_K_eff"] / res["layer_d_in"]))

    print(f"\n[Done] averaged over {n_layers} 2D weight layers "
          f"(mean d_in={avg_d_in:.0f}, random baseline ≈ {avg_baseline:.4f})")
    print("\nPer-task summary:")
    print(f"  {'task':<12} {'E_in (mean)':>14} {'E_in (weighted)':>18} "
          f"{'fold-enrich':>14}")
    for t, m, w, fm in zip(args.tasks,
                            res["layer_mean"],
                            res["energy_weighted"],
                            res["fold_mean"]):
        print(f"  {t:<12} {m:>14.4f} {w:>18.4f} {fm:>14.2f}x")

    # --- 数值结果 ---
    np.save(os.path.join(args.output_dir, "per_layer_e_in.npy"),
            res["per_layer"])
    np.save(os.path.join(args.output_dir, "per_layer_fold_enrichment.npy"),
            res["fold_enrichment"])
    save_layer_matrix_csv(
        res["per_layer"], args.tasks, res["layer_names"],
        os.path.join(args.output_dir, "per_layer_e_in.csv"),
    )
    save_layer_matrix_csv(
        res["fold_enrichment"], args.tasks, res["layer_names"],
        os.path.join(args.output_dir, "per_layer_fold_enrichment.csv"),
    )
    save_summary_csv(
        res["layer_mean"], res["energy_weighted"], res["fold_mean"],
        args.tasks,
        os.path.join(args.output_dir, "per_task_summary.csv"),
    )

    # --- 图 ---
    plot_stacked_bar(
        res["layer_mean"], res["energy_weighted"], args.tasks,
        os.path.join(args.output_dir, "energy_alignment_stacked.png"),
    )
    plot_per_layer_violin(
        res["per_layer"], args.tasks,
        os.path.join(args.output_dir, "energy_alignment_violin.png"),
    )
    plot_per_layer_violin(
        res["fold_enrichment"], args.tasks,
        os.path.join(args.output_dir, "fold_enrichment_violin.png"),
        ylabel=r"fold-enrichment over random  ($E_{\mathrm{in}} / (K/d_{\mathrm{in}})$)",
        title=r"Per-layer fold-enrichment of $\Delta_t$ in $V_{0,K}$",
    )

    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
