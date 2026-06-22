"""V₀,K 内部 per-axis 分布分析: 拆开 15 个主轴, 看任务方向画像.

动机:
    layer-mean E_in 只是标量聚合, 掩盖了 V₀,K 内部 15 个方向的分布.
    两个任务可能 E_in 都是 6%, 但一个集中在 V₀,K 的第 1/3/5 主轴,
    另一个集中在第 2/7/12 主轴 —— 几何上完全不重叠. 这才是
    "θ₀ 对每个任务不公平"的真正来源.

度量 (对每一层 W ∈ R^{d_out × d_in} 与 V₀,K = [v_1,...,v_K]):
    Δ_t                     = W_t - W_0
    a_i(t, ℓ)               = ‖Δ_t v_i‖² / ‖Δ_t‖²         绝对能量占比
    E_in(t, ℓ)              = sum_i a_i(t, ℓ)
    p_i(t, ℓ)               = a_i(t, ℓ) / E_in(t, ℓ)       V₀,K 内归一化分布
    profile(t, ℓ)           = (p_1, p_2, ..., p_K)        K 维任务方向画像

任务间不公平量化:
    cosine(t_a, t_b, ℓ)     = <profile(t_a, ℓ), profile(t_b, ℓ)>
                              / (||profile(t_a)|| · ||profile(t_b)||)
    js_div(t_a, t_b, ℓ)     = Jensen-Shannon divergence of profiles
                              (both are distributions summing to 1)
    top_task_per_axis(i, ℓ) = argmax_t a_i(t, ℓ)
                              如果跨方向分散在多个任务 → 不公平成立

聚焦层:
    脚本默认聚焦 hot layers (E_in 跨任务均值最高的若干层),
    因为后期 cold layers 的 E_in < 0.02, 方向分布全是噪声.

输出:
    outputs_alignment/
        per_axis_a.npy           shape (L, T, K)        a_i(t, ℓ)
        per_axis_p.npy           shape (L, T, K)        p_i(t, ℓ)
        axis_profile_hot.csv     hot layer 的归一化分布
        task_cosine_hot.csv      hot layer 上 8x8 cosine 相似度
        direction_ownership.csv  每个 hot layer × 每个方向: top-task
        axis_profile_heatmap.{png,pdf}     hot layers (T × K) 热力图
        task_cosine_heatmap.{png,pdf}      hot layer 平均的 8x8 任务相似度
        direction_ownership_bar.{png,pdf}  谁占有多少方向 (按任务)

用法:
    python -m analysis.lbw_subspaces.analyze_axis_profile \
        --model ViT-B-16 --model_location models/ckpts --K 15 --top_layers 6
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

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


# ---------------------------------------------------------------------------
# 核心计算
# ---------------------------------------------------------------------------


def compute_per_axis_distribution(
    pretrained_state: Mapping[str, torch.Tensor],
    task_states: Mapping[str, Mapping[str, torch.Tensor]],
    task_order: List[str],
    K: int = 15,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Returns:
        per_axis_a       : (L, T, K)  a_i(t, ℓ) = ‖Δ_t v_i‖² / ‖Δ_t‖²
        per_axis_p       : (L, T, K)  p_i(t, ℓ) = a_i / E_in (V₀,K 内归一化)
        e_in             : (L, T)     E_in(t, ℓ) = sum_i a_i
        layer_names      : list[str]
        layer_K_eff      : (L,)
    """
    n_tasks = len(task_order)
    layer_names: List[str] = []
    layer_K_eff: List[int] = []
    a_layers: List[np.ndarray] = []   # 每层一个 (T, K_eff) 数组
    e_in_layers: List[List[float]] = []

    for name, W_pre_cpu in pretrained_state.items():
        if not _is_2d_weight(name, W_pre_cpu):
            continue
        if any(name not in task_states[t] for t in task_order):
            continue

        W_pre = W_pre_cpu.to(device=device, dtype=torch.float32)
        V_pre_K = extract_pretrained_core_basis(W_pre, K=K)   # (d_in, K_eff)
        K_eff = V_pre_K.shape[1]

        per_axis_row = np.zeros((n_tasks, K_eff), dtype=np.float64)
        e_in_row: List[float] = []

        for t_idx, t in enumerate(task_order):
            W_t = task_states[t][name].to(device=device, dtype=torch.float32)
            Delta_t = W_t - W_pre
            delta_sq = float(torch.linalg.norm(Delta_t, ord="fro") ** 2)
            if delta_sq < 1e-12:
                e_in_row.append(0.0)
                del W_t, Delta_t
                continue

            projected = Delta_t @ V_pre_K                     # (d_out, K_eff)
            axis_energy = (projected ** 2).sum(dim=0)         # (K_eff,)
            axis_ratio = (axis_energy / delta_sq).detach().cpu().numpy()
            per_axis_row[t_idx, :] = axis_ratio
            e_in_row.append(float(axis_ratio.sum()))

            del W_t, Delta_t, projected, axis_energy

        a_layers.append(per_axis_row)
        e_in_layers.append(e_in_row)
        layer_names.append(name)
        layer_K_eff.append(int(K_eff))

        if verbose and len(layer_names) % 10 == 0:
            print(f"  processed {len(layer_names)} layers (last: {name})")

        del W_pre, V_pre_K
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if not layer_names:
        raise RuntimeError("没有可分析的 2D 权重层.")

    # 不同层可能 K_eff 不同 (取 min(K, d_out, d_in)). 这里 ViT-B-16 主要是 768
    # 维 + LayerNorm/proj, 但 visual.proj 可能只有 512. 我们填 NaN 到 max K.
    K_max = max(layer_K_eff)
    per_axis_a = np.full((len(layer_names), n_tasks, K_max), np.nan, dtype=np.float64)
    for i, arr in enumerate(a_layers):
        per_axis_a[i, :, : arr.shape[1]] = arr

    e_in = np.array(e_in_layers, dtype=np.float64)            # (L, T)
    e_in_safe = np.where(e_in > 1e-12, e_in, np.nan)
    per_axis_p = per_axis_a / e_in_safe[:, :, None]

    return {
        "per_axis_a": per_axis_a,                  # (L, T, K)
        "per_axis_p": per_axis_p,                  # (L, T, K)
        "e_in": e_in,                              # (L, T)
        "layer_names": layer_names,
        "layer_K_eff": np.array(layer_K_eff),
    }


def select_hot_layers(
    e_in: np.ndarray, layer_names: List[str], top_n: int = 8
) -> List[int]:
    """选 E_in 跨任务均值最高的 top_n 层."""
    layer_mean = e_in.mean(axis=1)
    order = np.argsort(layer_mean)[::-1]
    return order[:top_n].tolist()


def task_cosine_per_layer(per_axis_p: np.ndarray, layer_idx: int) -> np.ndarray:
    """单层上 8 任务画像两两 cosine. profile 已是 sum=1 的分布."""
    profile = per_axis_p[layer_idx]                # (T, K)
    valid = ~np.isnan(profile).any(axis=1)
    out = np.full((profile.shape[0], profile.shape[0]), np.nan, dtype=np.float64)
    if valid.sum() < 2:
        return out
    p = profile[valid]
    norm = np.linalg.norm(p, axis=1, keepdims=True)
    norm = np.where(norm > 1e-12, norm, 1.0)
    p_n = p / norm
    sim = p_n @ p_n.T
    out_idx = np.where(valid)[0]
    for i, ii in enumerate(out_idx):
        for j, jj in enumerate(out_idx):
            out[ii, jj] = sim[i, j]
    return out


def js_divergence_per_layer(per_axis_p: np.ndarray, layer_idx: int) -> np.ndarray:
    """单层上 8 任务画像两两 Jensen-Shannon divergence (base-2, range [0,1])."""
    profile = per_axis_p[layer_idx]                # (T, K)
    T = profile.shape[0]
    out = np.full((T, T), np.nan, dtype=np.float64)
    valid_mask = ~np.isnan(profile).any(axis=1)
    if valid_mask.sum() < 2:
        return out

    def safe_log2(x):
        return np.where(x > 0, np.log2(np.where(x > 0, x, 1.0)), 0.0)

    for i in range(T):
        if not valid_mask[i]:
            continue
        for j in range(T):
            if not valid_mask[j]:
                continue
            p_i = profile[i]
            p_j = profile[j]
            m = 0.5 * (p_i + p_j)
            kl_im = np.nansum(p_i * (safe_log2(p_i) - safe_log2(m)))
            kl_jm = np.nansum(p_j * (safe_log2(p_j) - safe_log2(m)))
            out[i, j] = 0.5 * (kl_im + kl_jm)
    return out


def direction_ownership(
    per_axis_a: np.ndarray, layer_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """对单层每个方向 i, 返回 (top_task_idx[K], top_ratio[K]).

    top_ratio = 第一名能量 / (第二名能量), 越大说明该方向被该任务"独占"得越明显.
    """
    a = per_axis_a[layer_idx]                       # (T, K)
    K = a.shape[1]
    top_task = np.full(K, -1, dtype=np.int64)
    top_gap = np.full(K, np.nan, dtype=np.float64)
    for i in range(K):
        col = a[:, i]
        if np.isnan(col).all():
            continue
        col_clean = np.where(np.isnan(col), -np.inf, col)
        order = np.argsort(col_clean)[::-1]
        if col_clean[order[0]] <= 0:
            continue
        top_task[i] = int(order[0])
        second = col_clean[order[1]] if len(order) > 1 and col_clean[order[1]] > 0 else 1e-12
        top_gap[i] = float(col_clean[order[0]] / second)
    return top_task, top_gap


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------


def _short_layer_name(name: str) -> str:
    import re
    m = re.search(r"resblocks\.(\d+)\.(.+)$", name)
    if m:
        b = int(m.group(1))
        suffix = m.group(2)
        mapping = {
            "attn.in_proj_weight": "attn.in_proj",
            "attn.out_proj.weight": "attn.out_proj",
            "mlp.c_fc.weight": "mlp.c_fc",
            "mlp.c_proj.weight": "mlp.c_proj",
        }
        return f"b{b:02d}.{mapping.get(suffix, suffix)}"
    return name.replace("model.visual.", "").replace("model.", "")


def plot_axis_profile_heatmap(
    per_axis_p: np.ndarray,
    layer_names: List[str],
    task_names: List[str],
    hot_indices: List[int],
    save_path: str,
) -> None:
    """每个 hot layer 一张 T × K 热力图. 显示 V₀,K 内归一化分布 p_i(t, ℓ)."""
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

    n_hot = len(hot_indices)
    n_cols = min(3, n_hot)
    n_rows = int(np.ceil(n_hot / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 2.7 * n_rows),
                              squeeze=False)
    cmap = sns.color_palette("rocket_r", as_cmap=True)

    for plot_idx, layer_idx in enumerate(hot_indices):
        r, c = divmod(plot_idx, n_cols)
        ax = axes[r][c]
        profile = per_axis_p[layer_idx]              # (T, K)
        K_eff = (~np.isnan(profile[0])).sum()
        profile = profile[:, :K_eff]
        sns.heatmap(
            profile, ax=ax,
            cmap=cmap, vmin=0.0,
            cbar=True, cbar_kws={"shrink": 0.7, "pad": 0.02},
            xticklabels=[f"v{i+1}" for i in range(K_eff)],
            yticklabels=task_names,
            linewidths=0.3, linecolor="white",
        )
        ax.set_title(_short_layer_name(layer_names[layer_idx]),
                     fontweight="bold")
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

    fig.suptitle(r"V$_{0,K}$-internal distribution $p_i(t,\ell)$  "
                 r"(rows=task, cols=axis of V$_{0,K}$)",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    fig.savefig(os.path.splitext(save_path)[0] + ".pdf")
    plt.close(fig)


def plot_task_cosine_heatmap(
    per_axis_p: np.ndarray,
    layer_names: List[str],
    task_names: List[str],
    hot_indices: List[int],
    save_path: str,
) -> None:
    """对 hot layers 上的 8x8 cosine 取平均, 画一张代表图."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })

    sims = []
    for li in hot_indices:
        s = task_cosine_per_layer(per_axis_p, li)
        if not np.isnan(s).all():
            sims.append(s)
    avg_sim = np.nanmean(np.stack(sims, axis=0), axis=0)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    sns.heatmap(
        avg_sim, ax=ax, cmap=cmap, vmin=0.0, vmax=1.0,
        annot=True, fmt=".2f", annot_kws={"size": 9, "weight": "bold"},
        xticklabels=task_names, yticklabels=task_names,
        cbar=True, cbar_kws={"shrink": 0.78, "pad": 0.02},
        linewidths=0.6, linecolor="white", square=True,
    )
    ax.set_title(
        f"Task profile cosine on V$_{{0,K}}$  "
        f"(avg over {len(sims)} hot layers)  —  lower = less fair",
        fontweight="bold", pad=10,
    )
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.set_yticklabels(task_names, rotation=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=0)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    fig.savefig(os.path.splitext(save_path)[0] + ".pdf")
    plt.close(fig)


def plot_direction_ownership_bar(
    per_axis_a: np.ndarray,
    layer_names: List[str],
    task_names: List[str],
    hot_indices: List[int],
    save_path: str,
) -> None:
    """跨 hot layers 统计: 有多少个 (layer × axis) 单元的 top-task 是任务 t."""
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
    counts = np.zeros(n_tasks, dtype=np.int64)
    total_dirs = 0
    for li in hot_indices:
        top_task, _ = direction_ownership(per_axis_a, li)
        for tt in top_task:
            if tt >= 0:
                counts[tt] += 1
                total_dirs += 1

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    palette = sns.color_palette("tab10", n_tasks)
    bars = ax.bar(np.arange(n_tasks), counts, color=palette, width=0.65)
    for b, c in zip(bars, counts):
        pct = 100.0 * c / max(1, total_dirs)
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{c}\n({pct:.0f}%)", ha="center", va="bottom",
                fontsize=9)
    ax.set_xticks(np.arange(n_tasks))
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.set_ylabel(r"# of (layer × axis) cells where task is the top owner")
    expected = total_dirs / n_tasks
    ax.axhline(expected, color="gray", linestyle="--", linewidth=1.0,
               label=f"uniform baseline = {expected:.1f}")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(
        f"Direction ownership on V$_{{0,K}}$  "
        f"({len(hot_indices)} hot layers × K dirs = {total_dirs} cells)",
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    fig.savefig(os.path.splitext(save_path)[0] + ".pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def save_hot_profiles_csv(
    per_axis_p, layer_names, task_names, hot_indices, out_path
):
    K_max = per_axis_p.shape[2]
    header = "layer,task," + ",".join(f"v{i+1}" for i in range(K_max))
    rows = []
    for li in hot_indices:
        short = _short_layer_name(layer_names[li])
        for t_idx, t in enumerate(task_names):
            vals = per_axis_p[li, t_idx]
            rows.append(short + "," + t + "," +
                        ",".join(f"{v:.6f}" if not np.isnan(v) else ""
                                 for v in vals))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")


def save_cosine_csv(per_axis_p, layer_names, task_names, hot_indices, out_path):
    sims = []
    for li in hot_indices:
        s = task_cosine_per_layer(per_axis_p, li)
        if not np.isnan(s).all():
            sims.append(s)
    avg_sim = np.nanmean(np.stack(sims, axis=0), axis=0)
    header = "," + ",".join(task_names)
    rows = [
        task_names[i] + "," + ",".join(f"{v:.6f}" for v in avg_sim[i])
        for i in range(avg_sim.shape[0])
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")


def save_ownership_csv(
    per_axis_a, layer_names, task_names, hot_indices, out_path
):
    rows = ["layer,axis_index,top_task,top1/top2_ratio,top_energy_ratio"]
    for li in hot_indices:
        top_task, top_gap = direction_ownership(per_axis_a, li)
        short = _short_layer_name(layer_names[li])
        for i, (tt, gap) in enumerate(zip(top_task, top_gap)):
            if tt < 0 or np.isnan(gap):
                continue
            top_energy = per_axis_a[li, tt, i]
            rows.append(f"{short},v{i+1},{task_names[tt]},"
                        f"{gap:.4f},{top_energy:.6f}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="V₀,K per-axis profile analysis"
    )
    parser.add_argument("--model", type=str, default="ViT-B-16")
    parser.add_argument(
        "--model_location", type=str,
        default=str(REPO_ROOT / "models_complete" / "models" / "checkpoints"),
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(Path(__file__).resolve().parent / "ouput_alignment"),
    )
    parser.add_argument("--K", type=int, default=15)
    parser.add_argument(
        "--top_layers", type=int, default=6,
        help="选 E_in 最高的多少层做 per-axis 分析 (default: 6)",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=DATASETS_8,
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
    print("V₀,K per-axis profile analysis")
    print(f"  Model       : {args.model}")
    print(f"  Tasks ({len(args.tasks)}): {args.tasks}")
    print(f"  K           : {args.K}")
    print(f"  top_layers  : {args.top_layers}")
    print(f"  Device      : {args.device}")
    print(f"  Output dir  : {args.output_dir}")
    print("=" * 72)

    pre_state, task_states = load_pretrained_and_task_states(
        args.model, args.model_location, args.tasks
    )

    res = compute_per_axis_distribution(
        pretrained_state=pre_state,
        task_states=task_states,
        task_order=args.tasks,
        K=args.K,
        device=args.device,
    )
    per_axis_a = res["per_axis_a"]
    per_axis_p = res["per_axis_p"]
    e_in = res["e_in"]
    layer_names = res["layer_names"]

    hot_indices = select_hot_layers(e_in, layer_names, top_n=args.top_layers)
    print("\n[Hot layers selected, sorted by mean E_in]")
    for li in hot_indices:
        print(f"  {_short_layer_name(layer_names[li]):<22} "
              f"mean E_in = {e_in[li].mean():.4f}")

    np.save(os.path.join(args.output_dir, "per_axis_a.npy"), per_axis_a)
    np.save(os.path.join(args.output_dir, "per_axis_p.npy"), per_axis_p)
    save_hot_profiles_csv(
        per_axis_p, layer_names, args.tasks, hot_indices,
        os.path.join(args.output_dir, "axis_profile_hot.csv"),
    )
    save_cosine_csv(
        per_axis_p, layer_names, args.tasks, hot_indices,
        os.path.join(args.output_dir, "task_cosine_hot.csv"),
    )
    save_ownership_csv(
        per_axis_a, layer_names, args.tasks, hot_indices,
        os.path.join(args.output_dir, "direction_ownership.csv"),
    )

    # 控制台打印 cosine summary
    sims = [task_cosine_per_layer(per_axis_p, li) for li in hot_indices]
    sims = [s for s in sims if not np.isnan(s).all()]
    avg_sim = np.nanmean(np.stack(sims, axis=0), axis=0)
    off_diag = avg_sim[~np.eye(avg_sim.shape[0], dtype=bool)]
    print(f"\n[Task profile cosine] avg over {len(sims)} hot layers")
    print(f"  off-diagonal mean : {np.nanmean(off_diag):.4f}")
    print(f"  off-diagonal min  : {np.nanmin(off_diag):.4f}")
    print(f"  off-diagonal max  : {np.nanmax(off_diag):.4f}")

    # 方向占有统计
    counts = np.zeros(len(args.tasks), dtype=int)
    total = 0
    for li in hot_indices:
        top_task, _ = direction_ownership(per_axis_a, li)
        for tt in top_task:
            if tt >= 0:
                counts[tt] += 1
                total += 1
    print(f"\n[Direction ownership across {len(hot_indices)} hot layers × K dirs]")
    print(f"  total cells   : {total}")
    print(f"  uniform expect: {total/len(args.tasks):.2f} per task")
    for t, c in zip(args.tasks, counts):
        pct = 100.0 * c / max(1, total)
        print(f"    {t:<12} {c:>4}  ({pct:5.1f}%)")

    plot_axis_profile_heatmap(
        per_axis_p, layer_names, args.tasks, hot_indices,
        os.path.join(args.output_dir, "axis_profile_heatmap.png"),
    )
    plot_task_cosine_heatmap(
        per_axis_p, layer_names, args.tasks, hot_indices,
        os.path.join(args.output_dir, "task_cosine_heatmap.png"),
    )
    plot_direction_ownership_bar(
        per_axis_a, layer_names, args.tasks, hot_indices,
        os.path.join(args.output_dir, "direction_ownership_bar.png"),
    )

    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
