"""PACT 算法动机的两个核心假设验证脚本。

实验目标:
    1) Sacred Space Similarity (神圣空间相似度矩阵 8x8, 对称)
       验证不同任务的"神圣空间"是否真的彼此差异化。
    2) Hidden Interference Ratio (隐蔽冲突比例矩阵 8x8, 非对称)
       验证一个任务的 Δ 在其他任务的神圣空间上"误伤"了多少能量。

数学定义 (针对每一层 2D 权重矩阵 W ∈ R^{d_out × d_in}):
    Δ_t          = W_t - W_pre
    V_pre^K      = 预训练核心基底 (W_pre 的 right-singular 前 K=38 个方向, d_in × K)
    V_t^k        = 任务显式基底  (Δ_t 的 right-singular 前 k=32 个方向, d_in × k)
    V_sacred^t   = Orth( (I - V_t^k V_t^{k,T}) V_pre^K )    # d_in × r_t

    Sim(A,B)     = || V_sacred^A^T V_sacred^B ||_F^2 / min(r_A, r_B)
    Interf(B→A) = || (V_B^k)^T V_sacred^A ||_F^2 / k          # 行=干扰方 B，列=被干扰方 A
                  纯几何量, 不依赖权重尺度: 任务 B 的显式更新基底与 A 的神圣空间的重合比例

实验设定:
    模型: ViT-B/16
    任务: MNIST, SVHN, GTSRB, EuroSAT, RESISC45, DTD, Cars, SUN397 (DATASETS_8)
    K (预训练核心维度) = 38, k (任务显式维度) = 32 (与 PACT 论文设定一致)

运行方式:
    cd <repo>
    python -m analysis.lbw_subspaces.analyze_pact_motivation \
        --model ViT-B-16 --model_location models/ckpts \
        --output_dir analysis/lbw_subspaces/outputs

输出:
    outputs/
        sacred_similarity.npy / .csv
        hidden_interference.npy / .csv
        sacred_similarity.pdf / .png
        hidden_interference.pdf / .png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch

# Make sure src.* is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.variables_and_paths import (  # noqa: E402
    DATASETS_8,
    get_finetuned_path,
    get_zeroshot_path,
)


TensorDict = Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# 数学辅助函数
# ---------------------------------------------------------------------------


def _right_singular_basis(matrix: torch.Tensor, q: int) -> torch.Tensor:
    """对 matrix ∈ R^{d_out × d_in} 取前 q 个右奇异向量, 返回 d_in × q 的列正交基。
    使用精确 SVD 确保正交投影的数学严谨性。
    """
    d_out, d_in = matrix.shape
    q_eff = max(1, min(q, d_out, d_in))

    # 使用精确 SVD (full_matrices=False 节省内存)
    # torch.linalg.svd 返回的是 Vh (即 V^T), 形状为 (min(d_out, d_in), d_in)
    _, _, Vh = torch.linalg.svd(matrix, full_matrices=False)

    # 截取前 q_eff 个奇异方向，并转置回 (d_in, q_eff) 的列基底形式
    V = Vh[:q_eff, :].t()
    return V


def extract_pretrained_core_basis(W_pre: torch.Tensor, K: int) -> torch.Tensor:
    """V_pre^K ∈ R^{d_in × K}。"""
    return _right_singular_basis(W_pre, K)


def extract_task_explicit_basis(Delta_t: torch.Tensor, k: int) -> torch.Tensor:
    """V_t^k ∈ R^{d_in × k}。"""
    return _right_singular_basis(Delta_t, k)


def compute_sacred_basis(V_pre_K: torch.Tensor, V_t_k: torch.Tensor) -> torch.Tensor:
    """V_sacred^t = Orth( (I - V_t^k V_t^{k,T}) V_pre^K )。

    返回 d_in × r 的列正交基, 当残差近零时退化为零列矩阵。
    """
    # 避免显式 d_in × d_in 投影矩阵: residual = V_pre - V_t (V_t^T V_pre)
    residual = V_pre_K - V_t_k @ (V_t_k.t() @ V_pre_K)
    norm = torch.linalg.norm(residual, ord="fro")
    if norm < 1e-8:
        return torch.zeros(
            (V_pre_K.shape[0], 0), dtype=V_pre_K.dtype, device=V_pre_K.device
        )
    # QR 拿到列正交基; rank-deficient 列由 R 的对角小值过滤。
    Q, R = torch.linalg.qr(residual, mode="reduced")
    diag = torch.abs(torch.diagonal(R))
    if diag.numel() == 0:
        return Q
    tol = diag.max() * max(R.shape) * torch.finfo(R.dtype).eps
    keep = diag > tol
    return Q[:, keep]


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------


def _load_state_dict(path: str) -> TensorDict:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_name" in state:
        state = dict(state)
        state.pop("model_name")
    return state


def _val_name(task: str) -> str:
    return task if task.endswith("Val") else f"{task}Val"


def load_pretrained_and_task_states(
    model: str, model_location: str, tasks: List[str]
) -> Tuple[TensorDict, Dict[str, TensorDict]]:
    """加载预训练 state_dict 与每个任务微调后的 state_dict。"""
    pretrained_path = get_zeroshot_path(model_location, "MNISTVal", model=model)
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"预训练 ckpt 不存在: {pretrained_path}")
    pre_state = _load_state_dict(pretrained_path)

    task_states: Dict[str, TensorDict] = {}
    for task in tasks:
        ft_path = get_finetuned_path(model_location, _val_name(task), model=model)
        if not os.path.exists(ft_path):
            raise FileNotFoundError(f"任务 {task} 的微调 ckpt 不存在: {ft_path}")
        task_states[task] = _load_state_dict(ft_path)
    return pre_state, task_states


def _is_2d_weight(name: str, tensor: torch.Tensor) -> bool:
    """筛选可分析的 2D 权重: 排除 bias / LayerNorm / 分类头 / text_projection 等 1D 或非通用层。"""
    if not torch.is_floating_point(tensor):
        return False
    if tensor.dim() != 2:
        return False
    lname = name.lower()
    if "text_projection" in lname:
        return False
    if "norm" in lname or "ln_" in lname:
        return False
    if "logit_scale" in lname:
        return False
    if "head" in lname or "classification_head" in lname:
        return False
    return True


# ---------------------------------------------------------------------------
# 主分析: 逐层累加 → 全局平均
# ---------------------------------------------------------------------------


def analyze_pact_motivation(
    pretrained_state: Mapping[str, torch.Tensor],
    task_states: Mapping[str, Mapping[str, torch.Tensor]],
    task_order: List[str],
    K: int = 15,
    k: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """逐层计算 Sim 与 Interf, 全局平均。

    Returns:
        sim_matrix: 8x8 numpy 数组, 对称, 主对角线为 1。
        interf_matrix: 8x8 numpy 数组, 非对称, 行=干扰方 (B), 列=被干扰方 (A)。
        n_layers: 参与平均的有效层数。
    """
    n_tasks = len(task_order)
    sim_accum = np.zeros((n_tasks, n_tasks), dtype=np.float64)
    interf_accum = np.zeros((n_tasks, n_tasks), dtype=np.float64)
    n_layers = 0

    # 以预训练 state_dict 的 key 顺序遍历, 保证逐层对齐。
    for name, W_pre_cpu in pretrained_state.items():
        if not _is_2d_weight(name, W_pre_cpu):
            continue

        # 跳过任意一个任务缺失该参数的情况。
        if any(name not in task_states[t] for t in task_order):
            continue

        W_pre = W_pre_cpu.to(device=device, dtype=torch.float32)

        # 计算每个任务的 V_t^k 与 V_sacred^t (纯几何量, 不再需要 Δ_t 与其范数)。
        task_bases: List[torch.Tensor] = []   # V_t^k
        sacreds: List[torch.Tensor] = []      # V_sacred^t

        V_pre_K = extract_pretrained_core_basis(W_pre, K=K)

        for t in task_order:
            W_t = task_states[t][name].to(device=device, dtype=torch.float32)
            Delta_t = W_t - W_pre
            V_t_k = extract_task_explicit_basis(Delta_t, k=k)
            V_sacred_t = compute_sacred_basis(V_pre_K, V_t_k)
            task_bases.append(V_t_k)
            sacreds.append(V_sacred_t)
            del W_t, Delta_t

        # Sim(A,B)     = || V_sacred^A^T V_sacred^B ||_F^2 / min(r_A, r_B)
        # Interf(B→A)  = || (V_B^k)^T V_sacred^A ||_F^2 / k
        for i in range(n_tasks):
            V_a = sacreds[i]
            r_a = V_a.shape[1]
            for j in range(n_tasks):
                V_b = sacreds[j]
                r_b = V_b.shape[1]

                # Sacred-space 相似度 (对称)
                if r_a == 0 or r_b == 0:
                    sim_val = 0.0
                else:
                    overlap = V_a.t() @ V_b  # r_a × r_b
                    sim_val = float(torch.linalg.norm(overlap, ord="fro") ** 2)
                    sim_val /= float(min(r_a, r_b))
                sim_accum[i, j] += sim_val

                # 隐蔽冲突 (几何投影版本): 行 i = 干扰方 B, 列 j = 被干扰方 A.
                # 度量 B 的显式更新基底 V_B^k 中, 有多大比例落入 A 的神圣空间 V_sacred^A.
                V_B_k = task_bases[i]
                V_sacred_A = sacreds[j]
                k_b = V_B_k.shape[1]
                if V_sacred_A.shape[1] == 0 or k_b == 0:
                    interf_val = 0.0
                else:
                    proj = V_B_k.t() @ V_sacred_A  # k_b × r_A
                    interf_val = float(torch.linalg.norm(proj, ord="fro") ** 2) / float(k_b)
                interf_accum[i, j] += interf_val

        n_layers += 1
        if verbose and n_layers % 10 == 0:
            print(f"  已处理 {n_layers} 个 2D 权重层 (最近一层: {name})")

        # 释放本层 GPU 显存
        del W_pre, V_pre_K, task_bases, sacreds
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if n_layers == 0:
        raise RuntimeError("没有找到任何可分析的 2D 权重层。")

    sim_matrix = sim_accum / n_layers
    interf_matrix = interf_accum / n_layers
    return sim_matrix, interf_matrix, n_layers


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------


def plot_heatmap(
    matrix: np.ndarray,
    task_names: List[str],
    title: str,
    save_path: str,
    cmap_name: str = "rocket",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Tufte 极简风学术热力图。

    - rocket / 深-浅渐变色;
    - 数值居中两位小数标注, annot 颜色随背景自动反转 (seaborn 默认行为);
    - X 轴标签旋转 45°, Y 轴标签放右侧水平显示;
    - 去除外框, 仅保留必要刻度.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.0,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )

    cmap = sns.color_palette(cmap_name, as_cmap=True)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9, "weight": "bold"},
        xticklabels=task_names,
        yticklabels=task_names,
        cbar=True,
        cbar_kws={"shrink": 0.78, "pad": 0.02},
        linewidths=0.6,
        linecolor="white",
        square=True,
    )

    # 标签与轴样式
    ax.set_title(title, pad=14, fontweight="bold")
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_yticklabels(task_names, rotation=0)

    # 去除外框 (Tufte 极简风)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)

    # colorbar 边框也去掉
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=0)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    fig.savefig(os.path.splitext(save_path)[0] + ".pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 结果保存
# ---------------------------------------------------------------------------


def save_matrix(matrix: np.ndarray, task_names: List[str], out_path_no_ext: str) -> None:
    """同名保存 .npy 与 .csv。"""
    np.save(out_path_no_ext + ".npy", matrix)
    header = "," + ",".join(task_names)
    rows = [
        task_names[i] + "," + ",".join(f"{v:.6f}" for v in matrix[i])
        for i in range(matrix.shape[0])
    ]
    with open(out_path_no_ext + ".csv", "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PACT 动机假设验证脚本")
    parser.add_argument("--model", type=str, default="ViT-B-16")
    parser.add_argument(
        "--model_location",
        type=str,
        default=str(REPO_ROOT / "models_complete" / "models" / "checkpoints"),
        help="ckpt 根目录, 相对路径会按 REPO_ROOT 解析",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs"),
    )
    parser.add_argument("--K", type=int, default=15, help="预训练核心基底维度")
    parser.add_argument("--k", type=int, default=8, help="任务显式基底维度")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=DATASETS_8,
        help="任务名列表, 顺序决定矩阵的行列顺序",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 相对路径按仓库根目录解析, 避免在子目录中运行时找不到 ckpt
    if not os.path.isabs(args.model_location):
        args.model_location = str((REPO_ROOT / args.model_location).resolve())
    if not os.path.isabs(args.output_dir):
        args.output_dir = str(Path(args.output_dir).resolve())
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("PACT Motivation Verification")
    print(f"  Model       : {args.model}")
    print(f"  Tasks ({len(args.tasks)}): {args.tasks}")
    print(f"  K (pretrain)= {args.K},  k (task) = {args.k}")
    print(f"  Device      : {args.device}")
    print(f"  Output dir  : {args.output_dir}")
    print("=" * 70)

    pre_state, task_states = load_pretrained_and_task_states(
        args.model, args.model_location, args.tasks
    )

    sim_matrix, interf_matrix, n_layers = analyze_pact_motivation(
        pretrained_state=pre_state,
        task_states=task_states,
        task_order=args.tasks,
        K=args.K,
        k=args.k,
        device=args.device,
    )
    print(f"\n[Done] 平均自 {n_layers} 个 2D 权重层。")

    sim_path = os.path.join(args.output_dir, "sacred_similarity")
    interf_path = os.path.join(args.output_dir, "hidden_interference")
    save_matrix(sim_matrix, args.tasks, sim_path)
    save_matrix(interf_matrix, args.tasks, interf_path)
    print(f"  Saved: {sim_path}.npy / .csv")
    print(f"  Saved: {interf_path}.npy / .csv")

    plot_heatmap(
        sim_matrix,
        args.tasks,
        title="Sacred Space Similarity (avg. over layers)",
        save_path=sim_path + ".png",
        vmin=0.0,
        vmax=1.0,
    )
    plot_heatmap(
        interf_matrix,
        args.tasks,
        title=r"Hidden Interference Ratio  (row $B$ → col $A$)",
        save_path=interf_path + ".png",
        vmin=0.0,
        vmax=float(np.max(interf_matrix)),
    )
    print(f"  Saved: {sim_path}.png / .pdf")
    print(f"  Saved: {interf_path}.png / .pdf")


if __name__ == "__main__":
    main()
