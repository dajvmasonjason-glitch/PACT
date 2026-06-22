"""Subspace Ablation Evaluation — 子空间消融与降解实验.

验证预训练模型"承重墙空间" V_{t,pact} 的真实物理效应：
给定一个微调任务 A，直接从 checkpoint 计算 Δ_A = θ_A - θ_0，
然后沿 V_{A,pact} 投影消除 W_A = W_0 + Δ_A 中该子空间的分量。
因果预期：消融 V_{pact} → 性能崩溃；消融等维度随机子空间 → 几乎不受影响。

该脚本只依赖单个微调任务 + 预训练模型，完全不需要任务对。

运行方式
--------
    python -m analysis.subspace_ablation_eval \
        --task Cars \
        --model-location models_complete/models/checkpoints \
        --data-location datasets \
        --device cuda
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


# ===========================================================================
# 第一部分：核心数学 —— SVD / QR / 子空间运算（完全自包含）
# ===========================================================================

def right_singular_basis(matrix: torch.Tensor, q: int) -> torch.Tensor:
    """matrix ∈ R^{d_out × d_in} 的前 q 个右奇异向量, 返回 d_in × q 列正交基."""
    d_out, d_in = matrix.shape
    q_eff = max(1, min(q, d_out, d_in))
    _, _, Vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    return Vh[:q_eff, :].t().contiguous()


def pact_subspace_basis(V_0K: torch.Tensor, V_tk: torch.Tensor) -> torch.Tensor:
    """V_{t,pact} = orth((I - V_{t,k} V_{t,k}^⊤) V_{0,K}), 返回 d_in × d_pact 列正交基."""
    residual = V_0K - V_tk @ (V_tk.t() @ V_0K)
    if torch.linalg.norm(residual, ord="fro") < 1e-8:
        return torch.zeros((V_0K.shape[0], 0), dtype=V_0K.dtype, device=V_0K.device)
    Q, R = torch.linalg.qr(residual.float(), mode="reduced")
    diag = torch.abs(torch.diagonal(R))
    if diag.numel() == 0:
        return Q
    tol = diag.max() * max(R.shape) * torch.finfo(R.dtype).eps
    return Q[:, diag > tol]


def random_orthogonal_in_perp(V_0K: torch.Tensor, d_pact: int, seed: int = 0) -> torch.Tensor:
    """在 V_{0,K}^⊥ 空间中随机生成 d_pact 维正交基 V_rand ∈ R^{d_in × d_pact}."""
    d_in = V_0K.shape[0]
    device = V_0K.device
    if d_pact <= 0:
        return torch.zeros((d_in, 0), dtype=V_0K.dtype, device=device)

    gen = torch.Generator(device=device).manual_seed(seed)
    rand_vecs = torch.randn(d_in, d_pact + V_0K.shape[1] + 8, device=device,
                            generator=gen, dtype=torch.float32)
    proj_rand = rand_vecs - V_0K.float() @ (V_0K.float().t() @ rand_vecs)
    Q_null, R_null = torch.linalg.qr(proj_rand, mode="reduced")
    diag = torch.abs(torch.diagonal(R_null))
    if diag.numel() > 0:
        tol = diag.max() * max(R_null.shape) * torch.finfo(R_null.dtype).eps
        Q_null = Q_null[:, diag > tol]

    n_null = Q_null.shape[1]
    if n_null <= d_pact:
        return Q_null.to(V_0K.dtype)
    perm = torch.randperm(n_null, generator=gen, device=device)[:d_pact]
    V_rand, _ = torch.linalg.qr(Q_null[:, perm], mode="reduced")
    return V_rand.to(V_0K.dtype)


def ablate_weight(W_A: torch.Tensor, V_subspace: torch.Tensor) -> torch.Tensor:
    """W_ablated = W_A (I - V_subspace V_subspace^⊤)."""
    if V_subspace.shape[1] == 0:
        return W_A.clone()
    return W_A - W_A @ (V_subspace @ V_subspace.t())


def ablate_delta_only(
    W_0: torch.Tensor, delta: torch.Tensor, V_subspace: torch.Tensor
) -> torch.Tensor:
    """只从 Δ 中消融，保留 W₀ 完整: W = W₀ + Δ(I - V V^⊤)."""
    if V_subspace.shape[1] == 0:
        return W_0 + delta
    return W_0 + delta - delta @ (V_subspace @ V_subspace.t())


# ===========================================================================
# 第二部分：辅助函数
# ===========================================================================

def _is_2d_weight(name: str, tensor: torch.Tensor) -> bool:
    if not torch.is_floating_point(tensor) or tensor.dim() != 2:
        return False
    lname = name.lower()
    for bad in ("text_projection", "norm", "ln_", "logit_scale", "head", "classification_head"):
        if bad in lname:
            return False
    return True


def _analyzable_2d_keys(
    tensors: Mapping[str, torch.Tensor],
    layer_regex: Optional[str] = None,
) -> List[str]:
    pattern = re.compile(layer_regex) if layer_regex else None
    keys = [k for k, v in tensors.items() if _is_2d_weight(k, v)]
    if pattern is not None:
        keys = [k for k in keys if pattern.search(k)]
    return keys


# ===========================================================================
# 第三部分：模型评估
# ===========================================================================

def evaluate_model(
    state_dict: Mapping[str, torch.Tensor],
    dataset_name: str,
    model_name: str,
    model_location: str,
    data_location: str,
    device: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> float:
    """通用推理评估接口：加载模型 + 测试数据，返回准确率."""
    from analysis.lbw_parameters.pact_insight_common import (
        build_args,
        classifier_from_state,
        get_loader,
        maybe_dictionarize,
        task_to_val_name,
    )
    from src.utils import utils

    args = build_args(model_name, model_location, data_location, batch_size, device, num_workers)
    val_name = task_to_val_name(dataset_name)
    model = classifier_from_state(model_name, state_dict, dataset_name, args)
    loader = get_loader(val_name, model.image_encoder.val_preprocess, args, "test", None, 0)

    correct, total = 0, 0
    for batch in loader:
        batch = maybe_dictionarize(batch)
        images = batch["images"].to(args.device)
        labels = batch["labels"].to(args.device)
        logits = utils.get_logits(images, model)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.numel()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(correct / max(total, 1))


# ===========================================================================
# 第四部分：消融实验主流程 —— 全模型一次性消融
# ===========================================================================

def _compute_layer_subspaces(
    state_pre: Dict[str, torch.Tensor],
    delta_A: Dict[str, torch.Tensor],
    layers: List[str],
    K: int,
    k: int,
    device: str,
) -> Dict[str, Dict]:
    """Phase 1: 对所有可分析层预计算 V_pact 和 d_pact, 不做任何评估.

    Returns:
        {layer_name: {"V_pact": Tensor, "W_A": Tensor, "V_0K": Tensor,
                       "d_pact": int, "d_out": int, "d_in": int, "orig_dtype": dtype}}
    """
    subspaces: Dict[str, Dict] = {}
    for layer_name in layers:
        if layer_name not in delta_A:
            continue
        W0 = state_pre[layer_name].detach().to(device=device, dtype=torch.float32)
        delta = delta_A[layer_name].detach().to(device=device, dtype=torch.float32)
        W_A = W0 + delta
        d_out, d_in = W_A.shape
        orig_dtype = state_pre[layer_name].dtype

        V_0K = right_singular_basis(W0, K)
        V_Ak = right_singular_basis(delta, k)
        V_pact = pact_subspace_basis(V_0K, V_Ak)
        d_pact = V_pact.shape[1]

        print(f"  [{layer_name}] d_out={d_out}, d_in={d_in}, d_pact={d_pact}")

        subspaces[layer_name] = {
            "V_pact": V_pact,
            "V_0K": V_0K,
            "W_A": W_A,
            "W_0": W0,
            "delta_A": delta,
            "d_pact": d_pact,
            "d_out": d_out,
            "d_in": d_in,
            "orig_dtype": orig_dtype,
        }
    return subspaces


def run_ablation_experiment(
    task: str,
    model_name: str,
    model_location: str,
    data_location: str,
    K: int,
    k: int,
    device: str,
    batch_size: int,
    num_workers: int,
    layer_regex: Optional[str],
    num_seeds: int,
    seed: int,
    out_dir: Optional[Path],
) -> Dict:
    """全模型一次性子空间消融实验.

    Phase 1: 对所有层预计算 V_{A,pact}（不做评估）
    Phase 2: 每个随机种子构建三个完整 state_dict:
        - orig:     所有层的 W_A = W_0 + Δ_A
        - pact:     所有层的 W_A (I - V_pact V_pact^⊤)
        - random:   所有层的 W_A (I - V_rand V_rand^⊤)
    Phase 3: 对每个完整 state_dict 评估一次（仅 3 次评估/种子）
    """
    from analysis.lbw_parameters.pact_insight_common import (
        load_checkpoint,
        task_to_val_name,
        tensor_dict_difference,
    )
    from src.utils.variables_and_paths import get_finetuned_path, get_zeroshot_path

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- 加载 checkpoint ----
    print(f"\n加载 checkpoint ...")
    pre_path = get_zeroshot_path(model_location, "MNISTVal", model=model_name)
    if not Path(pre_path).exists():
        raise FileNotFoundError(f"预训练 checkpoint 不存在: {pre_path}")
    state_pre = load_checkpoint(pre_path)
    print(f"  预训练: {pre_path}")

    ft_path = get_finetuned_path(model_location, task_to_val_name(task), model=model_name)
    if not Path(ft_path).exists():
        raise FileNotFoundError(f"微调 checkpoint 不存在: {ft_path}")
    state_ft = load_checkpoint(ft_path)
    print(f"  微调[{task}]: {ft_path}")

    delta_A = tensor_dict_difference(state_ft, state_pre)
    layers = _analyzable_2d_keys(state_pre, layer_regex)
    print(f"\n  可分析的 2D 权重层: {len(layers)} 层")

    # ---- Phase 1: 预计算所有层的 PACT 子空间 ----
    print(f"\n[Phase 1] 计算所有层的 PACT 子空间 ...")
    subspaces = _compute_layer_subspaces(state_pre, delta_A, layers, K, k, device)

    # 诊断信息：各层 d_pact
    layer_info = [
        {"layer": name, "d_out": info["d_out"], "d_in": info["d_in"],
         "d_pact": info["d_pact"]}
        for name, info in subspaces.items()
    ]
    total_d_pact = sum(info["d_pact"] for info in subspaces.values())
    total_d_in = sum(info["d_in"] for info in subspaces.values())
    print(f"  总 d_pact = {total_d_pact}  /  总 d_in = {total_d_in}"
          f"  ({100*total_d_pact/max(total_d_in,1):.2f}%)")

    # ---- Phase 2 & 3: 全模型消融 + 评估 ----
    print(f"\n[Phase 2] 全模型消融 + 评估 ({num_seeds} 个随机种子) ...")

    # 构建基准 state_dict: 即微调模型 state_ft（所有非 2D 层保持微调后的值）
    # 注意: 对于 2D 可分析层, state_ft 已经等于 W_0 + Δ_A = W_A
    orig_state = {k: v.clone() for k, v in state_ft.items()}

    # 先评估一次基准（与种子无关）
    print(f"  评估基准模型 (微调 checkpoint) ...")
    acc_orig = evaluate_model(orig_state, task, model_name,
                               model_location, data_location,
                               device, batch_size, num_workers)
    print(f"  基准准确率: {acc_orig:.4f}")

    seed_results: List[Dict] = []
    for s_idx in range(num_seeds):
        s = seed + s_idx * 1000
        torch.manual_seed(s)

        # 构建 PACT-消融 state_dict (从微调模型出发, 对所有 2D 层同时消融)
        state_pact = {k: v.clone() for k, v in state_ft.items()}
        # 构建 Random-消融 state_dict
        state_rand = {k: v.clone() for k, v in state_ft.items()}

        for name, info in subspaces.items():
            W_A = info["W_A"]
            W_0 = info["W_0"]
            delta_A = info["delta_A"]
            V_pact = info["V_pact"]
            V_0K = info["V_0K"]
            d_pact = info["d_pact"]
            odt = info["orig_dtype"]

            if d_pact == 0:
                # 该层无 PACT 子空间, 保持 W_A
                state_pact[name] = W_A.to(dtype=odt).cpu()
                state_rand[name] = W_A.to(dtype=odt).cpu()
            else:
                W_pact = ablate_delta_only(W_0, delta_A, V_pact)
                state_pact[name] = W_pact.to(dtype=odt).cpu()

                V_rand = random_orthogonal_in_perp(V_0K, d_pact, seed=s)
                W_rand = ablate_delta_only(W_0, delta_A, V_rand)
                state_rand[name] = W_rand.to(dtype=odt).cpu()

        print(f"\n  [seed={s}] 评估中 ...")
        acc_pact = evaluate_model(state_pact, task, model_name,
                                   model_location, data_location,
                                   device, batch_size, num_workers)
        acc_rand = evaluate_model(state_rand, task, model_name,
                                   model_location, data_location,
                                   device, batch_size, num_workers)

        seed_results.append({
            "seed": s,
            "acc_original": acc_orig,
            "acc_ablated_v_pact": acc_pact,
            "acc_ablated_v_rand": acc_rand,
            "delta_pact": acc_orig - acc_pact,
            "delta_rand": acc_orig - acc_rand,
            "pact_impact_ratio": (acc_orig - acc_pact) / max(acc_orig, 1e-8),
            "rand_impact_ratio": (acc_orig - acc_rand) / max(acc_orig, 1e-8),
        })
        print(f"    原始: {acc_orig:.4f}"
              f"  |  全模型消融 PACT: {acc_pact:.4f} (↓{acc_orig-acc_pact:.4f})"
              f"  |  全模型消融随机: {acc_rand:.4f} (↓{acc_orig-acc_rand:.4f})")

    # ---- 汇总 ----
    mean_delta_pact = np.mean([r["delta_pact"] for r in seed_results])
    std_delta_pact = np.std([r["delta_pact"] for r in seed_results])
    mean_delta_rand = np.mean([r["delta_rand"] for r in seed_results])
    std_delta_rand = np.std([r["delta_rand"] for r in seed_results])

    print(f"\n{'='*64}")
    print(f"全模型子空间消融实验 — Task: {task}")
    print(f"  可分析层数      : {len(subspaces)}")
    print(f"  总 d_pact       : {total_d_pact} / {total_d_in}"
          f" ({100*total_d_pact/max(total_d_in,1):.2f}%)")
    print(f"  随机种子数      : {num_seeds}")
    print(f"  基准准确率      : {acc_orig:.4f}")
    print(f"  全模型消融 PACT : {acc_orig - mean_delta_pact:.4f} (↓{mean_delta_pact:.4f} ± {std_delta_pact:.4f})")
    print(f"  全模型消融随机  : {acc_orig - mean_delta_rand:.4f} (↓{mean_delta_rand:.4f} ± {std_delta_rand:.4f})")
    print(f"  PACT / Random   : {mean_delta_pact/max(mean_delta_rand,1e-8):.2f}x")
    print(f"{'='*64}")

    result = {
        "layer_info": layer_info,
        "seed_results": seed_results,
        "summary": {
            "task": task,
            "n_layers": len(subspaces),
            "total_d_pact": total_d_pact,
            "total_d_in": total_d_in,
            "pact_fraction": float(total_d_pact / max(total_d_in, 1)),
            "num_seeds": num_seeds,
            "acc_original": acc_orig,
            "mean_acc_pact": float(acc_orig - mean_delta_pact),
            "mean_acc_rand": float(acc_orig - mean_delta_rand),
            "mean_delta_pact": mean_delta_pact,
            "std_delta_pact": std_delta_pact,
            "mean_delta_rand": mean_delta_rand,
            "std_delta_rand": std_delta_rand,
            "pact_vs_random_ratio": float(mean_delta_pact / max(mean_delta_rand, 1e-8)),
            "K": K, "k": k,
        },
        "meta": {
            "module": "subspace_ablation_eval",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "task": task, "model": model_name,
        },
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        import json
        import pandas as pd
        pd.DataFrame(layer_info).to_csv(out_dir / "layer_subspaces.csv", index=False)
        pd.DataFrame(seed_results).to_csv(out_dir / "ablation_per_seed.csv", index=False)
        torch.save(result, out_dir / "ablation_results.pt")
        with open(out_dir / "ablation_summary.json", "w", encoding="utf-8") as f:
            json.dump(result["summary"], f, indent=2, ensure_ascii=False)
        _plot_ablation_results(result["layer_info"], seed_results, subspaces, out_dir)
        print(f"\n✅ 结果已保存到 {out_dir}/")

    return result


# ===========================================================================
# 第五部分：可视化
# ===========================================================================

def _plot_ablation_results(
    layer_info: List[Dict],
    seed_results: List[Dict],
    subspaces: Dict[str, Dict],
    out_dir: Path,
) -> None:
    import pandas as pd
    df_layers = pd.DataFrame(layer_info)
    if df_layers.empty:
        return

    # 图1: 各层 d_pact 分布
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(df_layers)), df_layers["d_pact"], color="#ff7f0e", alpha=0.7)
    ax.set_xticks(range(len(df_layers)))
    ax.set_xticklabels([l.split(".")[-1][:15] for l in df_layers["layer"]],
                        rotation=45, fontsize=7, ha="right")
    ax.set_ylabel("d_pact")
    ax.set_title("PACT Subspace Dimensionality per Layer")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "d_pact_per_layer.png", dpi=200)
    plt.close(fig)

    # 图2: 消融结果柱状图 (全模型)
    if seed_results:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        seeds = [r["seed"] for r in seed_results]
        x = range(len(seeds))

        ax = axes[0]
        ax.bar([i - 0.15 for i in x], [r["delta_pact"] for r in seed_results],
               width=0.3, color="#d62728", alpha=0.85, label="Δ acc (PACT)")
        ax.bar([i + 0.15 for i in x], [r["delta_rand"] for r in seed_results],
               width=0.3, color="#7f7f7f", alpha=0.85, label="Δ acc (Random)")
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_ylabel("Accuracy Drop (Δ)")
        ax.set_title("Full-Model Ablation per Seed")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        ax = axes[1]
        mean_dp = np.mean([r["delta_pact"] for r in seed_results])
        mean_dr = np.mean([r["delta_rand"] for r in seed_results])
        std_dp = np.std([r["delta_pact"] for r in seed_results])
        std_dr = np.std([r["delta_rand"] for r in seed_results])
        bars = ax.bar([0, 1], [mean_dp, mean_dr], color=["#d62728", "#7f7f7f"], alpha=0.85)
        ax.errorbar([0, 1], [mean_dp, mean_dr], yerr=[std_dp, std_dr],
                     fmt="none", ecolor="black", capsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Ablate V_pact\n(all layers)", "Ablate V_rand\n(all layers)"])
        ax.set_ylabel("Mean Accuracy Drop (Δ)")
        ax.set_title(f"PACT/Random = {mean_dp/max(mean_dr,1e-8):.2f}x")
        ax.grid(axis="y", alpha=0.3)

        fig.suptitle("Full-Model Subspace Ablation", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "ablation_results.png", dpi=200)
        plt.close(fig)


# ===========================================================================
# 第六部分：CLI
# ===========================================================================

def _detect_available_tasks(model_location: str, model_name: str) -> List[str]:
    """自动扫描 checkpoint 目录, 返回所有可用微调任务名 (不含 Val 后缀)."""
    ckpt_dir = Path(model_location) / model_name
    if not ckpt_dir.exists():
        return []
    tasks = []
    for p in sorted(ckpt_dir.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if not name.endswith("Val"):
            continue
        task_base = name[:-3]  # 去掉 Val 后缀
        if task_base == "MNIST":
            continue
        tasks.append(task_base)
    return tasks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Subspace Ablation Evaluation — 子空间消融与降解实验"
    )
    # 任务指定 (三选一)
    task_group = p.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", default=None,
                            help="单个微调任务名 (如 Cars)")
    task_group.add_argument("--tasks", default=None,
                            help="逗号分隔的微调任务列表 (如 Cars,SUN397,GTSRB)")
    task_group.add_argument("--all-tasks", action="store_true",
                            help="自动扫描 checkpoint 目录下所有可用任务")

    p.add_argument("--model", default="ViT-B-16")
    p.add_argument("--model-location",
                   default=str(_REPO_ROOT / "models_complete" / "models" / "checkpoints"),
                   help="checkpoint 根目录")
    p.add_argument("--data-location",
                   default=str(_REPO_ROOT / "datasets"),
                   help="数据集根目录")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--K", type=int, default=15,
                   help="预训练基底奇异向量数")
    p.add_argument("--k", type=int, default=8,
                   help="任务更新基底奇异向量数")
    p.add_argument("--num-seeds", type=int, default=3,
                   help="随机 V_rand 的种子数量 (已自动跨种子平均)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layer-regex", default=None,
                   help="只分析匹配该正则的层")
    p.add_argument("--output-dir", default=None,
                   help="输出根目录 (默认: analysis/outputs3/subspace_ablation_only_delta/)")
    return p.parse_args()


def _resolve_tasks(cli: argparse.Namespace) -> List[str]:
    """根据 CLI 参数解析最终任务列表."""
    if cli.all_tasks:
        tasks = _detect_available_tasks(cli.model_location, cli.model)
        if not tasks:
            raise SystemExit(
                f"在 {cli.model_location}/{cli.model} 下未找到任何 Val 任务目录"
            )
        return tasks
    if cli.tasks:
        return [t.strip() for t in cli.tasks.split(",") if t.strip()]
    return [cli.task.strip()]


def main() -> None:
    cli = parse_args()
    tasks = _resolve_tasks(cli)

    if cli.output_dir:
        batch_root = Path(cli.output_dir)
    else:
        batch_root = _REPO_ROOT / "analysis" / "outputs3" / "subspace_ablation_only_delta"

    print("=" * 64)
    print("Subspace Ablation Evaluation")
    print(f"  Tasks       : {', '.join(tasks)} ({len(tasks)} 个)")
    print(f"  Model       : {cli.model}")
    print(f"  K={cli.K}  k={cli.k}  num_seeds={cli.num_seeds} (跨种子平均)")
    print(f"  Device      : {cli.device}")
    print(f"  Output      : {batch_root}")
    print("=" * 64)

    import pandas as pd
    task_summaries: List[Dict] = []
    all_per_layer: List[pd.DataFrame] = []

    for idx, task in enumerate(tasks):
        print(f"\n{'─'*48}")
        print(f"▶ [{idx+1}/{len(tasks)}] Task: {task}")
        print(f"{'─'*48}")

        task_out = batch_root / task
        try:
            result = run_ablation_experiment(
                task=task,
                model_name=cli.model,
                model_location=cli.model_location,
                data_location=cli.data_location,
                K=cli.K, k=cli.k,
                device=cli.device,
                batch_size=cli.batch_size,
                num_workers=cli.num_workers,
                layer_regex=cli.layer_regex,
                num_seeds=cli.num_seeds,
                seed=cli.seed,
                out_dir=task_out,
            )
            summary = result["summary"]
            summary["task"] = task
            task_summaries.append(summary)
            if result["layer_info"]:
                df = pd.DataFrame(result["layer_info"])
                df.insert(0, "task", task)
                all_per_layer.append(df)
        except Exception as e:
            print(f"  ⚠️ 失败: {e}")
            import traceback
            traceback.print_exc()

    if not task_summaries:
        raise SystemExit("所有任务均未能完成分析.")

    # 全局汇总
    summ_df = pd.DataFrame(task_summaries).sort_values(
        "pact_vs_random_ratio", ascending=False
    )
    full_df = pd.concat(all_per_layer, ignore_index=True)

    summ_df.to_csv(batch_root / "all_tasks_ablation_summary.csv", index=False)
    full_df.to_csv(batch_root / "all_tasks_layer_subspaces.csv", index=False)

    # 跨任务柱状图
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(summ_df))))
    xs = range(len(summ_df))
    ax.barh(list(xs), summ_df["pact_vs_random_ratio"], color="#d62728", alpha=0.85)
    ax.axvline(1.0, color="black", ls="--", lw=1.2, label="Ratio = 1 (no difference)")
    ax.set_yticks(list(xs))
    ax.set_yticklabels(summ_df["task"])
    ax.set_xlabel("Δ_pact / Δ_rand Ratio")
    ax.set_title("PACT vs Random Ablation Ratio across Tasks", fontweight="bold")
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(batch_root / "all_tasks_ablation_bar.png", dpi=200)
    plt.close(fig)

    # 额外图: Δ_pact vs Δ_rand 散点图 (全模型)
    if "mean_delta_pact" in summ_df.columns and "mean_delta_rand" in summ_df.columns:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(summ_df["mean_delta_rand"], summ_df["mean_delta_pact"],
                   c="#d62728", s=60, alpha=0.8, edgecolors="black")
        lims = [0, max(summ_df["mean_delta_pact"].max(), summ_df["mean_delta_rand"].max()) * 1.1]
        ax.plot(lims, lims, "k--", lw=1.0, label="y=x (equal impact)")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Δ acc (Random ablation, all layers)")
        ax.set_ylabel("Δ acc (PACT ablation, all layers)")
        ax.set_title("PACT vs Random: Full-Model Ablation Impact")
        ax.legend()
        ax.grid(alpha=0.3)
        # 标注任务名
        for _, row in summ_df.iterrows():
            ax.annotate(row["task"], (row["mean_delta_rand"], row["mean_delta_pact"]),
                        fontsize=6, alpha=0.7, ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(batch_root / "all_tasks_pact_vs_rand_scatter.png", dpi=200)
        plt.close(fig)

    g_mean = float(summ_df["pact_vs_random_ratio"].mean())
    print(f"\n{'='*64}")
    print(f"全局汇总 ({len(task_summaries)} 个任务，全模型一次性消融)")
    print(f"  平均 PACT/Random 比值 : {g_mean:.2f}x")
    print(f"{'='*64}")
    print(summ_df[["task", "n_layers", "mean_delta_pact", "mean_delta_rand",
                    "pact_vs_random_ratio"]].to_string(index=False))
    print(f"\n✅ 全局结果已保存到 {batch_root}/")
    print(f"   - all_tasks_ablation_summary.csv    各任务汇总 (全模型消融)")
    print(f"   - all_tasks_layer_subspaces.csv     所有层 PACT 维度信息")
    print(f"   - all_tasks_ablation_bar.png        跨任务 PACT/Random 比值")
    print(f"   - all_tasks_pact_vs_rand_scatter.png Δ_pact / Δ_rand 散点图")
    print(f"   每个任务子目录下另含 layer_subspaces.csv + 图")

    import json
    torch.save({
        "task_summaries": task_summaries,
        "global": {"mean_pact_vs_random_ratio": g_mean, "n_tasks": len(task_summaries)},
        "meta": {"module": "subspace_ablation_eval",
                 "timestamp": datetime.now().isoformat(timespec="seconds"),
                 "K": cli.K, "k": cli.k, "num_seeds": cli.num_seeds},
    }, batch_root / "all_tasks_results.pt")


if __name__ == "__main__":
    main()
