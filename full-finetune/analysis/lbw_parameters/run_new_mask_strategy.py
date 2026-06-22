#!/usr/bin/env python
"""新的mask策略实验

不考虑任务B的enrichment和p-value，直接根据任务A的tau值和fisher值划分进行mask。
只进行两个实验：
1. pre+tauA+tauB（标准合并）
2. crucial to pre（关键参数恢复到pre）

使用方法：
    python -m analysis.run_new_mask_strategy --task-a SUN397 --task-b Cars --mod1-path path/to/mod1_tensors.pt

    # 指定分位数阈值
    python -m analysis.run_new_mask_strategy --task-a SUN397 --task-b Cars --mod1-path path/to/mod1.pt --q-a-zero 0.30 --q-a-sensitive 0.70
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from analysis.lbw_parameters.pact_insight_common import (
    analyzable_keys,
    build_args,
    equalize_masks,
    evaluate_state_dict,
    flatten,
    git_commit,
    quantile,
    random_mask_like,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="新的mask策略实验")
    p.add_argument("--task-a", required=True, help="任务A名称")
    p.add_argument("--task-b", required=True, help="任务B名称")
    p.add_argument("--model", default="ViT-B-16")
    p.add_argument("--model-location", default="models/ckpts")
    p.add_argument("--data-location", default="datasets")
    p.add_argument("--mod1-path", required=True, help="mod1_tensors.pt的路径")
    p.add_argument("--output-dir", default=None, help="输出目录（默认与mod1同目录）")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--num-seeds", type=int, default=3, help="随机种子数量")
    p.add_argument("--seed", type=int, default=0, help="起始随机种子")
    p.add_argument("--q-a-zero", type=float, default=0.30, help="任务A的tau低分位数阈值")
    p.add_argument("--q-a-sensitive", type=float, default=0.70, help="任务A的fisher高分位数阈值")
    p.add_argument("--layer-regex", default=None, help="层名称正则表达式过滤")
    p.add_argument("--fisher-key", default="F_A", help="Fisher信息矩阵的键名")
    return p.parse_args()


def generate_masks_by_tau_fisher(
    tau_a: torch.Tensor,
    fisher: torch.Tensor,
    q_zero: float,
    q_sens: float,
) -> Dict[str, torch.Tensor]:
    """根据任务A的tau值和fisher值生成masks

    不考虑任务B的enrichment和p-value，直接根据阈值划分。

    Args:
        tau_a: 任务A的tau值
        fisher: 任务A的Fisher信息
        q_zero: tau_a的低分位数阈值（小于此值认为对任务A不重要）
        q_sens: fisher的高分位数阈值（大于此值认为对任务A敏感）

    Returns:
        包含mask_crucial的字典
    """
    abs_a = tau_a.abs()
    mask_a_low = abs_a < quantile(abs_a, q_zero)
    mask_f_high = fisher > quantile(fisher, q_sens)

    # crucial: 对任务A不重要但敏感的参数
    mask_crucial = mask_a_low & mask_f_high

    return {"mask_crucial": mask_crucial}


def apply_crucial_to_pre(
    state_a: Mapping[str, torch.Tensor],
    tau_a: Mapping[str, torch.Tensor],
    layers: List[str],
    masks_by_layer: Dict[str, Dict[str, torch.Tensor]],
) -> tuple[Dict[str, torch.Tensor], int]:
    """将crucial参数恢复到pre状态"""
    edited = {k: v.detach().clone() for k, v in state_a.items()}
    total_modified = 0

    for layer in layers:
        if layer not in edited:
            continue
        if layer not in masks_by_layer or "mask_crucial" not in masks_by_layer[layer]:
            continue

        mask = masks_by_layer[layer]["mask_crucial"]
        layer_tensor = edited[layer].detach().cpu()
        layer_mask = mask.to(dtype=torch.bool, device="cpu")

        # 计算 W_pre = W_A - tau_A
        w_pre = state_a[layer].detach().cpu() - tau_a[layer].detach().cpu()

        # 将crucial位置恢复到W_pre
        layer_tensor[layer_mask] = w_pre[layer_mask].to(layer_tensor.dtype)
        edited[layer] = layer_tensor
        total_modified += int(layer_mask.sum().item())

    return edited, total_modified


def plot_comparison(df: pd.DataFrame, out_path: Path) -> None:
    """绘制两个实验的对比图"""
    summary = df.groupby("experiment")["accuracy"].agg(["mean", "std"])
    plt.figure(figsize=(8, 5))
    x_pos = range(len(summary))
    plt.bar(x_pos, summary["mean"], yerr=summary["std"].fillna(0.0), capsize=5, alpha=0.8)
    plt.xticks(x_pos, summary.index.tolist(), rotation=15, ha="right")
    plt.ylabel("Task A Accuracy", fontsize=12)
    plt.xlabel("Experiment", fontsize=12)
    plt.title("New Mask Strategy: Standard Merge vs Crucial Protection", fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_layer_visualization(
    tau_a: torch.Tensor,
    fisher: torch.Tensor,
    mask: torch.Tensor,
    layer_name: str,
    out_path: Path,
) -> None:
    """可视化单层的mask分布"""
    abs_tau = flatten(tau_a.abs())
    f = flatten(fisher)
    m = flatten(mask).bool()

    # 采样以避免过多点
    if abs_tau.numel() > 10000:
        gen = torch.Generator().manual_seed(0)
        idx = torch.randperm(abs_tau.numel(), generator=gen)[:10000]
        abs_tau_sample = abs_tau[idx]
        f_sample = f[idx]
        m_sample = m[idx]
    else:
        abs_tau_sample = abs_tau
        f_sample = f
        m_sample = m

    plt.figure(figsize=(7, 5))
    plt.hexbin(abs_tau_sample.numpy(), f_sample.numpy(), gridsize=60, mincnt=1, bins="log", cmap="Blues")
    plt.scatter(abs_tau_sample[m_sample].numpy(), f_sample[m_sample].numpy(), s=5, c="red", label="mask_crucial", alpha=0.7)
    plt.xlabel("|tau_A|", fontsize=12)
    plt.ylabel("Fisher_A", fontsize=12)
    plt.title(f"Mask Distribution: {layer_name}", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    cli = parse_args()

    # 构建参数
    args = build_args(cli.model, cli.model_location, cli.data_location, cli.batch_size, cli.device, cli.num_workers)

    # 加载检查点
    _, state_a, _, paths = load_task_checkpoints(cli.model_location, cli.model, cli.task_a, cli.task_b)

    # 加载mod1
    mod1_path = Path(cli.mod1_path)
    mod1 = torch.load(mod1_path, map_location="cpu")

    tau_a = mod1["tau_A"]
    tau_b = mod1["tau_B"]
    fisher = mod1[cli.fisher_key]

    # 设置输出目录
    out_dir = Path(cli.output_dir) if cli.output_dir else mod1_path.parent / "new_mask_strategy"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"新的Mask策略实验")
    print(f"{'='*60}")
    print(f"任务A: {cli.task_a}")
    print(f"任务B: {cli.task_b}")
    print(f"q_A_zero: {cli.q_a_zero}")
    print(f"q_A_sensitive: {cli.q_a_sensitive}")
    print(f"Fisher key: {cli.fisher_key}")
    print(f"{'='*60}\n")

    # 为所有可分析的层生成masks
    print("生成masks...")
    analyzable_layers = analyzable_keys(tau_a, cli.layer_regex)
    masks_by_layer = {}
    mask_stats = []

    for layer in analyzable_layers:
        if layer not in tau_a or layer not in fisher:
            continue

        # 生成mask（不考虑任务B）
        raw_masks = generate_masks_by_tau_fisher(tau_a[layer], fisher[layer], cli.q_a_zero, cli.q_a_sensitive)
        masks_by_layer[layer] = raw_masks

        # 统计mask信息
        n_crucial = int(raw_masks["mask_crucial"].sum().item())
        n_total = raw_masks["mask_crucial"].numel()
        mask_stats.append({
            "layer": layer,
            "n_crucial": n_crucial,
            "n_total": n_total,
            "ratio": n_crucial / n_total if n_total > 0 else 0.0,
        })

    # 保存mask统计
    mask_df = pd.DataFrame(mask_stats)
    mask_df.to_csv(out_dir / "mask_statistics.csv", index=False)
    print(f"✓ 生成了 {len(masks_by_layer)} 层的masks")
    print(f"  平均crucial参数比例: {mask_df['ratio'].mean():.4f}")

    # 选择crucial参数最多的前几层进行可视化
    top_layers = mask_df.nlargest(3, "n_crucial")["layer"].tolist()
    for layer in top_layers:
        safe_layer_name = layer.replace("/", "_").replace(".", "_")
        plot_layer_visualization(
            tau_a[layer],
            fisher[layer],
            masks_by_layer[layer]["mask_crucial"],
            layer,
            out_dir / f"mask_viz_{safe_layer_name}.png",
        )
    print(f"✓ 可视化了前3层的mask分布\n")

    # 运行实验
    print(f"{'='*60}")
    print("运行实验...")
    print(f"{'='*60}\n")

    rows = []

    for seed_offset in range(cli.num_seeds):
        seed = cli.seed + seed_offset
        print(f"Seed {seed}:")

        # 实验1: 标准合并 (W_pre + tau_A + tau_B)
        print("  [1] 标准合并 (pre + tau_A + tau_B)...")
        state_merged = {k: v.detach().clone() for k, v in state_a.items()}
        for key in state_merged.keys():
            if key in tau_b:
                # W_merged = state_a + tau_B (因为 state_a = W_pre + tau_A)
                state_merged[key] = state_a[key].detach().cpu() + tau_b[key].detach().cpu()

        acc_standard = evaluate_state_dict(state_merged, cli.task_a, args)
        rows.append({
            "seed": seed,
            "experiment": "standard_merge",
            "accuracy": acc_standard,
        })
        print(f"      准确率: {acc_standard:.4f}")

        # 实验2: Crucial保护 (将crucial参数恢复到W_pre)
        print("  [2] Crucial保护 (crucial to pre)...")
        all_layers = list(masks_by_layer.keys())
        state_protected, total_protected = apply_crucial_to_pre(
            state_merged, tau_a, all_layers, masks_by_layer
        )

        acc_protected = evaluate_state_dict(state_protected, cli.task_a, args)
        rows.append({
            "seed": seed,
            "experiment": "crucial_protected",
            "accuracy": acc_protected,
            "protected_params": total_protected,
        })
        print(f"      准确率: {acc_protected:.4f}")
        print(f"      保护参数数: {total_protected:,}\n")

    # 保存结果
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "new_mask_results.csv", index=False)

    # 打印汇总
    print(f"\n{'='*60}")
    print(f"实验汇总")
    print(f"{'='*60}")
    summary = df.groupby("experiment")["accuracy"].agg(["mean", "std"])
    print(summary.to_string())

    # 计算改进
    standard_mean = float(summary.loc["standard_merge", "mean"])
    protected_mean = float(summary.loc["crucial_protected", "mean"])
    improvement = protected_mean - standard_mean
    print(f"\n性能改进: {improvement:+.4f} ({improvement*100:+.2f}%)")
    print(f"{'='*60}\n")

    # 绘制对比图
    plot_comparison(df, out_dir / "new_mask_comparison.png")

    # 保存详细结果
    payload = {
        "results": df.to_dict(orient="records"),
        "summary": {
            exp: {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "per_seed": [float(x) for x in vals.tolist()],
            }
            for exp, vals in df.groupby("experiment")["accuracy"]
        },
        "improvement": float(improvement),
        "mask_statistics": mask_df.to_dict(orient="records"),
        "meta": {
            "module": "new_mask_strategy",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "task_A": cli.task_a,
            "task_B": cli.task_b,
            "model": cli.model,
            "paths": paths,
            "mod1_path": str(mod1_path),
            "num_seeds": cli.num_seeds,
            "seed": cli.seed,
            "q_A_zero": cli.q_a_zero,
            "q_A_sensitive": cli.q_a_sensitive,
            "fisher_key": cli.fisher_key,
            "num_layers": len(masks_by_layer),
            "total_protected_params": int(df[df["experiment"] == "crucial_protected"]["protected_params"].mean()),
        },
    }
    torch.save(payload, out_dir / "new_mask_results.pt")

    print(f"✅ 结果已保存到:")
    print(f"   - {out_dir / 'new_mask_results.csv'}")
    print(f"   - {out_dir / 'new_mask_results.pt'}")
    print(f"   - {out_dir / 'new_mask_comparison.png'}")
    print(f"   - {out_dir / 'mask_statistics.csv'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

