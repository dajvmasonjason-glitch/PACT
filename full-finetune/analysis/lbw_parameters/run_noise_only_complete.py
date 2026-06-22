#!/usr/bin/env python
"""单独运行高斯噪声实验（G7和G8）- 完整版

使用已有的module1和module2结果，只测试噪声添加的效果。

使用方法：
    python -m analysis.run_noise_only_complete --task-a SUN397 --task-b Cars

    # 指定噪声标准差
    python -m analysis.run_noise_only_complete --task-a SUN397 --task-b Cars --noise-std 0.01

    # 指定已有的mod1和mod2路径
    python -m analysis.run_noise_only_complete --mod1-path path/to/mod1_tensors.pt --mod2-path path/to/mod2_masks.pt
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
    build_args,
    equalize_masks,
    evaluate_state_dict,
    git_commit,
    random_mask_like,
)


# 噪声实验组
NOISE_GROUPS = {
    "G7_crucial_noise": ("mask_crucial", "fixed_noise"),
    "G8_safe_noise": ("mask_safe", "fixed_noise"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="单独运行高斯噪声实验（G7和G8）")
    p.add_argument("--task-a", help="任务A名称（如果不指定mod1/mod2路径则必需）")
    p.add_argument("--task-b", help="任务B名称（如果不指定mod1/mod2路径则必需）")
    p.add_argument("--model", default="ViT-B-16")
    p.add_argument("--model-location", default=None, help="模型检查点路径（默认从配置文件读取）")
    p.add_argument("--data-location", default=None, help="数据集路径（默认从配置文件读取）")
    p.add_argument("--mod1-path", help="mod1_tensors.pt的路径")
    p.add_argument("--mod2-path", help="mod2_masks.pt的路径")
    p.add_argument("--output-dir", default=None, help="输出目录（默认与mod2同目录）")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--num-seeds", type=int, default=3, help="随机种子数量")
    p.add_argument("--seed", type=int, default=0, help="起始随机种子")
    p.add_argument("--noise-std", type=float, default=0.005, help="高斯噪声标准差")
    return p.parse_args()


def masks_for_seed(mod2: Mapping, layer: str, seed: int) -> Dict[str, torch.Tensor]:
    """为指定种子生成masks"""
    if "raw_masks" in mod2 and layer in mod2["raw_masks"]:
        raw = mod2["raw_masks"][layer]
        eq = equalize_masks(raw, seed)
        k = int(next(iter(eq.values())).sum().item()) if eq else 0
        shape = next(iter(raw.values())).shape
        eq["mask_random"] = random_mask_like(shape, k, seed + 1000003)
        eq["k"] = k
        return eq
    return mod2["masks"][layer]


def apply_noise_ablation(
    state_a: Mapping[str, torch.Tensor],
    tau_a: Mapping[str, torch.Tensor],
    layers: List[str],
    masks_by_layer: Dict[str, Dict[str, torch.Tensor]],
    mask_name: str,
    noise_std: float,
    seed: int,
) -> tuple[Dict[str, torch.Tensor], int]:
    """对指定层应用高斯噪声消融"""
    edited = {k: v.detach().clone() for k, v in state_a.items()}
    total_modified = 0

    for layer in layers:
        if layer not in edited:
            continue
        if layer not in masks_by_layer or mask_name not in masks_by_layer[layer]:
            continue

        mask = masks_by_layer[layer][mask_name]
        layer_tensor = edited[layer].detach().cpu()
        layer_mask = mask.to(dtype=torch.bool, device="cpu")

        # 计算 W_pre = W_A - tau_A
        w_pre = state_a[layer].detach().cpu() - tau_a[layer].detach().cpu()

        # 生成高斯噪声: W_pre + N(0, noise_std)
        gen = torch.Generator().manual_seed(seed + hash(layer) % 1000000)
        noise = torch.randn(layer_tensor[layer_mask].shape, generator=gen, dtype=layer_tensor.dtype) * noise_std

        # 应用噪声
        layer_tensor[layer_mask] = (w_pre[layer_mask] + noise).to(layer_tensor.dtype)
        edited[layer] = layer_tensor
        total_modified += int(layer_mask.sum().item())

    return edited, total_modified


def plot_noise_results(df: pd.DataFrame, out_path: Path) -> None:
    """绘制噪声实验结果柱状图"""
    summary = df.groupby("group")["accuracy"].agg(["mean", "std"])
    plt.figure(figsize=(8, 5))
    x_pos = range(len(summary))
    plt.bar(x_pos, summary["mean"], yerr=summary["std"].fillna(0.0), capsize=5, alpha=0.8)
    plt.xticks(x_pos, summary.index, rotation=15, ha="right")
    plt.ylabel("Task A Accuracy", fontsize=12)
    plt.xlabel("Noise Group", fontsize=12)
    plt.title("Gaussian Noise Ablation Results (G7 & G8)", fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    cli = parse_args()

    # 从配置文件读取默认路径
    from analysis.lbw_parameters.pact_config import get_config
    config = get_config()

    cli.model_location = cli.model_location or config["model_location"]
    cli.data_location = cli.data_location or config["data_location"]

    # 验证参数
    if not cli.mod1_path or not cli.mod2_path:
        if not cli.task_a or not cli.task_b:
            raise ValueError("必须指定 --mod1-path 和 --mod2-path，或者指定 --task-a 和 --task-b")

    # 构建参数
    args = build_args(cli.model, cli.model_location, cli.data_location, cli.batch_size, cli.device, cli.num_workers)

    # 加载mod1和mod2
    mod1_path = Path(cli.mod1_path) if cli.mod1_path else None
    mod2_path = Path(cli.mod2_path) if cli.mod2_path else None

    if not mod1_path or not mod2_path:
        raise ValueError("必须指定 --mod1-path 和 --mod2-path")

    mod1 = torch.load(mod1_path, map_location="cpu")
    mod2 = torch.load(mod2_path, map_location="cpu")

    # 从mod1的meta中获取任务A的检查点路径并加载
    if "meta" in mod1 and "paths" in mod1["meta"] and "A" in mod1["meta"]["paths"]:
        checkpoint_path_a = mod1["meta"]["paths"]["A"]
        print(f"从 mod1 meta 加载任务A检查点: {checkpoint_path_a}")
        state_a = torch.load(checkpoint_path_a, map_location="cpu")
        if "model_name" in state_a:
            state_a = dict(state_a)
            state_a.pop("model_name")
    else:
        raise ValueError("mod1_tensors.pt 中缺少 meta['paths']['A']，无法加载任务A的检查点")

    tau_a = mod1["tau_A"]

    # 设置输出目录
    out_dir = Path(cli.output_dir) if cli.output_dir else mod2_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 获取选定的层
    selected_layers = list(mod2["selected_layers"])
    if not selected_layers:
        raise ValueError("mod2中没有选定的层，请重新运行Module 2")

    print(f"\n{'='*60}")
    print(f"单独运行高斯噪声实验 (G7 & G8)")
    print(f"{'='*60}")
    print(f"任务A: {cli.task_a}")
    print(f"任务B: {cli.task_b}")
    print(f"目标层数: {len(selected_layers)}")
    print(f"噪声标准差: {cli.noise_std}")
    print(f"随机种子数: {cli.num_seeds}")
    print(f"{'='*60}\n")

    # 准备masks
    masks_by_layer = {}
    for layer in selected_layers:
        masks_by_layer[layer] = masks_for_seed(mod2, layer, cli.seed)

    # 运行噪声实验
    rows = []

    for seed_offset in range(cli.num_seeds):
        seed = cli.seed + seed_offset
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        # 为当前种子重新生成masks
        for layer in selected_layers:
            masks_by_layer[layer] = masks_for_seed(mod2, layer, seed)

        # 运行G7和G8
        for group, (mask_name, _) in NOISE_GROUPS.items():
            edited, total_modified = apply_noise_ablation(
                state_a, tau_a, selected_layers, masks_by_layer, mask_name, cli.noise_std, seed
            )
            acc = evaluate_state_dict(edited, cli.task_a, args)

            rows.append({
                "seed": seed,
                "group": group,
                "mask_name": mask_name,
                "accuracy": acc,
                "num_layers": len(selected_layers),
                "total_params_modified": total_modified,
                "noise_std": cli.noise_std,
            })
            print(f"  {group}: accuracy={acc:.4f}, modified_params={total_modified:,}")

    # 保存结果
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "noise_only_results.csv", index=False)

    # 打印汇总
    print(f"\n{'='*60}")
    print(f"高斯噪声实验汇总")
    print(f"{'='*60}")
    summary = df.groupby("group")["accuracy"].agg(["mean", "std"])
    print(summary.to_string())
    print(f"{'='*60}\n")

    # 绘制结果图
    plot_noise_results(df, out_dir / "noise_only_results.png")

    # 保存详细结果
    payload = {
        "results": df.to_dict(orient="records"),
        "summary": {
            group: {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "per_seed": [float(x) for x in vals.tolist()],
            }
            for group, vals in df.groupby("group")["accuracy"]
        },
        "meta": {
            "module": "noise_only_experiment",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "task_A": cli.task_a,
            "task_B": cli.task_b,
            "model": cli.model,
            "paths": mod1["meta"]["paths"] if "meta" in mod1 and "paths" in mod1["meta"] else {},
            "mod1_path": str(mod1_path),
            "mod2_path": str(mod2_path),
            "num_seeds": cli.num_seeds,
            "seed": cli.seed,
            "noise_std": cli.noise_std,
            "target_layers": selected_layers,
            "num_target_layers": len(selected_layers),
        },
    }
    torch.save(payload, out_dir / "noise_only_results.pt")

    print(f"✅ 结果已保存到:")
    print(f"   - {out_dir / 'noise_only_results.csv'}")
    print(f"   - {out_dir / 'noise_only_results.pt'}")
    print(f"   - {out_dir / 'noise_only_results.png'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()


