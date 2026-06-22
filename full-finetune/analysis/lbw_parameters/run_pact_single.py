#!/usr/bin/env python
"""运行单个任务对的 PACT Insight 实验

使用方法：
    python -m analysis.run_pact_single --task-a SUN397 --task-b Cars

或者使用默认配置：
    python -m analysis.run_pact_single
"""

import argparse
import subprocess
import sys
from pathlib import Path

from analysis.lbw_parameters.pact_config import get_config, print_config, DEFAULT_TASK_A, DEFAULT_TASK_B


def parse_args():
    parser = argparse.ArgumentParser(description="运行单个任务对的 PACT Insight 实验")
    parser.add_argument("--task-a", default=DEFAULT_TASK_A, help="任务 A 名称")
    parser.add_argument("--task-b", default=DEFAULT_TASK_B, help="任务 B 名称")
    parser.add_argument("--skip-mod1", action="store_true", help="跳过 Module 1")
    parser.add_argument("--skip-mod2", action="store_true", help="跳过 Module 2")
    parser.add_argument("--skip-mod3", action="store_true", help="跳过 Module 3")
    return parser.parse_args()


def run_module_1(task_a, task_b, config):
    """运行 Module 1: 计算任务向量和 Fisher 矩阵"""
    print("\n" + "=" * 60)
    print(f"Module 1: 计算任务向量和 Fisher 矩阵 ({task_a} vs {task_b})")
    print("=" * 60)

    cmd = [
        sys.executable, "-m", "analysis.lbw_parameters.pact_insight_compute",
        "--task-a", task_a,
        "--task-b", task_b,
        "--model", config["model"],
        "--model-location", config["model_location"],
        "--data-location", config["data_location"],
        "--output-root", config["output_root"],
        "--device", config["device"],
        "--batch-size", str(config["batch_size"]),
        "--num-workers", str(config["num_workers"]),
        "--fisher-type", config["fisher_type"],
        "--num-mc-samples", str(config["num_mc_samples"]),
        "--fisher-samples", str(config["fisher_samples"]),
        "--seed", str(config["seed"]),
    ]

    print(f"运行命令: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ Module 1 失败，退出码: {result.returncode}")
        sys.exit(1)

    print("\n✅ Module 1 完成")
    return Path(config["output_root"]) / f"{task_a}-{task_b}" / "mod1_tensors.pt"


def run_module_2(mod1_path, config):
    """运行 Module 2: 扫描层并生成掩码"""
    print("\n" + "=" * 60)
    print("Module 2: 扫描层并生成掩码")
    print("=" * 60)

    cmd = [
        sys.executable, "-m", "analysis.lbw_parameters.pact_insight_masks",
        "--mod1-path", str(mod1_path),
        "--n-permutations", str(config["n_permutations"]),
    ]

    print(f"运行命令: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ Module 2 失败，退出码: {result.returncode}")
        sys.exit(1)

    print("\n✅ Module 2 完成")
    return mod1_path.parent / "mod2_masks.pt"


def run_module_3(task_a, task_b, mod1_path, mod2_path, config):
    """运行 Module 3: 消融实验"""
    print("\n" + "=" * 60)
    print("Module 3: 消融实验")
    print("=" * 60)

    cmd = [
        sys.executable, "-m", "analysis.lbw_parameters.pact_insight_ablation",
        "--task-a", task_a,
        "--task-b", task_b,
        "--model", config["model"],
        "--model-location", config["model_location"],
        "--data-location", config["data_location"],
        "--mod1-path", str(mod1_path),
        "--mod2-path", str(mod2_path),
        "--device", config["device"],
        "--batch-size", str(config["batch_size"]),
        "--num-workers", str(config["num_workers"]),
        "--num-seeds", str(config["num_seeds"]),
    ]

    print(f"运行命令: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ Module 3 失败，退出码: {result.returncode}")
        sys.exit(1)

    print("\n✅ Module 3 完成")


def main():
    args = parse_args()
    config = get_config()

    print_config()
    print(f"\n任务对: {args.task_a} vs {args.task_b}\n")

    # Module 1
    if not args.skip_mod1:
        mod1_path = run_module_1(args.task_a, args.task_b, config)
    else:
        mod1_path = Path(config["output_root"]) / f"{args.task_a}-{args.task_b}" / "mod1_tensors.pt"
        print(f"⏭️  跳过 Module 1，使用现有文件: {mod1_path}")

    # Module 2
    if not args.skip_mod2:
        mod2_path = run_module_2(mod1_path, config)
    else:
        mod2_path = mod1_path.parent / "mod2_masks.pt"
        print(f"⏭️  跳过 Module 2，使用现有文件: {mod2_path}")

    # Module 3
    if not args.skip_mod3:
        run_module_3(args.task_a, args.task_b, mod1_path, mod2_path, config)
    else:
        print("⏭️  跳过 Module 3")

    print("\n" + "=" * 60)
    print("🎉 实验完成！")
    print("=" * 60)
    print(f"\n输出目录: {mod1_path.parent}")
    print("\n生成的文件:")
    print("  - mod1_tensors.pt (任务向量和 Fisher 矩阵)")
    print("  - mod2_masks.pt (层级掩码)")
    print("  - mod2_layer_scan.csv (层级统计)")
    print("  - mod2_*.png (可视化图表)")
    print("  - mod3_results.csv (消融实验结果)")
    print("  - mod3_*.png (结果可视化)")


if __name__ == "__main__":
    main()
