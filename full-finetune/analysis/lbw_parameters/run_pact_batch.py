#!/usr/bin/env python
"""批量运行多个任务对的 PACT Insight 实验

使用方法：
    # 运行所有预定义的任务对
    python -m analysis.run_pact_batch

    # 运行指定的任务对
    python -m analysis.run_pact_batch --task-pairs "SUN397:Cars,EuroSAT:SVHN"

    # 运行后生成汇总报告
    python -m analysis.run_pact_batch --aggregate
"""

import argparse
import subprocess
import sys
from pathlib import Path

from analysis.lbw_parameters.pact_config import get_config, print_config, AVAILABLE_TASK_PAIRS


def parse_args():
    parser = argparse.ArgumentParser(description="批量运行 PACT Insight 实验")
    parser.add_argument(
        "--task-pairs",
        help='任务对列表，格式: "TaskA:TaskB,TaskC:TaskD"',
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="运行完成后生成汇总报告",
    )
    return parser.parse_args()


def parse_task_pairs(task_pairs_str):
    """解析任务对字符串"""
    if not task_pairs_str:
        return AVAILABLE_TASK_PAIRS

    pairs = []
    for pair_str in task_pairs_str.split(","):
        task_a, task_b = pair_str.strip().split(":")
        pairs.append((task_a.strip(), task_b.strip()))
    return pairs


def run_batch(task_pairs, config):
    """批量运行实验"""
    print("\n" + "=" * 60)
    print(f"批量运行 {len(task_pairs)} 个任务对")
    print("=" * 60)

    for i, (task_a, task_b) in enumerate(task_pairs, 1):
        print(f"\n[{i}/{len(task_pairs)}] 运行任务对: {task_a} vs {task_b}")
        print("-" * 60)

        # 使用 run_pact_single.py 运行每个任务对
        cmd = [
            sys.executable, "-m", "analysis.run_pact_single",
            "--task-a", task_a,
            "--task-b", task_b,
        ]

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"\n❌ 任务对 {task_a}-{task_b} 失败")
            response = input("继续运行其他任务对？(y/n): ")
            if response.lower() != "y":
                sys.exit(1)
        else:
            print(f"\n✅ 任务对 {task_a}-{task_b} 完成")

    print("\n" + "=" * 60)
    print("🎉 所有任务对运行完成！")
    print("=" * 60)


def run_aggregate(config):
    """生成汇总报告"""
    print("\n" + "=" * 60)
    print("生成汇总报告")
    print("=" * 60)

    cmd = [
        sys.executable, "-m", "analysis.lbw_parameters.pact_insight_aggregate",
        "--output-root", config["output_root"],
    ]

    print(f"运行命令: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ 汇总报告生成失败")
        sys.exit(1)

    print("\n✅ 汇总报告生成完成")
    print(f"\n汇总输出目录: {Path(config['output_root']) / 'summary'}")


def main():
    args = parse_args()
    config = get_config()

    print_config()

    # 解析任务对
    task_pairs = parse_task_pairs(args.task_pairs)

    print("\n将运行以下任务对:")
    for i, (task_a, task_b) in enumerate(task_pairs, 1):
        print(f"  {i}. {task_a} vs {task_b}")

    #response = input("\n确认运行？(y/n): ")
    response = "y"
    if response.lower() != "y":
        print("已取消")
        sys.exit(0)

    # 批量运行
    run_batch(task_pairs, config)

    # 生成汇总报告
    if args.aggregate:
        run_aggregate(config)


if __name__ == "__main__":
    main()
