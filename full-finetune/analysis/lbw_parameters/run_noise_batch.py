#!/usr/bin/env python
"""批量运行多个任务对的高斯噪声实验（G7和G8）

使用方法：
    # 运行所有预定义的任务对
    python -m analysis.run_noise_batch

    # 运行指定的任务对
    python -m analysis.run_noise_batch --task-pairs "SUN397:Cars,GTSRB:MNIST"

    # 指定噪声标准差
    python -m analysis.run_noise_batch --noise-std 0.01

    # 从配置文件读取
    python -m analysis.run_noise_batch --use-config
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from analysis.lbw_parameters.pact_config import get_config, AVAILABLE_TASK_PAIRS


def task_to_base_name(task: str) -> str:
    """移除任务名称中的 'Val' 后缀"""
    return task[:-3] if task.endswith("Val") else task


def make_output_dir(output_root: str, task_a: str, task_b: str) -> Path:
    """创建输出目录"""
    out = Path(output_root) / f"{task_to_base_name(task_a)}-{task_to_base_name(task_b)}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def parse_task_pairs(task_pairs_str: str) -> list[tuple[str, str]]:
    """解析任务对字符串"""
    if not task_pairs_str:
        return AVAILABLE_TASK_PAIRS

    pairs = []
    for pair_str in task_pairs_str.split(","):
        task_a, task_b = pair_str.strip().split(":")
        pairs.append((task_a.strip(), task_b.strip()))
    return pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="批量运行高斯噪声实验")
    p.add_argument(
        "--task-pairs",
        help='任务对列表，格式: "TaskA:TaskB,TaskC:TaskD"。不指定则使用配置文件中的所有任务对',
    )
    p.add_argument("--model", default="ViT-B-16")
    p.add_argument("--model-location", default=None, help="模型检查点路径（默认从配置文件读取）")
    p.add_argument("--data-location", default=None, help="数据集路径（默认从配置文件读取）")
    p.add_argument("--output-root", default=None, help="输出根目录（默认从配置文件读取）")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-std", type=float, default=0.01, help="高斯噪声标准差")
    p.add_argument("--use-config", action="store_true", help="从pact_config.py读取配置")
    p.add_argument("--skip-missing", action="store_true", help="跳过缺失mod1/mod2的任务对")
    return p.parse_args()


def find_mod_files(output_root: str, task_a: str, task_b: str) -> tuple[Path | None, Path | None]:
    """查找mod1和mod2文件

    Returns:
        (mod1_path, mod2_path) 如果文件不存在则返回None
    """
    out_dir = make_output_dir(output_root, task_a, task_b)
    mod1_path = out_dir / "mod1_tensors.pt"
    mod2_path = out_dir / "mod2_masks.pt"

    return (
        mod1_path if mod1_path.exists() else None,
        mod2_path if mod2_path.exists() else None,
    )


def run_single_noise_experiment(
    task_a: str,
    task_b: str,
    mod1_path: Path,
    mod2_path: Path,
    cli: argparse.Namespace,
) -> bool:
    """运行单个任务对的噪声实验

    Returns:
        True if successful, False otherwise
    """
    cmd = [
        sys.executable, "-m", "analysis.run_noise_only_complete",
        "--task-a", task_a,
        "--task-b", task_b,
        "--model", cli.model,
        "--model-location", cli.model_location,
        "--data-location", cli.data_location,
        "--mod1-path", str(mod1_path),
        "--mod2-path", str(mod2_path),
        "--device", cli.device,
        "--batch-size", str(cli.batch_size),
        "--num-workers", str(cli.num_workers),
        "--num-seeds", str(cli.num_seeds),
        "--seed", str(cli.seed),
        "--noise-std", str(cli.noise_std),
    ]

    print(f"\n运行命令: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    return result.returncode == 0


def main() -> None:
    cli = parse_args()

    # 读取配置文件中的默认路径（与 run_pact_batch 保持一致）
    config = get_config()
    cli.model_location = cli.model_location or config["model_location"]
    cli.data_location = cli.data_location or config["data_location"]
    cli.output_root = cli.output_root or config["output_root"]

    # --use-config 时额外覆盖运行参数
    if cli.use_config:
        cli.device = config["device"]
        cli.batch_size = config["batch_size"]
        cli.num_workers = config["num_workers"]
        cli.num_seeds = config["num_seeds"]
        cli.seed = config["seed"]

    # 解析任务对
    if cli.task_pairs:
        task_pairs = parse_task_pairs(cli.task_pairs)
    else:
        task_pairs = AVAILABLE_TASK_PAIRS

    print("\n" + "=" * 60)
    print(f"批量运行高斯噪声实验 (G7 & G8)")
    print("=" * 60)
    print(f"任务对数量: {len(task_pairs)}")
    print(f"噪声标准差: {cli.noise_std}")
    print(f"随机种子数: {cli.num_seeds}")
    print(f"输出根目录: {cli.output_root}")
    print("=" * 60)

    # 检查mod文件并运行实验
    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, (task_a, task_b) in enumerate(task_pairs, 1):
        print(f"\n[{i}/{len(task_pairs)}] 任务对: {task_a} vs {task_b}")
        print("-" * 60)

        # 查找mod文件
        mod1_path, mod2_path = find_mod_files(cli.output_root, task_a, task_b)

        if mod1_path is None or mod2_path is None:
            missing = []
            if mod1_path is None:
                missing.append("mod1_tensors.pt")
            if mod2_path is None:
                missing.append("mod2_masks.pt")

            print(f"⚠️  缺失文件: {', '.join(missing)}")

            if cli.skip_missing:
                print("跳过此任务对")
                skip_count += 1
                continue
            else:
                print("❌ 终止运行（使用 --skip-missing 跳过缺失的任务对）")
                sys.exit(1)

        print(f"✓ 找到 mod1: {mod1_path}")
        print(f"✓ 找到 mod2: {mod2_path}")

        # 运行实验
        success = run_single_noise_experiment(task_a, task_b, mod1_path, mod2_path, cli)

        if success:
            print(f"✅ 任务对 {task_a}-{task_b} 完成")
            success_count += 1
        else:
            print(f"❌ 任务对 {task_a}-{task_b} 失败")
            fail_count += 1

            if not cli.skip_missing:
                response = "y"  # 自动继续运行其他任务对
                if response.lower() != "y":
                    break

    # 打印汇总
    print("\n" + "=" * 60)
    print("批量运行完成")
    print("=" * 60)
    print(f"✅ 成功: {success_count}")
    print(f"⚠️  跳过: {skip_count}")
    print(f"❌ 失败: {fail_count}")
    print(f"总计: {len(task_pairs)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
