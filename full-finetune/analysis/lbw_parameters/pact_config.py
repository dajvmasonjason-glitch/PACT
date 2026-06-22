"""PACT Insight 实验配置文件

根据你的服务器环境修改以下路径配置
"""

import os
from pathlib import Path

# ============ 路径配置 ============
# 项目根目录（服务器上的路径）
PROJECT_ROOT = Path.home() / "iso-merging-main"

# 模型检查点路径
# 你的实际路径：~/iso-merging-main/models_complete/models/checkpoints/
MODEL_LOCATION = str(PROJECT_ROOT / "models_complete" / "models" / "checkpoints")

# 数据集路径
# 你的实际路径：~/iso-merging-main/datasets/
DATA_LOCATION = str(PROJECT_ROOT / "datasets")

# 输出路径
OUTPUT_ROOT = str(PROJECT_ROOT / "analysis" / "outputs3")

# ============ 模型配置 ============
MODEL = "ViT-B-16"

# ============ 设备配置 ============
DEVICE = "cuda"  # 如果没有 GPU，改为 "cpu"
BATCH_SIZE = 32
NUM_WORKERS = 6

# ============ Fisher 计算配置 ============
FISHER_TYPE = "true"  # 可选: "true", "empirical", "squared_grad"
NUM_MC_SAMPLES = 1
FISHER_SAMPLES = 1000  # 用于计算 Fisher 的样本数
SEED = 0

# ============ Module 2 配置 ============
N_PERMUTATIONS = 1000  # 置换检验次数

# ============ Module 3 配置 ============
NUM_SEEDS = 3  # 消融实验的随机种子数

# ============ 任务对配置 ============
# 可用的任务对列表
AVAILABLE_TASK_PAIRS = [
    # ==========================================
    # 1. 之前已验证的优秀组合 (保留并补齐双向)
    # ==========================================
    ("SUN397", "Cars"),
    ("Cars", "SUN397"),
    
    ("GTSRB", "MNIST"),
    ("MNIST", "GTSRB"),  # 注：MNIST作为Task A可能会遇到天花板，但留作对比极其有价值
    
    ("DTD", "RESISC45"),
    ("RESISC45", "DTD"),

    # ==========================================
    # 2. 新增的黄金组合：领域冲突极致测试
    # ==========================================
    ("Cars", "PCAM"),
    ("PCAM", "Cars"),

    # ==========================================
    # 3. 新增的黄金组合：细粒度特征抢夺战
    # ==========================================
    ("Food101", "Flowers102"),
    ("Flowers102", "Food101"),

    # ==========================================
    # 4. 新增的黄金组合：几何与纹理的碰撞
    # ==========================================
    ("GTSRB", "DTD"),
    ("DTD", "GTSRB"),

    # ==========================================
    # 5. 新增的黄金组合：高级场景 vs 低级对象
    # ==========================================
    ("SUN397", "OxfordIIITPet"),
    ("OxfordIIITPet", "SUN397"),

    # ==========================================
    # 6. 新增的黄金组合：真实自然 vs 符号渲染
    # ==========================================
    ("CIFAR100", "RenderedSST2"),
    ("RenderedSST2", "CIFAR100"),
]

# 默认任务对（用于单次运行）
DEFAULT_TASK_A = "SUN397"
DEFAULT_TASK_B = "Cars"


def get_config():
    """返回配置字典"""
    return {
        "model_location": MODEL_LOCATION,
        "data_location": DATA_LOCATION,
        "output_root": OUTPUT_ROOT,
        "model": MODEL,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "fisher_type": FISHER_TYPE,
        "num_mc_samples": NUM_MC_SAMPLES,
        "fisher_samples": FISHER_SAMPLES,
        "seed": SEED,
        "n_permutations": N_PERMUTATIONS,
        "num_seeds": NUM_SEEDS,
    }


def print_config():
    """打印当前配置"""
    print("=" * 60)
    print("PACT Insight 实验配置")
    print("=" * 60)
    print(f"模型检查点路径: {MODEL_LOCATION}")
    print(f"数据集路径: {DATA_LOCATION}")
    print(f"输出路径: {OUTPUT_ROOT}")
    print(f"模型: {MODEL}")
    print(f"设备: {DEVICE}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"Fisher 类型: {FISHER_TYPE}")
    print(f"Fisher 样本数: {FISHER_SAMPLES}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
