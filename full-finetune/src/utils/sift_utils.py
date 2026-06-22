"""
SIFT: Shared-space Interference Filtering for Tasks

简化版PACT算法：直接从任务向量中清洗掉与预训练top-K子空间冲突的分量，
不再提取任务向量自身的top-k方向。

核心操作: Δ_cleaned = Δ - Δ @ V_pre_K @ V_pre_K^T

与PACT的关键区别:
- PACT: 提取预训练top-K + 任务top-k → 计算隐式依赖空间 → 跨任务正交过滤
- SIFT: 仅提取预训练top-K → 直接投影清洗每个任务向量在共享骨架上的分量
"""

import torch


def sift_filter(Delta_t, V_pre_K):
    """
    SIFT核心操作：从任务向量中移除其在预训练核心空间上的投影分量

    Args:
        Delta_t: 任务参数更新量 (d_out, d_in)
        V_pre_K: 预训练核心特征子空间基底 (d_in, K)

    Returns:
        Delta_filtered: 清洗后的任务更新量 (d_out, d_in)
    """
    return Delta_t - Delta_t @ V_pre_K @ V_pre_K.t()


def extract_pretrained_core_space_fixed(W_pre, K_ratio):
    """
    基于固定比例提取预训练核心空间（与pact_utils中相同逻辑，独立实现避免耦合）

    Args:
        W_pre: 预训练权重矩阵 (d_out, d_in)
        K_ratio: 预训练核心空间比例

    Returns:
        V_pre_K: 预训练核心特征子空间基底 (d_in, K)
        K: 保留维度数
    """
    _, S, V = torch.linalg.svd(W_pre, full_matrices=False)
    total_dims = len(S)
    K = max(1, int(total_dims * K_ratio))
    V_pre_K = V[:K, :].t()  # (d_in, K)
    return V_pre_K, K


def extract_pretrained_core_space_adaptive(W_pre, tau_pre):
    """
    基于能量阈值自适应提取预训练核心空间

    Args:
        W_pre: 预训练权重矩阵 (d_out, d_in)
        tau_pre: 预训练能量阈值

    Returns:
        V_pre_K: 预训练核心特征子空间基底 (d_in, K)
        K: 自适应确定的保留维度
    """
    _, S, V = torch.linalg.svd(W_pre, full_matrices=False)
    energy = S ** 2
    total_energy = energy.sum()
    cumulative_energy = torch.cumsum(energy, dim=0)
    energy_ratio = cumulative_energy / total_energy

    indices = (energy_ratio >= tau_pre).nonzero(as_tuple=True)[0]
    if len(indices) > 0:
        K = int(indices[0].item()) + 1
    else:
        K = len(S)

    V_pre_K = V[:K, :].t()  # (d_in, K)
    return V_pre_K, K


def sift_ta_merge(task_vectors, pretrained_vector, config):
    """
    SIFT-TA: SIFT过滤 + Task Arithmetic融合

    Args:
        task_vectors: 任务向量列表
        pretrained_vector: 预训练模型向量
        config: 配置对象

    Returns:
        new_vector: 融合后的向量字典
    """
    device = config.device
    rank_selection = getattr(config.method, 'rank_selection', 'fixed')

    if rank_selection == 'adaptive':
        tau_pre = getattr(config.method, 'tau_pre', 0.85)
        print(f"Computing SIFT filtering with adaptive rank (tau_pre={tau_pre})...")
    else:
        K_ratio = getattr(config.method, 'K_ratio', 0.8)
        print(f"Computing SIFT filtering with fixed rank (K_ratio={K_ratio})...")

    filtered_task_vectors = []

    with torch.no_grad():
        for task_idx, task_vector in enumerate(task_vectors):
            filtered_vector_dict = {}

            for key in task_vector.vector:
                W_pre = pretrained_vector.vector[key].to(device)
                Delta_t = task_vector.vector[key].to(device)

                is_2d_matrix = (
                    len(Delta_t.shape) == 2 and "text_projection" not in key
                )

                if not is_2d_matrix:
                    filtered_vector_dict[key] = Delta_t.cpu()
                    del Delta_t, W_pre
                    continue

                # 提取预训练核心空间
                if rank_selection == 'adaptive':
                    V_pre_K, K = extract_pretrained_core_space_adaptive(
                        W_pre, tau_pre
                    )
                else:
                    V_pre_K, K = extract_pretrained_core_space_fixed(
                        W_pre, K_ratio
                    )

                if task_idx == 0:
                    print(f"  Layer {key}: K={K} ({rank_selection})")

                del W_pre

                # 直接投影清洗
                Delta_filtered = sift_filter(Delta_t, V_pre_K)
                del Delta_t, V_pre_K

                filtered_vector_dict[key] = Delta_filtered.cpu()
                del Delta_filtered

            torch.cuda.empty_cache()
            filtered_task_vectors.append(
                type(task_vector)(
                    model_name=task_vector.model_name,
                    vector=filtered_vector_dict,
                )
            )

    # Task Arithmetic: 直接求平均
    print("Applying Task Arithmetic to SIFT-filtered task vectors...")
    new_vector = {}
    with torch.no_grad():
        for key in filtered_task_vectors[0].vector:
            tvs = [tv.vector[key].to(device) for tv in filtered_task_vectors]
            new_vector[key] = sum(tvs) / len(tvs)
            del tvs
        torch.cuda.empty_cache()

    return new_vector


def sift_isoc_merge(task_vectors, pretrained_vector, config):
    """
    SIFT-ISOC: SIFT过滤 + ISOC融合

    Args:
        task_vectors: 任务向量列表
        pretrained_vector: 预训练模型向量
        config: 配置对象

    Returns:
        new_vector: 融合后的向量字典
    """
    device = config.device
    rank_selection = getattr(config.method, 'rank_selection', 'fixed')

    if rank_selection == 'adaptive':
        tau_pre = getattr(config.method, 'tau_pre', 0.85)
        print(f"Computing SIFT filtering with adaptive rank (tau_pre={tau_pre})...")
    else:
        K_ratio = getattr(config.method, 'K_ratio', 0.8)
        print(f"Computing SIFT filtering with fixed rank (K_ratio={K_ratio})...")

    filtered_task_vectors = []

    with torch.no_grad():
        for task_idx, task_vector in enumerate(task_vectors):
            filtered_vector_dict = {}

            for key in task_vector.vector:
                W_pre = pretrained_vector.vector[key].to(device)
                Delta_t = task_vector.vector[key].to(device)

                is_2d_matrix = (
                    len(Delta_t.shape) == 2 and "text_projection" not in key
                )

                if not is_2d_matrix:
                    filtered_vector_dict[key] = Delta_t.cpu()
                    del Delta_t, W_pre
                    continue

                if rank_selection == 'adaptive':
                    V_pre_K, K = extract_pretrained_core_space_adaptive(
                        W_pre, tau_pre
                    )
                else:
                    V_pre_K, K = extract_pretrained_core_space_fixed(
                        W_pre, K_ratio
                    )

                if task_idx == 0:
                    print(f"  Layer {key}: K={K} ({rank_selection})")

                del W_pre

                Delta_filtered = sift_filter(Delta_t, V_pre_K)
                del Delta_t, V_pre_K

                filtered_vector_dict[key] = Delta_filtered.cpu()
                del Delta_filtered

            torch.cuda.empty_cache()
            filtered_task_vectors.append(
                type(task_vector)(
                    model_name=task_vector.model_name,
                    vector=filtered_vector_dict,
                )
            )

    # ISOC融合：SVD + 奇异值均匀化
    print("Applying ISOC merging to SIFT-filtered task vectors...")
    new_vector = {}
    with torch.no_grad():
        for key in filtered_task_vectors[0].vector:
            tvs = [tv.vector[key].to(device) for tv in filtered_task_vectors]

            is_2d_matrix = (
                len(filtered_task_vectors[0].vector[key].shape) == 2
                and "text_projection" not in key
            )

            if not is_2d_matrix:
                new_vector[key] = sum(tvs) / len(tvs)
                del tvs
            else:
                delta_sum = sum(tvs)
                del tvs

                U, S, V = torch.linalg.svd(delta_sum, full_matrices=False)
                del delta_sum

                S_mean = torch.ones_like(S) * S.mean()
                del S

                new_vector[key] = torch.linalg.multi_dot(
                    (U, torch.diag(S_mean), V)
                )
                del U, S_mean, V

        torch.cuda.empty_cache()

    return new_vector
