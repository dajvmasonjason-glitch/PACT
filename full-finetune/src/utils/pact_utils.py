import torch


def select_pact_layer_keys(task_vector, ratio, include_patterns=None, exclude_patterns=None):
    """
    选择 PACT 过滤要作用的 2D 矩阵 key 集合（取最靠前的 ratio 比例）。

    机理说明：
      - 这里只决定「PACT 过滤（步骤 1~4）」作用在哪些 2D 矩阵上；
        步骤 5 的融合骨干（TA/ISOC/TSVM）仍对全部 2D 矩阵生效。
      - 未被选中的 2D 矩阵不会写入 implicit_spaces，步骤 4 会自动跳过其过滤
        （依赖 `if key not in implicit_spaces: continue`），其 Δ 原样进入融合。
      - 顺序按 state_dict 插入顺序（即模型定义顺序，浅层 → 深层）。
      - include_patterns: 非 None 时，只保留 key 中包含任一 pattern 的层
      - exclude_patterns: 非 None 时，排除 key 中包含任一 pattern 的层
      - include 先于 exclude 应用；ratio 截取在模式过滤之后进行

    Args:
        task_vector: 任一任务向量（用其 key 顺序与形状确定 2D 矩阵集合）
        ratio: PACT 作用的前段比例，1.0 表示全部（等价于原始算法）
        include_patterns: 可选，白名单模式列表（如 ["attn.in_proj", "mlp.c_fc"]）
        exclude_patterns: 可选，黑名单模式列表（如 ["attn.out_proj", "mlp.c_proj"]）

    Returns:
        selected_keys: 需要做 PACT 过滤的 key 集合 (set)
        num_2d: 模型中可作用的 2D 矩阵总数
        cutoff: 实际被选中的 2D 矩阵个数
    """
    keys_2d = [
        k for k, v in task_vector.vector.items()
        if len(v.shape) == 2 and "text_projection" not in k
    ]
    total_2d = len(keys_2d)

    # 按名称模式过滤（include 先于 exclude）
    if include_patterns is not None:
        keys_2d = [k for k in keys_2d if any(p in k for p in include_patterns)]
    if exclude_patterns is not None:
        keys_2d = [k for k in keys_2d if not any(p in k for p in exclude_patterns)]

    num_2d = len(keys_2d)
    cutoff = int(round(num_2d * ratio))
    cutoff = max(0, min(num_2d, cutoff))
    selected_keys = set(keys_2d[:cutoff])
    return selected_keys, total_2d, cutoff


def extract_pretrained_core_space_rand(W_pre, K_ratio, niter=2):
    """
    步骤 1（随机SVD固定比例版本）: 使用固定比例提取预训练核心空间
    使用 torch.svd_lowrank（随机SVD）替换 torch.linalg.svd（完整SVD）

    Args:
        W_pre: 预训练权重矩阵 (d_out, d_in)
        K_ratio: 预训练核心空间比例，例如 0.8 表示保留 80% 的维度
        niter: 随机SVD的幂迭代次数（通常2已足够）

    Returns:
        V_pre_K: 预训练核心特征子空间基底 (d_in, K)
        K: 确定的保留维度
    """
    dim = min(W_pre.shape[0], W_pre.shape[1])
    K = max(1, int(dim * K_ratio))

    # 使用 torch.svd_lowrank（官方随机SVD）
    _, _, V = torch.svd_lowrank(W_pre, q=K, niter=niter)
    # V 形状为 (d_in, K)，可以直接使用

    return V, K


def extract_task_explicit_space_rand(Delta_t, k, niter=2):
    """
    步骤 2（随机SVD固定维度版本）: 使用固定维度提取任务显式变化空间
    使用 torch.svd_lowrank（随机SVD）替换 torch.linalg.svd（完整SVD）

    Args:
        Delta_t: 任务参数更新量 (d_out, d_in)
        k: 每个任务分配的维度数
        niter: 随机SVD的幂迭代次数

    Returns:
        V_t_k: 任务显式变化子空间基底 (d_in, k)
        k: 确定的保留维度
    """
    dim = min(Delta_t.shape[0], Delta_t.shape[1])
    k = min(k, dim)  # 不超过矩阵的秩

    # 使用 torch.svd_lowrank（官方随机SVD）
    _, _, V = torch.svd_lowrank(Delta_t, q=k, niter=niter)
    # V 形状为 (d_in, k)，可以直接使用

    return V, k


def extract_pretrained_core_space(W_pre, tau_pre):
    """
    步骤 1: 基于能量阈值自适应提取预训练核心空间
    对预训练权重矩阵进行 SVD，根据能量阈值 tau_pre 自适应确定保留维度 K

    Args:
        W_pre: 预训练权重矩阵 (d_out, d_in)
        tau_pre: 预训练能量阈值，例如 0.85

    Returns:
        V_pre_K: 预训练核心特征子空间基底 (d_in, K)
        K: 自适应确定的保留维度
    """
    _, S, V = torch.linalg.svd(W_pre, full_matrices=False)

    # 计算累计能量占比
    energy = S ** 2
    total_energy = energy.sum()
    cumulative_energy = torch.cumsum(energy, dim=0)
    energy_ratio = cumulative_energy / total_energy

    # 找到满足能量阈值的最小秩 K
    indices = (energy_ratio >= tau_pre).nonzero(as_tuple=True)[0]
    if len(indices) > 0:
        K = int(indices[0].item()) + 1
    else:
        # 如果没有满足阈值的，保留所有维度
        K = len(S)

    V_pre_K = V[:K, :].t()  # (d_in, K)
    return V_pre_K, K


def extract_task_explicit_space(Delta_t, tau_task):
    """
    步骤 2: 基于能量阈值自适应提取任务显式变化空间
    对任务参数更新量进行 SVD，根据能量阈值 tau_task 自适应确定保留维度 k

    Args:
        Delta_t: 任务参数更新量 (d_out, d_in)
        tau_task: 任务能量阈值，例如 0.95

    Returns:
        V_t_k: 任务显式变化子空间基底 (d_in, k)
        k: 自适应确定的保留维度
    """
    _, S, V = torch.linalg.svd(Delta_t, full_matrices=False)

    # 计算累计能量占比
    energy = S ** 2
    total_energy = energy.sum()
    cumulative_energy = torch.cumsum(energy, dim=0)
    energy_ratio = cumulative_energy / total_energy

    # 找到满足能量阈值的最小秩 k
    indices = (energy_ratio >= tau_task).nonzero(as_tuple=True)[0]
    if len(indices) > 0:
        k = int(indices[0].item()) + 1
    else:
        # 如果没有满足阈值的，保留所有维度
        k = len(S)

    V_t_k = V[:k, :].t()  # (d_in, k)
    return V_t_k, k


def extract_pretrained_core_space_fixed(W_pre, K_ratio):
    """
    步骤 1（固定比例版本）: 基于固定比例提取预训练核心空间
    对预训练权重矩阵进行 SVD，保留固定比例的维度

    Args:
        W_pre: 预训练权重矩阵 (d_out, d_in)
        K_ratio: 预训练核心空间比例，例如 0.8 表示保留 80% 的维度

    Returns:
        V_pre_K: 预训练核心特征子空间基底 (d_in, K)
        K: 确定的保留维度
    """
    _, S, V = torch.linalg.svd(W_pre, full_matrices=False)

    # 根据比例计算保留维度
    total_dims = len(S)
    K = int(total_dims * K_ratio)
    K = max(1, K)  # 至少保留 1 个维度

    V_pre_K = V[:K, :].t()  # (d_in, K)
    return V_pre_K, K


def extract_task_explicit_space_fixed(Delta_t, k_per_task):
    """
    步骤 2（固定维度版本）: 基于固定维度提取任务显式变化空间
    对任务参数更新量进行 SVD，为每个任务分配固定维度

    Args:
        Delta_t: 任务参数更新量 (d_out, d_in)
        k_per_task: 每个任务分配的维度数

    Returns:
        V_t_k: 任务显式变化子空间基底 (d_in, k)
        k: 确定的保留维度
    """
    _, S, V = torch.linalg.svd(Delta_t, full_matrices=False)

    # 使用固定的每任务维度
    k = min(k_per_task, len(S))  # 不超过矩阵的秩

    V_t_k = V[:k, :].t()  # (d_in, k)
    return V_t_k, k


def compute_implicit_reliance_space(V_pre_K, V_t_k):
    """
    步骤 3: 计算任务的隐式依赖空间
    将预训练核心基底投影到任务显式变化空间的正交补空间

    Args:
        V_pre_K: 预训练核心特征子空间基底 (d_in, K)
        V_t_k: 任务显式变化子空间基底 (d_in, k)

    Returns:
        V_rel_t: 任务隐式依赖空间基底 (d_in, r), r <= K
    """
    # Avoid materializing the full d_in×d_in projection matrix.
    # P_t_perp @ V_pre_K = V_pre_K - V_t_k @ (V_t_k.T @ V_pre_K)
    V_rel_t_tilde = V_pre_K - V_t_k @ (V_t_k.t() @ V_pre_K)

    # 检查投影结果的范数，如果太小说明隐式依赖空间接近零
    norm = torch.linalg.norm(V_rel_t_tilde, ord='fro')
    if norm < 1e-6:
        # 隐式依赖空间接近零，返回空矩阵
        return torch.zeros((V_pre_K.shape[0], 0), device=V_pre_K.device, dtype=V_pre_K.dtype)

    # 正交化 (QR 分解)
    V_rel_t, _ = torch.linalg.qr(V_rel_t_tilde)

    return V_rel_t


def orthogonal_core_filtering(Delta_j, V_protect_j):
    """
    步骤 4: 无干涉正交过滤
    对任务更新矩阵进行过滤，剔除其在保护空间上的分量

    Args:
        Delta_j: 任务 j 的更新矩阵 (d_out, d_in)
        V_protect_j: 任务 j 需要避让的全局保护空间基底 (d_in, p)

    Returns:
        Delta_j_filtered: 过滤后的更新矩阵 (d_out, d_in)
    """
    # Delta_j^filtered = Delta_j - Delta_j V_protect^(j) (V_protect^(j))^T
    Delta_j_filtered = Delta_j - Delta_j @ V_protect_j @ V_protect_j.t()
    return Delta_j_filtered


def pact_isoc_merge(task_vectors, pretrained_vector, config):
    """
    PACT-ISOC 融合方法
    先使用 PACT 方法过滤任务向量，然后使用 ISOC 进行融合
    支持两种秩选择机制：能量阈值（adaptive）或固定比例（fixed）

    Args:
        task_vectors: 任务向量列表
        pretrained_vector: 预训练模型向量
        config: 配置对象

    Returns:
        new_vector: 融合后的向量字典
    """
    device = config.device

    # 选择秩选择策略
    rank_selection = getattr(config.method, 'rank_selection', 'adaptive')  # 'adaptive' 或 'fixed'

    if rank_selection == 'adaptive':
        # 能量阈值方法
        tau_pre = getattr(config.method, 'tau_pre', 0.85)
        tau_task = getattr(config.method, 'tau_task', 0.95)
        print(f"Computing PACT filtering with adaptive rank selection "
              f"(tau_pre={tau_pre}, tau_task={tau_task})...")
    else:
        # 固定比例方法（参考 ISO-CTS）
        K_ratio = getattr(config.method, 'K_ratio', 0.8)  # 预训练核心空间占比
        k_per_task = getattr(config.method, 'k_per_task', 10)  # 每个任务的维度
        print(f"Computing PACT filtering with fixed rank selection "
              f"(K_ratio={K_ratio}, k_per_task={k_per_task})...")

    # PACT 作用范围：只对最靠前的 pact_layer_ratio 比例的 2D 矩阵做过滤，
    # 1.0（默认）表示作用于全部 2D 矩阵，等价于原始算法。
    # pact_include_patterns / pact_exclude_patterns: 按层类型名称选择性过滤
    pact_layer_ratio = getattr(config.method, 'pact_layer_ratio', 1.0)
    pact_include = getattr(config.method, 'pact_include_patterns', None)
    pact_exclude = getattr(config.method, 'pact_exclude_patterns', None)
    pact_keys, num_2d, cutoff = select_pact_layer_keys(
        task_vectors[0], pact_layer_ratio,
        include_patterns=pact_include, exclude_patterns=pact_exclude)
    print(f"PACT layer scope: applying to {cutoff}/{num_2d} 2D matrices "
          f"(pact_layer_ratio={pact_layer_ratio})")

    with torch.no_grad():
        # 首先收集所有任务的隐式依赖空间
        implicit_spaces = {}
        filtered_task_vectors = []

        for task_idx, task_vector in enumerate(task_vectors):
            filtered_vector_dict = {}

            for key in task_vector.vector:
                # W_pre 只在 SVD 期间需要，用完即释放，不长期驻留 GPU
                W_pre = pretrained_vector.vector[key].to(device)
                Delta_t = task_vector.vector[key].to(device)

                is_2d_matrix = (
                    len(Delta_t.shape) == 2 and "text_projection" not in key
                )

                # 不在 PACT 作用范围内的 2D 矩阵：当作「不过滤」处理，
                # 不写入 implicit_spaces，步骤 4 会自动跳过，Δ 原样进入步骤 5 融合。
                if is_2d_matrix and key not in pact_keys:
                    filtered_vector_dict[key] = Delta_t.cpu()
                    del Delta_t, W_pre
                    continue

                if not is_2d_matrix:
                    # 非 2D 张量存 CPU，步骤 5 用到时再搬回 GPU
                    filtered_vector_dict[key] = Delta_t.cpu()
                    del Delta_t, W_pre
                    continue

                # 步骤 1 & 2: 根据策略提取空间基底
                if rank_selection == 'adaptive':
                    V_pre_K, K = extract_pretrained_core_space(W_pre, tau_pre)
                    V_t_k, k = extract_task_explicit_space(Delta_t, tau_task)
                    if task_idx == 0:
                        print(f"  Layer {key}: K={K}, k={k} (adaptive)")
                else:
                    V_pre_K, K = extract_pretrained_core_space_fixed(W_pre, K_ratio)
                    V_t_k, k = extract_task_explicit_space_fixed(Delta_t, k_per_task)
                    if task_idx == 0:
                        print(f"  Layer {key}: K={K}, k={k} (fixed)")

                # W_pre 已不再需要，立即释放
                del W_pre

                # 步骤 3: 计算任务的隐式依赖空间
                V_rel_t = compute_implicit_reliance_space(V_pre_K, V_t_k)
                del V_pre_K, V_t_k

                # 隐式依赖空间存 CPU，步骤 4 用到时再搬回 GPU
                if key not in implicit_spaces:
                    implicit_spaces[key] = []
                implicit_spaces[key].append(V_rel_t.cpu())
                del V_rel_t

                # Delta_t 存 CPU，步骤 4 用到时再搬回 GPU
                filtered_vector_dict[key] = Delta_t.cpu()
                del Delta_t

            torch.cuda.empty_cache()

            # 创建过滤后的任务向量，保持与原始任务向量相同的类型和 model_name
            filtered_task_vectors.append(
                type(task_vector)(model_name=task_vector.model_name, vector=filtered_vector_dict)
            )

        # 步骤 4: 对每个任务进行正交过滤
        for task_idx, task_vector in enumerate(filtered_task_vectors):
            for key in task_vector.vector:
                is_2d_matrix = (
                    len(task_vector.vector[key].shape) == 2 and "text_projection" not in key
                )

                if not is_2d_matrix:
                    continue

                # 检查该 key 是否在 implicit_spaces 中（可能在第一轮被跳过）
                if key not in implicit_spaces:
                    continue

                # 构建保护空间：所有其他任务的隐式依赖空间的并集（从 CPU 取出搬到 GPU）
                protect_spaces = []
                for other_idx in range(len(filtered_task_vectors)):
                    if other_idx != task_idx:
                        V_rel = implicit_spaces[key][other_idx]
                        if V_rel.shape[1] > 0:
                            protect_spaces.append(V_rel.to(device))

                if len(protect_spaces) > 0:
                    # 合并所有保护空间并正交化
                    V_protect_j = torch.cat(protect_spaces, dim=1)
                    del protect_spaces
                    V_protect_j, _ = torch.linalg.qr(V_protect_j)

                    # Delta_j 从 CPU 搬到 GPU，过滤后结果存回 CPU
                    Delta_j = task_vector.vector[key].to(device)
                    Delta_j_filtered = orthogonal_core_filtering(Delta_j, V_protect_j)
                    del Delta_j, V_protect_j
                    task_vector.vector[key] = Delta_j_filtered.cpu()
                    del Delta_j_filtered

            torch.cuda.empty_cache()

    # 步骤 5: 使用原始 ISOC 算法进行融合
    print("Applying original ISOC merging to filtered task vectors...")

    new_vector = {}
    with torch.no_grad():
        for key in filtered_task_vectors[0].vector:
            tvs = [tv.vector[key].to(device) for tv in filtered_task_vectors]

            is_2d_matrix = (
                len(filtered_task_vectors[0].vector[key].shape) == 2
                and "text_projection" not in key
            )

            if not is_2d_matrix:
                # 1D 向量：简单平均
                new_vector[key] = sum(tvs) / len(tvs)
                del tvs
            else:
                # 2D 矩阵：应用原始 ISOC 算法
                delta_sum = sum(tvs)
                del tvs

                # SVD 分解
                U, S, V = torch.linalg.svd(delta_sum, full_matrices=False)
                del delta_sum

                # 奇异值均匀化：用平均值替换所有奇异值
                S_mean = torch.ones_like(S) * S.mean()
                del S

                # 重构矩阵
                new_vector[key] = torch.linalg.multi_dot((U, torch.diag(S_mean), V))
                del U, S_mean, V

        torch.cuda.empty_cache()

    return new_vector


def pact_isocts_merge(task_vectors, pretrained_vector, config):
    """
    PACT-ISOCTS 融合方法
    先使用 PACT 方法过滤任务向量，然后使用 ISO-CTS 进行融合
    支持两种秩选择机制：能量阈值（adaptive）或固定比例（fixed）

    Args:
        task_vectors: 任务向量列表
        pretrained_vector: 预训练模型向量
        config: 配置对象

    Returns:
        new_vector: 融合后的向量字典
    """
    device = config.device

    # 选择秩选择策略
    rank_selection = getattr(config.method, 'rank_selection', 'adaptive')

    if rank_selection == 'adaptive':
        # 能量阈值方法
        tau_pre = getattr(config.method, 'tau_pre', 0.85)
        tau_task = getattr(config.method, 'tau_task', 0.95)
        print(f"Computing PACT filtering with adaptive rank selection "
              f"(tau_pre={tau_pre}, tau_task={tau_task})...")
    else:
        # 固定比例方法
        K_ratio = getattr(config.method, 'K_ratio', 0.8)
        k_per_task = getattr(config.method, 'k_per_task', 10)
        print(f"Computing PACT filtering with fixed rank selection "
              f"(K_ratio={K_ratio}, k_per_task={k_per_task})...")

    with torch.no_grad():
        # 首先收集所有任务的隐式依赖空间
        implicit_spaces = {}
        filtered_task_vectors = []

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

                # 步骤 1 & 2: 根据策略提取空间基底
                if rank_selection == 'adaptive':
                    V_pre_K, K = extract_pretrained_core_space(W_pre, tau_pre)
                    V_t_k, k = extract_task_explicit_space(Delta_t, tau_task)
                    if task_idx == 0:
                        print(f"  Layer {key}: K={K}, k={k} (adaptive)")
                else:
                    V_pre_K, K = extract_pretrained_core_space_fixed(W_pre, K_ratio)
                    V_t_k, k = extract_task_explicit_space_fixed(Delta_t, k_per_task)
                    if task_idx == 0:
                        print(f"  Layer {key}: K={K}, k={k} (fixed)")

                del W_pre

                # 步骤 3: 计算任务的隐式依赖空间
                V_rel_t = compute_implicit_reliance_space(V_pre_K, V_t_k)
                del V_pre_K, V_t_k

                if key not in implicit_spaces:
                    implicit_spaces[key] = []
                implicit_spaces[key].append(V_rel_t.cpu())
                del V_rel_t

                filtered_vector_dict[key] = Delta_t.cpu()
                del Delta_t

            torch.cuda.empty_cache()

            filtered_task_vectors.append(
                type(task_vector)(model_name=task_vector.model_name, vector=filtered_vector_dict)
            )

        # 步骤 4: 对每个任务进行正交过滤
        for task_idx, task_vector in enumerate(filtered_task_vectors):
            for key in task_vector.vector:
                is_2d_matrix = (
                    len(task_vector.vector[key].shape) == 2 and "text_projection" not in key
                )

                if not is_2d_matrix:
                    continue

                if key not in implicit_spaces:
                    continue

                # 构建保护空间：所有其他任务的隐式依赖空间的并集
                protect_spaces = []
                for other_idx in range(len(filtered_task_vectors)):
                    if other_idx != task_idx:
                        V_rel = implicit_spaces[key][other_idx]
                        if V_rel.shape[1] > 0:
                            protect_spaces.append(V_rel.to(device))

                if len(protect_spaces) > 0:
                    V_protect_j = torch.cat(protect_spaces, dim=1)
                    del protect_spaces
                    V_protect_j, _ = torch.linalg.qr(V_protect_j)

                    Delta_j = task_vector.vector[key].to(device)
                    Delta_j_filtered = orthogonal_core_filtering(Delta_j, V_protect_j)
                    del Delta_j, V_protect_j
                    task_vector.vector[key] = Delta_j_filtered.cpu()
                    del Delta_j_filtered

            torch.cuda.empty_cache()

    # 步骤 5: 使用 ISO-CTS 算法进行融合
    print("Applying ISO-CTS merging to filtered task vectors...")

    new_vector = {}
    common_space_fraction = getattr(config.method, 'common_space_fraction', 0.5)

    with torch.no_grad():
        for key in filtered_task_vectors[0].vector:
            shape_ = filtered_task_vectors[0].vector[key].shape

            is_2d_matrix = (len(shape_) == 2) and ("text_projection" not in key)
            if not is_2d_matrix:
                # 1D 向量：简单平均
                print(f"Combining by avg {key}...")
                tvs = [tv.vector[key].to(device) for tv in filtered_task_vectors]
                new_vector[key] = sum(tvs) / len(tvs)
                del tvs
                continue

            print(f"Computing common space using sum for {key}...")
            combined_w = sum([tv.vector[key].to(device) for tv in filtered_task_vectors])

            # 计算共同空间大小（确保任务特定空间可以均分）
            common_space_index_s = int(min(shape_) * common_space_fraction)
            _task_specific_total_space_index_s = round((min(shape_) - common_space_index_s) / len(config.DATASETS)) * len(config.DATASETS)
            common_space_index_s = min(shape_) - _task_specific_total_space_index_s

            u, s, v = torch.linalg.svd(combined_w, full_matrices=False)
            shared_space_u = u[:, :common_space_index_s]
            shared_space_s = s[:common_space_index_s]
            shared_space_v = v[:common_space_index_s, :]

            # 计算任务特定空间
            n_dims_per_task = int((min(shape_) - common_space_index_s) / len(config.DATASETS))
            for i, task_vector in enumerate(filtered_task_vectors):
                w = task_vector.vector[key].to(device)

                # 计算投影到任务特定空间以去除共同空间
                w_ts = w - shared_space_u @ shared_space_u.T @ w
                u_ts, s_ts, v_ts = torch.linalg.svd(w_ts, full_matrices=False)

                if i == 0:
                    final_u = torch.zeros_like(u_ts, device=device)
                    final_s = torch.zeros_like(s_ts, device=device)
                    final_v = torch.zeros_like(v_ts, device=device)

                final_u[:, i * n_dims_per_task : (i + 1) * n_dims_per_task] = u_ts[:, :n_dims_per_task]
                final_s[i * n_dims_per_task : (i + 1) * n_dims_per_task] = s_ts[:n_dims_per_task]
                final_v[i * n_dims_per_task : (i + 1) * n_dims_per_task, :] = v_ts[:n_dims_per_task, :]

            # 将共同空间的分量放入最终矩阵的末尾
            final_u[:, len(config.DATASETS) * n_dims_per_task : len(config.DATASETS) * n_dims_per_task + common_space_index_s] = shared_space_u
            final_s[len(config.DATASETS) * n_dims_per_task : len(config.DATASETS) * n_dims_per_task + common_space_index_s] = shared_space_s
            final_v[len(config.DATASETS) * n_dims_per_task : len(config.DATASETS) * n_dims_per_task + common_space_index_s, :] = shared_space_v

            # 正交化 final_u 和 final_v
            # 使用正交 Procrustes 方法：对矩阵 M 做 SVD 得到 U*S*V^T，则 U*V^T 是最接近 M 的正交矩阵
            u_final_u, _, v_final_u = torch.linalg.svd(final_u, full_matrices=False)
            u_final_v, _, v_final_v = torch.linalg.svd(final_v, full_matrices=False)
            final_u = u_final_u @ v_final_u
            final_v = u_final_v @ v_final_v

            # 奇异值均匀化
            final_s = torch.ones_like(final_s) * final_s.mean()

            new_vector[key] = torch.linalg.multi_dot(
                (
                    final_u,
                    torch.diag(final_s),
                    final_v,
                )
            )

            del combined_w, u, s, v, shared_space_u, shared_space_s, shared_space_v
            del final_u, final_s, final_v, u_final_u, v_final_u, u_final_v, v_final_v

        torch.cuda.empty_cache()

    return new_vector


def pact_ta_merge(task_vectors, pretrained_vector, config):
    """
    PACT-TA 融合方法
    先使用 PACT 方法过滤任务向量，然后直接求和（Task Arithmetic）
    支持两种秩选择机制：能量阈值（adaptive）或固定比例（fixed）

    Args:
        task_vectors: 任务向量列表
        pretrained_vector: 预训练模型向量
        config: 配置对象

    Returns:
        new_vector: 融合后的向量字典
    """
    device = config.device

    # 选择秩选择策略
    rank_selection = getattr(config.method, 'rank_selection', 'adaptive')

    if rank_selection == 'adaptive':
        # 能量阈值方法
        tau_pre = getattr(config.method, 'tau_pre', 0.85)
        tau_task = getattr(config.method, 'tau_task', 0.95)
        print(f"Computing PACT filtering with adaptive rank selection "
              f"(tau_pre={tau_pre}, tau_task={tau_task})...")
    else:
        # 固定比例方法
        K_ratio = getattr(config.method, 'K_ratio', 0.8)
        k_per_task = getattr(config.method, 'k_per_task', 10)
        print(f"Computing PACT filtering with fixed rank selection "
              f"(K_ratio={K_ratio}, k_per_task={k_per_task})...")


    # PACT 作用范围：只对最靠前的 pact_layer_ratio 比例的 2D 矩阵做过滤，
    # 1.0（默认）表示作用于全部 2D 矩阵，等价于原始算法。
    # pact_include_patterns / pact_exclude_patterns: 按层类型名称选择性过滤
    pact_layer_ratio = getattr(config.method, 'pact_layer_ratio', 1.0)
    pact_include = getattr(config.method, 'pact_include_patterns', None)
    pact_exclude = getattr(config.method, 'pact_exclude_patterns', None)
    pact_keys, num_2d, cutoff = select_pact_layer_keys(
        task_vectors[0], pact_layer_ratio,
        include_patterns=pact_include, exclude_patterns=pact_exclude)
    print(f"PACT layer scope: applying to {cutoff}/{num_2d} 2D matrices "
          f"(pact_layer_ratio={pact_layer_ratio})")

    with torch.no_grad():
        # 首先收集所有任务的隐式依赖空间
        implicit_spaces = {}
        filtered_task_vectors = []

        for task_idx, task_vector in enumerate(task_vectors):
            filtered_vector_dict = {}

            for key in task_vector.vector:
                W_pre = pretrained_vector.vector[key].to(device)
                Delta_t = task_vector.vector[key].to(device)

                is_2d_matrix = (
                    len(Delta_t.shape) == 2 and "text_projection" not in key
                )


                # 不在 PACT 作用范围内的 2D 矩阵：当作「不过滤」处理，
                # 不写入 implicit_spaces，步骤 4 会自动跳过，Δ 原样进入步骤 5 融合。
                if is_2d_matrix and key not in pact_keys:
                    filtered_vector_dict[key] = Delta_t.cpu()
                    del Delta_t, W_pre
                    continue

                if not is_2d_matrix:
                    filtered_vector_dict[key] = Delta_t.cpu()
                    del Delta_t, W_pre
                    continue

                # 步骤 1 & 2: 根据策略提取空间基底
                if rank_selection == 'adaptive':
                    V_pre_K, K = extract_pretrained_core_space(W_pre, tau_pre)
                    V_t_k, k = extract_task_explicit_space(Delta_t, tau_task)
                    if task_idx == 0:
                        print(f"  Layer {key}: K={K}, k={k} (adaptive)")
                else:
                    V_pre_K, K = extract_pretrained_core_space_fixed(W_pre, K_ratio)
                    V_t_k, k = extract_task_explicit_space_fixed(Delta_t, k_per_task)
                    if task_idx == 0:
                        print(f"  Layer {key}: K={K}, k={k} (fixed)")

                del W_pre

                # 步骤 3: 计算任务的隐式依赖空间
                V_rel_t = compute_implicit_reliance_space(V_pre_K, V_t_k)
                del V_pre_K, V_t_k

                if key not in implicit_spaces:
                    implicit_spaces[key] = []
                implicit_spaces[key].append(V_rel_t.cpu())
                del V_rel_t

                filtered_vector_dict[key] = Delta_t.cpu()
                del Delta_t

            torch.cuda.empty_cache()

            filtered_task_vectors.append(
                type(task_vector)(model_name=task_vector.model_name, vector=filtered_vector_dict)
            )

        # 步骤 4: 对每个任务进行正交过滤
        for task_idx, task_vector in enumerate(filtered_task_vectors):
            for key in task_vector.vector:
                is_2d_matrix = (
                    len(task_vector.vector[key].shape) == 2 and "text_projection" not in key
                )

                if not is_2d_matrix:
                    continue

                if key not in implicit_spaces:
                    continue

                # 构建保护空间：所有其他任务的隐式依赖空间的并集
                protect_spaces = []
                for other_idx in range(len(filtered_task_vectors)):
                    if other_idx != task_idx:
                        V_rel = implicit_spaces[key][other_idx]
                        if V_rel.shape[1] > 0:
                            protect_spaces.append(V_rel.to(device))

                if len(protect_spaces) > 0:
                    V_protect_j = torch.cat(protect_spaces, dim=1)
                    del protect_spaces
                    V_protect_j, _ = torch.linalg.qr(V_protect_j)

                    Delta_j = task_vector.vector[key].to(device)
                    Delta_j_filtered = orthogonal_core_filtering(Delta_j, V_protect_j)
                    del Delta_j, V_protect_j
                    task_vector.vector[key] = Delta_j_filtered.cpu()
                    del Delta_j_filtered

            torch.cuda.empty_cache()

    # 步骤 5: 直接对过滤后的任务向量求和（Task Arithmetic）
    print("Applying Task Arithmetic (direct sum) to filtered task vectors...")

    new_vector = {}
    with torch.no_grad():
        for key in filtered_task_vectors[0].vector:
            tvs = [tv.vector[key].to(device) for tv in filtered_task_vectors]
            # 直接求和
            new_vector[key] = sum(tvs) / len(tvs)
            del tvs

        torch.cuda.empty_cache()

    return new_vector


def pact_tsvm_merge(task_vectors, pretrained_vector, config):
    """
    PACT-TSVM 融合方法
    先使用 PACT 方法过滤任务向量，然后使用 TSVM（Task Singular Vector Merging）进行融合
    支持两种秩选择机制：能量阈值（adaptive）或固定比例（fixed）

    TSVM 融合步骤：
      1) 对每个任务向量的 2D 矩阵进行 SVD，仅保留前 sv_reduction 比例的奇异分量
      2) 将所有任务保留的分量拼接成大矩阵
      3) 对拼接后的 U 和 V 再做一次 SVD 正交化

    Args:
        task_vectors: 任务向量列表
        pretrained_vector: 预训练模型向量
        config: 配置对象

    Returns:
        new_vector: 融合后的向量字典
    """
    device = config.device
    num_tasks = len(task_vectors)

    # 选择秩选择策略
    rank_selection = getattr(config.method, 'rank_selection', 'adaptive')

    if rank_selection == 'adaptive':
        tau_pre = getattr(config.method, 'tau_pre', 0.85)
        tau_task = getattr(config.method, 'tau_task', 0.95)
        print(f"Computing PACT filtering with adaptive rank selection "
              f"(tau_pre={tau_pre}, tau_task={tau_task})...")
    else:
        K_ratio = getattr(config.method, 'K_ratio', 0.8)
        k_per_task = getattr(config.method, 'k_per_task', 10)
        print(f"Computing PACT filtering with fixed rank selection "
              f"(K_ratio={K_ratio}, k_per_task={k_per_task})...")


    # PACT 作用范围：只对最靠前的 pact_layer_ratio 比例的 2D 矩阵做过滤，
    # 1.0（默认）表示作用于全部 2D 矩阵，等价于原始算法。
    # pact_include_patterns / pact_exclude_patterns: 按层类型名称选择性过滤
    pact_layer_ratio = getattr(config.method, 'pact_layer_ratio', 1.0)
    pact_include = getattr(config.method, 'pact_include_patterns', None)
    pact_exclude = getattr(config.method, 'pact_exclude_patterns', None)
    pact_keys, num_2d, cutoff = select_pact_layer_keys(
        task_vectors[0], pact_layer_ratio,
        include_patterns=pact_include, exclude_patterns=pact_exclude)
    print(f"PACT layer scope: applying to {cutoff}/{num_2d} 2D matrices "
          f"(pact_layer_ratio={pact_layer_ratio})")

    with torch.no_grad():
        # === 步骤 1~3: 收集所有任务的隐式依赖空间 ===
        implicit_spaces = {}
        filtered_task_vectors = []

        for task_idx, task_vector in enumerate(task_vectors):
            filtered_vector_dict = {}

            for key in task_vector.vector:
                W_pre = pretrained_vector.vector[key].to(device)
                Delta_t = task_vector.vector[key].to(device)

                is_2d_matrix = (
                    len(Delta_t.shape) == 2 and "text_projection" not in key
                )


                # 不在 PACT 作用范围内的 2D 矩阵：当作「不过滤」处理，
                # 不写入 implicit_spaces，步骤 4 会自动跳过，Δ 原样进入步骤 5 融合。
                if is_2d_matrix and key not in pact_keys:
                    filtered_vector_dict[key] = Delta_t.cpu()
                    del Delta_t, W_pre
                    continue

                if not is_2d_matrix:
                    filtered_vector_dict[key] = Delta_t.cpu()
                    del Delta_t, W_pre
                    continue

                # 步骤 1 & 2: 提取空间基底
                if rank_selection == 'adaptive':
                    V_pre_K, K = extract_pretrained_core_space(W_pre, tau_pre)
                    V_t_k, k = extract_task_explicit_space(Delta_t, tau_task)
                    if task_idx == 0:
                        print(f"  Layer {key}: K={K}, k={k} (adaptive)")
                else:
                    V_pre_K, K = extract_pretrained_core_space_fixed(W_pre, K_ratio)
                    V_t_k, k = extract_task_explicit_space_fixed(Delta_t, k_per_task)
                    if task_idx == 0:
                        print(f"  Layer {key}: K={K}, k={k} (fixed)")

                del W_pre

                # 步骤 3: 计算隐式依赖空间
                V_rel_t = compute_implicit_reliance_space(V_pre_K, V_t_k)
                del V_pre_K, V_t_k

                if key not in implicit_spaces:
                    implicit_spaces[key] = []
                implicit_spaces[key].append(V_rel_t.cpu())
                del V_rel_t

                filtered_vector_dict[key] = Delta_t.cpu()
                del Delta_t

            torch.cuda.empty_cache()

            filtered_task_vectors.append(
                type(task_vector)(model_name=task_vector.model_name, vector=filtered_vector_dict)
            )

        # === 步骤 4: 正交过滤 ===
        for task_idx, task_vector in enumerate(filtered_task_vectors):
            for key in task_vector.vector:
                is_2d_matrix = (
                    len(task_vector.vector[key].shape) == 2 and "text_projection" not in key
                )

                if not is_2d_matrix:
                    continue

                if key not in implicit_spaces:
                    continue

                protect_spaces = []
                for other_idx in range(len(filtered_task_vectors)):
                    if other_idx != task_idx:
                        V_rel = implicit_spaces[key][other_idx]
                        if V_rel.shape[1] > 0:
                            protect_spaces.append(V_rel.to(device))

                if len(protect_spaces) > 0:
                    V_protect_j = torch.cat(protect_spaces, dim=1)
                    del protect_spaces
                    V_protect_j, _ = torch.linalg.qr(V_protect_j)

                    Delta_j = task_vector.vector[key].to(device)
                    Delta_j_filtered = orthogonal_core_filtering(Delta_j, V_protect_j)
                    del Delta_j, V_protect_j
                    task_vector.vector[key] = Delta_j_filtered.cpu()
                    del Delta_j_filtered

            torch.cuda.empty_cache()

    # === 步骤 5: 使用 TSVM 算法对过滤后的任务向量进行融合 ===
    print("Applying TSVM (Task Singular Vector Merging) to filtered task vectors...")

    new_vector = {}

    with torch.no_grad():
        for key in filtered_task_vectors[0].vector:
            shape_ = filtered_task_vectors[0].vector[key].shape

            is_2d_matrix = (len(shape_) == 2) and ("text_projection" not in key)

            if not is_2d_matrix:
                # 1D 向量：增量式平均
                for i, tv in enumerate(filtered_task_vectors):
                    vec = tv.vector[key].to(device)
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)
                    del vec
                torch.cuda.empty_cache()
                continue

            # 2D 矩阵：执行 TSVM 拼接融合
            # 首先计算每个任务应该保留的维度数，确保充分利用所有维度
            total_dims = min(shape_)
            base_dims_per_task = total_dims // num_tasks
            extra_dims = total_dims % num_tasks  # 剩余维度

            for i, tv in enumerate(filtered_task_vectors):
                vec = tv.vector[key].to(device)
                u, s, v = torch.linalg.svd(vec, full_matrices=False)

                if i == 0:
                    print(f"  Computing TSVM for {key}...")
                    sum_u = torch.zeros_like(u, device=device)
                    sum_s = torch.zeros_like(s, device=device)
                    sum_v = torch.zeros_like(v, device=device)

                # 为前 extra_dims 个任务多分配一个维度
                dims_for_this_task = base_dims_per_task + (1 if i < extra_dims else 0)
                dims_for_this_task = max(1, dims_for_this_task)  # 至少保留 1 个分量

                # 计算当前任务在拼接矩阵中的起始位置
                start_idx = i * base_dims_per_task + min(i, extra_dims)
                end_idx = start_idx + dims_for_this_task

                sum_u[:, start_idx:end_idx] = u[:, :dims_for_this_task]
                sum_s[start_idx:end_idx] = s[:dims_for_this_task]
                sum_v[start_idx:end_idx, :] = v[:dims_for_this_task, :]

                del vec, u, s, v

            torch.cuda.empty_cache()

            # 对拼接后的 U 和 V 进行正交化
            # 使用正交 Procrustes 方法：对矩阵 M 做 SVD 得到 U*S*V^T，则 U*V^T 是最接近 M 的正交矩阵
            u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
            u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)

            # 重构矩阵：(U_orth) @ diag(S) @ (V_orth)
            # 其中 U_orth = u_u @ v_u, V_orth = u_v @ v_v 是正交矩阵
            new_vector[key] = torch.linalg.multi_dot(
                (
                    u_u,
                    v_u,
                    torch.diag(sum_s),
                    u_v,
                    v_v,
                )
            )

            del sum_u, sum_s, sum_v, u_u, v_u, u_v, v_v
            torch.cuda.empty_cache()

    return new_vector


def pact_isoc_randsvd_merge(task_vectors, pretrained_vector, config):
    """
    PACT-ISOC 简化版（随机SVD + NS迭代）
    使用随机SVD替换完整SVD进行PACT过滤，ISOC步骤使用Newton-Schulz迭代替换完整SVD。

    步骤 1-4: PACT过滤（使用 torch.svd_lowrank 随机SVD）
    步骤 5: 使用 Newton-Schulz 迭代进行各向同性融合（替代完整SVD的ISOC）

    Args:
        task_vectors: 任务向量列表
        pretrained_vector: 预训练模型向量
        config: 配置对象

    Returns:
        new_vector: 融合后的向量字典
    """
    device = config.device

    # 固定比例方法参数
    K_ratio = getattr(config.method, 'K_ratio', 0.8)
    k_per_task = getattr(config.method, 'k_per_task', 10)
    ns_iterations = getattr(config.method, 'ns_iterations', 5)
    eps = getattr(config.method, 'ns_eps', 1e-8)

    print(f"Computing PACT-RANDSVD filtering "
          f"(K_ratio={K_ratio}, k_per_task={k_per_task})...")
    print(f"ISOC step using Newton-Schulz iteration "
          f"(ns_iterations={ns_iterations})...")

    with torch.no_grad():
        # 首先收集所有任务的隐式依赖空间
        implicit_spaces = {}
        filtered_task_vectors = []

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

                # 步骤 1 & 2: 使用随机SVD（固定比例）提取空间基底
                V_pre_K, K = extract_pretrained_core_space_rand(W_pre, K_ratio)
                V_t_k, k = extract_task_explicit_space_rand(Delta_t, k_per_task)
                if task_idx == 0:
                    print(f"  Layer {key}: K={K}, k={k} (randsvd)")

                del W_pre

                # 步骤 3: 计算任务的隐式依赖空间
                V_rel_t = compute_implicit_reliance_space(V_pre_K, V_t_k)
                del V_pre_K, V_t_k

                if key not in implicit_spaces:
                    implicit_spaces[key] = []
                implicit_spaces[key].append(V_rel_t.cpu())
                del V_rel_t

                filtered_vector_dict[key] = Delta_t.cpu()
                del Delta_t

            torch.cuda.empty_cache()

            filtered_task_vectors.append(
                type(task_vector)(model_name=task_vector.model_name, vector=filtered_vector_dict)
            )

        # 步骤 4: 对每个任务进行正交过滤
        for task_idx, task_vector in enumerate(filtered_task_vectors):
            for key in task_vector.vector:
                is_2d_matrix = (
                    len(task_vector.vector[key].shape) == 2 and "text_projection" not in key
                )

                if not is_2d_matrix:
                    continue

                if key not in implicit_spaces:
                    continue

                protect_spaces = []
                for other_idx in range(len(filtered_task_vectors)):
                    if other_idx != task_idx:
                        V_rel = implicit_spaces[key][other_idx]
                        if V_rel.shape[1] > 0:
                            protect_spaces.append(V_rel.to(device))

                if len(protect_spaces) > 0:
                    V_protect_j = torch.cat(protect_spaces, dim=1)
                    del protect_spaces
                    V_protect_j, _ = torch.linalg.qr(V_protect_j)

                    Delta_j = task_vector.vector[key].to(device)
                    Delta_j_filtered = orthogonal_core_filtering(Delta_j, V_protect_j)
                    del Delta_j, V_protect_j
                    task_vector.vector[key] = Delta_j_filtered.cpu()
                    del Delta_j_filtered

            torch.cuda.empty_cache()

    # 步骤 5: 使用 Newton-Schulz 迭代进行各向同性融合（替代完整SVD的ISOC）
    print("Applying Newton-Schulz Isotropic Merging to filtered task vectors...")

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
                continue

            # 2D 矩阵：应用 Newton-Schulz 迭代
            W_sum = sum(tvs)
            del tvs

            m, n = W_sum.shape
            min_dim = min(m, n)

            norm_W = torch.norm(W_sum, p='fro')
            if norm_W < eps:
                new_vector[key] = W_sum / len(task_vectors)
                del W_sum
                continue

            X = W_sum / (norm_W + eps)

            for _ in range(ns_iterations):
                if m >= n:
                    A = X.t() @ X
                    A.mul_(-1).add_(3.0 * torch.eye(n, device=device, dtype=W_sum.dtype))
                    X = torch.matmul(X, A).mul_(0.5)
                    del A
                else:
                    A = X @ X.t()
                    A.mul_(-1).add_(3.0 * torch.eye(m, device=device, dtype=W_sum.dtype))
                    X = torch.matmul(A, X).mul_(0.5)
                    del A

            sum_sigma = torch.sum(X * W_sum)
            mean_sigma = sum_sigma / min_dim

            new_vector[key] = mean_sigma * X

            del X, W_sum, norm_W, sum_sigma, mean_sigma
            torch.cuda.empty_cache()

    return new_vector


def pact_ta_randsvd_merge(task_vectors, pretrained_vector, config):
    """
    PACT-TA 简化版（随机SVD）
    使用随机SVD替换完整SVD进行PACT过滤，然后直接平均（Task Arithmetic）

    步骤 1-4: PACT过滤（使用 torch.svd_lowrank 随机SVD）
    步骤 5: Task Arithmetic（直接平均）

    Args:
        task_vectors: 任务向量列表
        pretrained_vector: 预训练模型向量
        config: 配置对象

    Returns:
        new_vector: 融合后的向量字典
    """
    device = config.device

    # 固定比例方法参数
    K_ratio = getattr(config.method, 'K_ratio', 0.8)
    k_per_task = getattr(config.method, 'k_per_task', 10)

    print(f"Computing PACT-RANDSVD filtering "
          f"(K_ratio={K_ratio}, k_per_task={k_per_task})...")
    print(f"Applying Task Arithmetic (direct sum) to filtered task vectors...")

    with torch.no_grad():
        # 首先收集所有任务的隐式依赖空间
        implicit_spaces = {}
        filtered_task_vectors = []

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

                # 步骤 1 & 2: 使用随机SVD（固定比例）提取空间基底
                V_pre_K, K = extract_pretrained_core_space_rand(W_pre, K_ratio)
                V_t_k, k = extract_task_explicit_space_rand(Delta_t, k_per_task)
                if task_idx == 0:
                    print(f"  Layer {key}: K={K}, k={k} (randsvd)")

                del W_pre

                # 步骤 3: 计算任务的隐式依赖空间
                V_rel_t = compute_implicit_reliance_space(V_pre_K, V_t_k)
                del V_pre_K, V_t_k

                if key not in implicit_spaces:
                    implicit_spaces[key] = []
                implicit_spaces[key].append(V_rel_t.cpu())
                del V_rel_t

                filtered_vector_dict[key] = Delta_t.cpu()
                del Delta_t

            torch.cuda.empty_cache()

            filtered_task_vectors.append(
                type(task_vector)(model_name=task_vector.model_name, vector=filtered_vector_dict)
            )

        # 步骤 4: 对每个任务进行正交过滤
        for task_idx, task_vector in enumerate(filtered_task_vectors):
            for key in task_vector.vector:
                is_2d_matrix = (
                    len(task_vector.vector[key].shape) == 2 and "text_projection" not in key
                )

                if not is_2d_matrix:
                    continue

                if key not in implicit_spaces:
                    continue

                protect_spaces = []
                for other_idx in range(len(filtered_task_vectors)):
                    if other_idx != task_idx:
                        V_rel = implicit_spaces[key][other_idx]
                        if V_rel.shape[1] > 0:
                            protect_spaces.append(V_rel.to(device))

                if len(protect_spaces) > 0:
                    V_protect_j = torch.cat(protect_spaces, dim=1)
                    del protect_spaces
                    V_protect_j, _ = torch.linalg.qr(V_protect_j)

                    Delta_j = task_vector.vector[key].to(device)
                    Delta_j_filtered = orthogonal_core_filtering(Delta_j, V_protect_j)
                    del Delta_j, V_protect_j
                    task_vector.vector[key] = Delta_j_filtered.cpu()
                    del Delta_j_filtered

            torch.cuda.empty_cache()

    # 步骤 5: Task Arithmetic（直接平均）
    new_vector = {}
    with torch.no_grad():
        for key in filtered_task_vectors[0].vector:
            tvs = [tv.vector[key].to(device) for tv in filtered_task_vectors]
            new_vector[key] = sum(tvs) / len(tvs)
            del tvs

        torch.cuda.empty_cache()

    return new_vector


def pact_tsvm_randsvd_merge(task_vectors, pretrained_vector, config):
    """
    PACT-TSVM 随机SVD版本
    在两处使用随机SVD替换完整SVD：
      1) PACT 过滤阶段（步骤1&2）：使用 torch.svd_lowrank 提取预训练核心空间和任务显式空间
      2) TSVM 融合阶段（步骤5）：使用 torch.svd_lowrank 替换完整SVD进行分解和正交化

    Args:
        task_vectors: 任务向量列表
        pretrained_vector: 预训练模型向量
        config: 配置对象

    Returns:
        new_vector: 融合后的向量字典
    """
    device = config.device
    num_tasks = len(task_vectors)

    K_ratio = getattr(config.method, 'K_ratio', 0.8)
    k_per_task = getattr(config.method, 'k_per_task', 10)
    niter = getattr(config.method, 'niter', 2)

    print(f"Computing PACT-TSVM-RSVD "
          f"(K_ratio={K_ratio}, k_per_task={k_per_task}, niter={niter})...")

    with torch.no_grad():
        # === 步骤 1~3: 使用随机SVD收集所有任务的隐式依赖空间 ===
        implicit_spaces = {}
        filtered_task_vectors = []

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

                # 步骤 1 & 2: 使用随机SVD提取空间基底（改造点1）
                V_pre_K, K = extract_pretrained_core_space_rand(W_pre, K_ratio, niter=niter)
                V_t_k, k = extract_task_explicit_space_rand(Delta_t, k_per_task, niter=niter)
                if task_idx == 0:
                    print(f"  Layer {key}: K={K}, k={k} (randsvd, niter={niter})")

                del W_pre

                # 步骤 3: 计算隐式依赖空间
                V_rel_t = compute_implicit_reliance_space(V_pre_K, V_t_k)
                del V_pre_K, V_t_k

                if key not in implicit_spaces:
                    implicit_spaces[key] = []
                implicit_spaces[key].append(V_rel_t.cpu())
                del V_rel_t

                filtered_vector_dict[key] = Delta_t.cpu()
                del Delta_t

            torch.cuda.empty_cache()

            filtered_task_vectors.append(
                type(task_vector)(model_name=task_vector.model_name, vector=filtered_vector_dict)
            )

        # === 步骤 4: 正交过滤 ===
        for task_idx, task_vector in enumerate(filtered_task_vectors):
            for key in task_vector.vector:
                is_2d_matrix = (
                    len(task_vector.vector[key].shape) == 2 and "text_projection" not in key
                )

                if not is_2d_matrix:
                    continue

                if key not in implicit_spaces:
                    continue

                protect_spaces = []
                for other_idx in range(len(filtered_task_vectors)):
                    if other_idx != task_idx:
                        V_rel = implicit_spaces[key][other_idx]
                        if V_rel.shape[1] > 0:
                            protect_spaces.append(V_rel.to(device))

                if len(protect_spaces) > 0:
                    V_protect_j = torch.cat(protect_spaces, dim=1)
                    del protect_spaces
                    V_protect_j, _ = torch.linalg.qr(V_protect_j)

                    Delta_j = task_vector.vector[key].to(device)
                    Delta_j_filtered = orthogonal_core_filtering(Delta_j, V_protect_j)
                    del Delta_j, V_protect_j
                    task_vector.vector[key] = Delta_j_filtered.cpu()
                    del Delta_j_filtered

            torch.cuda.empty_cache()

    # === 步骤 5: 使用随机SVD版本的TSVM进行融合（改造点2） ===
    print("Applying TSVM with Randomized SVD to filtered task vectors...")

    new_vector = {}

    with torch.no_grad():
        for key in filtered_task_vectors[0].vector:
            shape_ = filtered_task_vectors[0].vector[key].shape

            is_2d_matrix = (len(shape_) == 2) and ("text_projection" not in key)

            if not is_2d_matrix:
                for i, tv in enumerate(filtered_task_vectors):
                    vec = tv.vector[key].to(device)
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)
                    del vec
                torch.cuda.empty_cache()
                continue

            # 2D 矩阵：使用随机SVD执行TSVM拼接融合
            total_dims = min(shape_)
            base_dims_per_task = total_dims // num_tasks
            extra_dims = total_dims % num_tasks

            sum_u = torch.zeros(shape_[0], total_dims, device=device, dtype=torch.float32)
            sum_s = torch.zeros(total_dims, device=device, dtype=torch.float32)
            sum_v = torch.zeros(total_dims, shape_[1], device=device, dtype=torch.float32)

            for i, tv in enumerate(filtered_task_vectors):
                vec = tv.vector[key].to(device)

                dims_for_this_task = base_dims_per_task + (1 if i < extra_dims else 0)
                dims_for_this_task = max(1, dims_for_this_task)

                # 随机SVD：只计算需要的低秩分量
                u_r, s_r, v_r = torch.svd_lowrank(vec, q=dims_for_this_task, niter=niter)
                v_r = v_r.t()

                start_idx = i * base_dims_per_task + min(i, extra_dims)
                end_idx = start_idx + dims_for_this_task

                sum_u[:, start_idx:end_idx] = u_r[:, :dims_for_this_task]
                sum_s[start_idx:end_idx] = s_r[:dims_for_this_task]
                sum_v[start_idx:end_idx, :] = v_r[:dims_for_this_task, :]

                del vec, u_r, s_r, v_r

            torch.cuda.empty_cache()

            # 对拼接后的 U 和 V 使用随机SVD进行正交化
            effective_rank = min(
                sum_u.shape[0], sum_u.shape[1],
                num_tasks * (base_dims_per_task + 1)
            )
            effective_rank = max(1, effective_rank)

            u_u, _, v_u = torch.svd_lowrank(sum_u, q=effective_rank, niter=niter)
            v_u = v_u.t()

            u_v, _, v_v = torch.svd_lowrank(sum_v, q=effective_rank, niter=niter)
            v_v = v_v.t()

            new_vector[key] = torch.linalg.multi_dot(
                (u_u, v_u, torch.diag(sum_s), u_v, v_v)
            )

            del sum_u, sum_s, sum_v, u_u, v_u, u_v, v_v
            torch.cuda.empty_cache()

    return new_vector
