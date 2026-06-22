import torch
import math


###############
#### TSV Merge Orthogonalization
def compute_and_sum_svd_mem_reduction(task_vectors, config):
    """
    Computes the Singular Value Decomposition (SVD) for each vector in the task_vectors,
    reduces the dimensionality of the vectors based on the sv_reduction factor, and concatenate
    the low-rank matrices. If the vector is not a 2D tensor or is "text_projection", it computes the mean of the vectors.
    Computation of the SVD is performed also for the second operation.

    Args:
        task_vectors (list): A list of task vector objects, where each object contains a
                             dictionary of vectors.
        config (object): Configuration object containing the following attributes:
                         - DATASETS (list): List of datasets.
                         - device (torch.device): The device to perform computations on.

    Returns:
        dict: A dictionary containing the new vectors after SVD computation and merging.
    """
    sv_reduction = 1 / len(config.DATASETS)
    device = config.device
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = {}
            for i, (task_vector, dataset) in enumerate(
                zip(task_vectors, config.DATASETS)
            ):
                vec = task_vector.vector[key].to(device)

                if (
                    len(task_vector.vector[key].shape) == 2
                    and "text_projection" not in key
                ):
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)
                    reduced_index_s = int(s.shape[0] * sv_reduction)

                    # select only the first reduced_index_s columns of u and place them
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                        :, :reduced_index_s
                    ]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                        :reduced_index_s
                    ]
                    # select only the first reduced_index_s rows of v and place them
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                        :reduced_index_s, :
                    ]

                else:
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if len(task_vector.vector[key].shape) == 2 and "text_projection" not in key:
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

                new_vector[key] = torch.linalg.multi_dot(
                    (
                        u_u,
                        v_u,
                        torch.diag(sum_s),
                        u_v,
                        v_v,
                    )
                )

    return new_vector


###############
#### TSV Merge Orthogonalization (Randomized SVD version)
def compute_and_sum_svd_mem_reduction_rsvd(task_vectors, config):
    """
    TSVM 的随机SVD版本。
    使用 torch.svd_lowrank（随机SVD）替换 torch.linalg.svd（完整SVD），
    加速计算并降低内存占用。

    实现要点：
      - 每个任务只对低秩部分 (rank = min(shape) * sv_reduction) 进行随机SVD，
        无需计算整个 SVD 分解。
      - 第二次正交化时，sum_u / sum_v 的目标秩本身已经被截断到 num_tasks*reduced
        左右，因此对 sum_u / sum_v 同样使用 torch.svd_lowrank，避免完整 SVD。

    Args:
        task_vectors (list): 任务向量列表
        config (object): 配置对象，需包含:
                         - DATASETS (list): 数据集列表
                         - device (torch.device): 计算设备
                         - method.niter (int, optional): 随机SVD幂迭代次数，默认2

    Returns:
        dict: 融合后的向量字典
    """
    sv_reduction = 1 / len(config.DATASETS)
    device = config.device
    niter = getattr(config.method, 'niter', 2)
    print(f"Computing TSVM with Randomized SVD (niter={niter})...")

    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = {}

            is_2d_matrix = (
                len(task_vectors[0].vector[key].shape) == 2
                and "text_projection" not in key
            )

            if not is_2d_matrix:
                # 非 2D 张量：增量平均
                for i, task_vector in enumerate(task_vectors):
                    vec = task_vector.vector[key].to(device)
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)
                continue

            # 2D 矩阵：每个任务做随机 SVD 并拼接
            sum_u = None
            sum_s = None
            sum_v = None
            reduced_index_s = None

            for i, task_vector in enumerate(task_vectors):
                vec = task_vector.vector[key].to(device)

                if i == 0:
                    total_dims = min(vec.shape)
                    reduced_index_s = max(1, int(total_dims * sv_reduction))
                    print(
                        f"Computing RSVD for {key} "
                        f"(rank per task={reduced_index_s}, total={total_dims})..."
                    )
                    sum_u = torch.zeros(
                        vec.shape[0], total_dims, device=device, dtype=vec.dtype
                    )
                    sum_s = torch.zeros(total_dims, device=device, dtype=vec.dtype)
                    sum_v = torch.zeros(
                        total_dims, vec.shape[1], device=device, dtype=vec.dtype
                    )

                # 随机SVD：只计算前 reduced_index_s 个奇异分量
                u_r, s_r, v_r = torch.svd_lowrank(
                    vec, q=reduced_index_s, niter=niter
                )
                # torch.svd_lowrank 返回 V (d_in, q)，需要转置成行向量形式
                v_r = v_r.t()

                start_idx = i * reduced_index_s
                end_idx = start_idx + reduced_index_s
                # 防止越界（最后一个任务可能溢出 total_dims）
                end_idx = min(end_idx, total_dims)
                actual = end_idx - start_idx

                sum_u[:, start_idx:end_idx] = u_r[:, :actual]
                sum_s[start_idx:end_idx] = s_r[:actual]
                sum_v[start_idx:end_idx, :] = v_r[:actual, :]

                del vec, u_r, s_r, v_r

            # 对拼接矩阵再次进行随机SVD做正交化
            # sum_u, sum_v 的有效秩 <= num_tasks * reduced_index_s，因此可以用 lowrank
            effective_rank = min(
                sum_u.shape[0], sum_u.shape[1],
                len(task_vectors) * reduced_index_s
            )
            effective_rank = max(1, effective_rank)

            u_u, _, v_u = torch.svd_lowrank(
                sum_u, q=effective_rank, niter=niter
            )
            v_u = v_u.t()

            u_v, _, v_v = torch.svd_lowrank(
                sum_v, q=effective_rank, niter=niter
            )
            v_v = v_v.t()

            # 重构：U_orth = u_u @ v_u, V_orth = u_v @ v_v
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

    return new_vector
