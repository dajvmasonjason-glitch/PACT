import torch
import math
import time


class GPUMemoryMonitor:
    def __init__(self, device, report_title, label_prefix):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.report_title = report_title
        self.label_prefix = label_prefix
        self.enabled = torch.cuda.is_available() and self.device.type == 'cuda'
        self.marks = []
        self.baseline_mem = 0.0
        self.start_time = None

    def _allocated_mb(self):
        return torch.cuda.memory_allocated(self.device) / (1024.0 * 1024.0)

    def _peak_mb(self):
        return torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)

    def start(self):
        if not self.enabled:
            return

        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.start_time = time.time()
        self.baseline_mem = self._allocated_mb()
        self.marks = []

    def reset_baseline(self):
        if not self.enabled:
            return

        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.start_time = time.time()
        self.baseline_mem = self._allocated_mb()
        self.marks = []

    def mark(self, label: str):
        if not self.enabled:
            return

        torch.cuda.synchronize(self.device)
        elapsed = time.time() - self.start_time
        allocated = self._allocated_mb()
        # 读取本区间（自上次 reset 以来）的真实峰值，然后立即重置供下一区间使用
        interval_peak = self._peak_mb()
        torch.cuda.reset_peak_memory_stats(self.device)
        self.marks.append((f"{self.label_prefix}/{label}", elapsed, allocated, interval_peak))

    def stop(self):
        pass

    def report(self):
        if not self.enabled:
            return

        print(f"\n========== {self.report_title} Memory Report ==========")
        print(f"Baseline (task vectors loaded): {self.baseline_mem:.1f} MB")
        print("Checkpoints:")
        prev_allocated = self.baseline_mem
        for label, elapsed, allocated, interval_peak in self.marks:
            print(
                f"  [{label:<23}] t={elapsed:.2f}s  "
                f"allocated={allocated:.1f} MB  "
                f"delta={allocated - self.baseline_mem:+.1f} MB  "
                f"interval_peak={interval_peak:.1f} MB  "
                f"scratch={interval_peak - prev_allocated:+.1f} MB"
            )
            prev_allocated = allocated


# ---------------------------------------------------------------------------
# FP8 (e4m3) simulation helpers
# ---------------------------------------------------------------------------
# E4M3: exponent bits=4, mantissa bits=3  →  max representable value = 448.0
_FP8_E4M3_MAX = 448.0


def _simulate_fp8_quantize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Mathematically simulate FP8 (e4m3) quantization without relying on PyTorch >= 2.1.
    This dynamically calculates the Exponent and truncates the Mantissa to 3 bits,
    perfectly reproducing floating-point representation error.
    """
    orig_dtype = tensor.dtype
    t = tensor.float()  # 在 FP32 下进行精确计算

    absmax = t.abs().max()
    if absmax < 1e-30:
        return tensor  # 全 0 张量直接跳过

    # 1. Scaling: 缩放最大值至安全边界 440 (E4M3 绝对最大值为 448)
    scale = 440.0 / absmax
    t_scaled = t * scale

    # 获取符号和绝对值
    sign = torch.sign(t_scaled)
    abs_val = torch.abs(t_scaled)

    # E4M3 能够表示的最小正数 (非零值) 是 2^-9 (大概 0.001953)
    # 低于这个值的数字在 FP8 硬件中会触发 Underflow (下溢)，变成 0
    min_val = 2.0 ** -9
    mask = abs_val >= min_val  # 记录没有下溢的位置

    # 2. 提取指数 (Exponent): E = floor(log2(x))
    # 为防止 log2 报错，用 clamp 限定下界。
    # 巧妙之处：将下界设为 2^-6 (E4M3 的最小 Normal number 指数)，
    # 这配合后面的尾数舍入，能完美模拟 subnormal (非规格化数字) 行为！
    E = torch.floor(torch.log2(torch.clamp(abs_val, min=2.0 ** -6)))

    # 3. 提取尾数 (Mantissa): M = x / 2^E
    # 对于常规数字，M 会落在 [1.0, 2.0) 之间
    M = abs_val / (2.0 ** E)

    # 4. 尾数量化 (Quantize Mantissa): E4M3 只有 3 位尾数 (2^3 = 8 个刻度)
    # 我们将尾数强制舍入到最接近的 1/8 刻度上
    M_rounded = torch.round(M * 8.0) / 8.0

    # 5. 重新组装浮点数: x_approx = Sign * M_rounded * 2^E
    t_fp8 = sign * M_rounded * (2.0 ** E)

    # 强制让极小值下溢归零
    t_fp8 = t_fp8 * mask

    # 6. 还原 Scaling
    t_restored = t_fp8 / scale

    return t_restored.to(orig_dtype)


def isoc_ns_merge(task_vectors, config):
    """
    Iso-C merging using Newton-Schulz iteration instead of SVD.

    When config.method.enable_fp8 is True, each task vector is passed through
    a simulated FP8 (e4m3) quantizer before being summed and processed.  This
    lets you measure how much accuracy degrades when task vectors are stored /
    communicated in FP8 precision.

    All Newton-Schulz arithmetic runs in the tensor's native dtype (float32 by
    default); the bfloat16 compute path has been removed.

    When config.method.gram is True, avoids allocating intermediate A matrix
    by computing Gram matrix operations in-place, saving GPU memory.

    When config.method.use_redundancy_weighting is True, applies inverse-redundancy
    weighting before Newton-Schulz iteration (same as iso_c Strategy 2).
    """
    device = torch.device(config.device) if isinstance(config.device, str) else config.device
    num_iterations = getattr(config.method, 'ns_iterations', 5)
    eps = getattr(config.method, 'ns_eps', 1e-8)
    enable_fp8 = getattr(config.method, 'enable_fp8', False)
    gram = getattr(config.method, 'gram', False)
    use_redundancy_weighting = getattr(config.method, 'use_redundancy_weighting', False)

    print(f"Computing Newton-Schulz Isotropic Merging "
          f"(iterations={num_iterations}, simulate_fp8={enable_fp8}, gram_mode={gram}, "
          f"redundancy_weighting={use_redundancy_weighting})...")

    monitor = GPUMemoryMonitor(device, "Iso-C-NS", "isoc_ns")

    with torch.no_grad():
        new_vector = {}
        layer_idx = 0
        baseline_recorded = False

        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        monitor.start()

        try:
            for key in task_vectors[0].vector:
                tvs = [task_vector.vector[key].to(device) for task_vector in task_vectors]
                if not baseline_recorded:
                    monitor.reset_baseline()
                    baseline_recorded = True

                # --- optional FP8 simulation: quantize each task vector ---
                if enable_fp8:
                    tvs = [_simulate_fp8_quantize(tv) for tv in tvs]

                is_2d_matrix = (
                    len(task_vectors[0].vector[key].shape) == 2
                    and "text_projection" not in key
                )

                # --- Apply redundancy weighting for 2D matrices ---
                if use_redundancy_weighting and is_2d_matrix:
                    # Flatten and compute unit vectors for similarity
                    X = torch.stack([tv.reshape(-1) for tv in tvs], dim=0)  # (T, num_params)
                    norms = X.norm(dim=1, keepdim=True)                      # (T, 1)
                    # Use safe eps based on dtype
                    safe_eps = torch.finfo(X.dtype).eps * 10 if X.dtype in [torch.float16, torch.bfloat16] else eps
                    X_norm = X / (norms + safe_eps)                          # (T, num_params)
                    # Absolute cosine similarity matrix
                    C_abs = torch.abs(X_norm @ X_norm.t())                   # (T, T)
                    # Redundancy score per task
                    R = C_abs.sum(dim=1)                                     # (T,)
                    # Inverse-redundancy weights, normalized so sum = T
                    weights = 1.0 / R                                        # (T,)
                    weights = weights * (len(tvs) / weights.sum())           # (T,)
                    # Weighted sum of ORIGINAL matrices
                    W_sum = sum(w * tv for w, tv in zip(weights, tvs))
                    del X, X_norm, C_abs, R, weights, norms
                else:
                    # Original behavior: simple sum
                    W_sum = sum(tvs)

                del tvs  # 及时释放内存

                if not is_2d_matrix:
                    new_vector[key] = W_sum / len(task_vectors)
                    del W_sum
                    continue

                monitor.mark(f"layer_{layer_idx}_start")

                W = W_sum
                m, n = W.shape
                min_dim = min(m, n)

                # Newton-Schulz runs in the tensor's native dtype (no bfloat16 cast)
                norm_W = torch.norm(W, p='fro')
                if norm_W < eps:
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    monitor.mark(f"layer_{layer_idx}_done")
                    layer_idx += 1
                    continue

                X = W / (norm_W + eps)

                for _ in range(num_iterations):
                    if gram:
                        # 不分配 A 矩阵，直接计算以节省显存
                        if m >= n:
                            # X = 0.5 * X @ (3I - X^T @ X)
                            X = torch.matmul(X, 3.0 * torch.eye(n, device=device, dtype=W.dtype) - X.t() @ X).mul_(0.5)
                        else:
                            # X = 0.5 * (3I - X @ X^T) @ X
                            X = torch.matmul(3.0 * torch.eye(m, device=device, dtype=W.dtype) - X @ X.t(), X).mul_(0.5)
                    else:
                        # 原始方法：显式分配 A 矩阵
                        if m >= n:
                            A = X.t() @ X
                            A.mul_(-1).add_(3.0 * torch.eye(n, device=device, dtype=W.dtype))
                            X = torch.matmul(X, A).mul_(0.5)
                            del A
                        else:
                            A = X @ X.t()
                            A.mul_(-1).add_(3.0 * torch.eye(m, device=device, dtype=W.dtype))
                            X = torch.matmul(A, X).mul_(0.5)
                            del A

                sum_sigma = torch.sum(X * W)
                mean_sigma = sum_sigma / min_dim

                new_vector[key] = mean_sigma * X

                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                monitor.mark(f"layer_{layer_idx}_done")

                del X, W,W_sum, norm_W, sum_sigma, mean_sigma

                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                layer_idx += 1
        finally:
            monitor.stop()
            monitor.report()

    return new_vector


def iso_c(task_vectors, config, use_magnitude_weighting=False):
    device = config.device
    enable_fp8 = getattr(config.method, 'enable_fp8', False)
    aggregation_1d = getattr(config.method, 'aggregation_1d', 'mean')  # mean, median, sign_consensus, max_abs
    enable_layerwise_scaling = getattr(config.method, 'enable_layerwise_scaling', True)
    eps = 1e-8
    print(f"Computing SVD (simulate_fp8={enable_fp8}, magnitude_weighting={use_magnitude_weighting}, "
          f"aggregation_1d={aggregation_1d}, layerwise_scaling={enable_layerwise_scaling})...")
    monitor = GPUMemoryMonitor(device, "Iso-C", "iso_c")
    with torch.no_grad():
        new_vector = {}
        layer_idx = 0
        baseline_recorded = False

        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        monitor.start()

        try:
            for key in task_vectors[0].vector:
                tvs = [task_vector.vector[key].to(device) for task_vector in task_vectors]
                if not baseline_recorded:
                    monitor.reset_baseline()
                    baseline_recorded = True

                if enable_fp8:
                    tvs = [_simulate_fp8_quantize(tv) for tv in tvs]

                # Check if this is a 2D weight matrix (not 1D bias/LayerNorm)
                is_2d_matrix = len(task_vectors[0].vector[key].shape) == 2 and "text_projection" not in key

                if use_magnitude_weighting and is_2d_matrix:
                    norms = [torch.linalg.norm(tv, ord='fro') for tv in tvs]
                    tvs_normalized = [tv / (norm + eps) for tv, norm in zip(tvs, norms)]
                    new_vector[key] = sum(tvs_normalized)
                elif not is_2d_matrix:
                    # 1D vectors: apply different aggregation methods
                    if aggregation_1d == 'median':
                        # Median: select the median value across tasks for each parameter
                        stacked = torch.stack(tvs, dim=0)  # (T, ...)
                        new_vector[key] = torch.median(stacked, dim=0).values
                    elif aggregation_1d == 'sign_consensus':
                        # Sign consensus with masking: vote on sign, then average only agreeing tasks
                        stacked = torch.stack(tvs, dim=0)  # (T, ...)
                        # Count positive and negative votes
                        signs = torch.sign(stacked)  # (T, ...)
                        positive_votes = (signs > 0).sum(dim=0)  # (...)
                        negative_votes = (signs < 0).sum(dim=0)  # (...)
                        # Determine consensus sign (majority vote)
                        consensus_sign = torch.where(positive_votes >= negative_votes,
                                                     torch.ones_like(positive_votes),
                                                     -torch.ones_like(positive_votes))
                        # Mask: keep only tasks that agree with consensus
                        mask = (signs * consensus_sign.unsqueeze(0)) > 0  # (T, ...)
                        # Average only agreeing tasks (handle case where no task agrees)
                        masked_sum = (stacked * mask).sum(dim=0)
                        count = mask.sum(dim=0).clamp(min=1)  # Avoid division by zero
                        new_vector[key] = masked_sum / count
                    elif aggregation_1d == 'max_abs':
                        # Max absolute value pooling: keep the update with largest absolute value
                        stacked = torch.stack(tvs, dim=0)  # (T, ...)
                        abs_vals = torch.abs(stacked)  # (T, ...)
                        max_indices = torch.argmax(abs_vals, dim=0)  # (...)
                        # Gather values from the task with max absolute value
                        new_vector[key] = torch.gather(stacked, 0, max_indices.unsqueeze(0)).squeeze(0)
                    else:
                        # Default: mean (original behavior)
                        new_vector[key] = sum(tvs) / len(tvs)
                else:
                    # Original behavior: simple average for 2D when no weighting
                    new_vector[key] = sum(tvs) / len(tvs)

                if is_2d_matrix:
                    monitor.mark(f"layer_{layer_idx}_start")
                    if use_magnitude_weighting:
                        # Already summed (not averaged) above
                        pass
                    else:
                        # Original behavior: multiply back by len(tvs)
                        new_vector[key] *= len(tvs)

                    delta_ta = new_vector[key]

                    U, S, V = torch.linalg.svd(delta_ta, full_matrices=False)
                    S_mean = torch.ones_like(S) * S.mean()

                    delta_iso = torch.linalg.multi_dot(
                        (
                            U,
                            torch.diag(S_mean),
                            V,
                        )
                    )

                    if enable_layerwise_scaling:
                        norm_ta = torch.linalg.norm(delta_ta, ord='fro')
                        norm_iso = torch.linalg.norm(delta_iso, ord='fro')
                        gamma = norm_ta / (norm_iso + 1e-8)
                        new_vector[key] = gamma * delta_iso
                    else:
                        new_vector[key] = delta_iso

                    if torch.device(device).type == 'cuda':
                        torch.cuda.synchronize(torch.device(device))
                    monitor.mark(f"layer_{layer_idx}_done")
                    layer_idx += 1
        finally:
            monitor.stop()
            monitor.report()

    return new_vector


@torch.no_grad()
def iso_cts(task_vectors, config):
    device = config.device
    new_vector = {}

    print("Computing SVD...")
    for key in task_vectors[0].vector:
        shape_ = task_vectors[0].vector[key].shape

        is_2d_matrix = (len(shape_) == 2) and ("text_projection" not in key)
        if not is_2d_matrix:
            print(f"Combining by avg {key}...")
            for i, (task_vector, dataset) in enumerate(zip(task_vectors, config.DATASETS)):
                vec = task_vector.vector[key].to(device)
                if i == 0:
                    new_vector[key] = vec.clone()
                else:
                    new_vector[key] += (vec - new_vector[key]) / (i + 1)
            continue
        
        print(f"Computing common space using sum for {key}...")
        combined_w = sum([task_vector.vector[key].to(device) for task_vector in task_vectors])

        ### Calculate the common space size (making sure that task specific space is equally divisible) ###
        common_space_index_s = int(min(shape_) * config.method.common_space_fraction)
        _task_specific_total_space_index_s = round((min(shape_) - common_space_index_s) / len(config.DATASETS)) * len(config.DATASETS)
        common_space_index_s = min(shape_) - _task_specific_total_space_index_s

        u, s, v = torch.linalg.svd(combined_w, full_matrices=False)
        common_space_u = u[:, :common_space_index_s]
        common_space_s = s[:common_space_index_s]
        common_space_v = v[:common_space_index_s, :]
        ###################################################################
        
        ### Calculate task specific space ###
        n_dims_per_task = int((min(shape_) - common_space_index_s) / len(config.DATASETS))
        for i, task_vector in enumerate(task_vectors):
            w = task_vector.vector[key].to(device)

            # calculate the projection onto task specific space to remove the common space
            w_ts = w - common_space_u @ common_space_u.T @ w
            u_ts, s_ts, v_ts = torch.linalg.svd(w_ts, full_matrices=False)            
            
            if i == 0:
                combined_space_u = torch.zeros_like(u_ts, device=device)
                combined_space_s = torch.zeros_like(s_ts, device=device)
                combined_space_v = torch.zeros_like(v_ts, device=device)
                
            combined_space_u[:, i * n_dims_per_task : (i + 1) * n_dims_per_task] = u_ts[:, :n_dims_per_task]
            combined_space_s[i * n_dims_per_task : (i + 1) * n_dims_per_task] = s_ts[:n_dims_per_task]
            combined_space_v[i * n_dims_per_task : (i + 1) * n_dims_per_task, :] = v_ts[:n_dims_per_task, :]
        ###################################################################
        
        combined_space_u[:, len(config.DATASETS) * n_dims_per_task : len(config.DATASETS) * n_dims_per_task + common_space_index_s] = common_space_u
        combined_space_s[len(config.DATASETS) * n_dims_per_task : len(config.DATASETS) * n_dims_per_task + common_space_index_s] = common_space_s
        combined_space_v[len(config.DATASETS) * n_dims_per_task : len(config.DATASETS) * n_dims_per_task + common_space_index_s, :] = common_space_v
        
        ### Orthogonalize combined_space_u and combined_space_v ###
        u_combined_space_u, s_combined_space_u, v_combined_space_u = torch.linalg.svd(combined_space_u, full_matrices=False)
        u_combined_space_v, s_combined_space_v, v_combined_space_v = torch.linalg.svd(combined_space_v, full_matrices=False)
        combined_space_u = u_combined_space_u @ v_combined_space_u
        combined_space_v = u_combined_space_v @ v_combined_space_v
        ###################################################################
        
        combined_space_s = torch.ones_like(combined_space_s) * combined_space_s.mean()
                
        new_vector[key] = torch.linalg.multi_dot(
            (
                combined_space_u,
                torch.diag(combined_space_s),
                combined_space_v,
            )
        )
    
    return new_vector
