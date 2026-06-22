import torch
from collections import OrderedDict
from copy import deepcopy


# =============================================================================
# 随机SVD辅助函数（从 iso-merging-main/src/utils/pact_utils.py 迁移）
# 使用 torch.svd_lowrank 替代 torch.linalg.svd 以加速计算
# =============================================================================

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


def extract_task_explicit_space_rand_lora(A_t, k, niter=2):
    """
    步骤 2（随机SVD固定维度版本 - LoRA优化版）:
    对 LoRA 的 A 矩阵使用随机SVD提取任务显式变化空间

    Args:
        A_t: LoRA 的 A 矩阵 (r, d_in)，r 通常很小（如16）
        k: 每个任务分配的维度数
        niter: 随机SVD的幂迭代次数

    Returns:
        V_t_k: 任务显式变化子空间基底 (d_in, k)
        k_actual: 实际保留的维度
    """
    dim = min(A_t.shape[0], A_t.shape[1])
    k = min(k, dim)

    _, _, V = torch.svd_lowrank(A_t, q=k, niter=niter)
    # V 形状为 (d_in, k)

    return V, k


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
    V_rel_t_tilde = V_pre_K - V_t_k @ (V_t_k.t() @ V_pre_K)

    norm = torch.linalg.norm(V_rel_t_tilde, ord='fro')
    if norm < 1e-6:
        return torch.zeros((V_pre_K.shape[0], 0), device=V_pre_K.device, dtype=V_pre_K.dtype)

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
    if V_protect_j.shape[1] == 0:
        return Delta_j.clone()

    Delta_j_filtered = Delta_j - Delta_j @ V_protect_j @ V_protect_j.t()
    return Delta_j_filtered


def get_lora_pairs(state_dict):
    """
    从 state_dict 中识别 LoRA 权重对 (lora_A, lora_B)
    """
    lora_pairs = OrderedDict()

    for key, val in state_dict.items():
        if 'lora_A' in key:
            base_name = key.split('.lora_A')[0]
            if base_name not in lora_pairs:
                lora_pairs[base_name] = {}
            lora_pairs[base_name]['A'] = val
            lora_pairs[base_name]['A_key'] = key
        elif 'lora_B' in key:
            base_name = key.split('.lora_B')[0]
            if base_name not in lora_pairs:
                lora_pairs[base_name] = {}
            lora_pairs[base_name]['B'] = val
            lora_pairs[base_name]['B_key'] = key

    return lora_pairs


def pact_isoc_randsvd_merge(lora_task_vectors, pretrained_vector, config):
    """
    PACT-ISOC 随机SVD版本 (LoRA版本)

    使用 torch.svd_lowrank（随机SVD）替换完整SVD进行PACT过滤，
    并在ISOC融合步骤中使用 Newton-Schulz 迭代替代完整SVD。

    关键优化：
    1. 步骤1&2: torch.svd_lowrank 替代 torch.linalg.svd
    2. 步骤5: Newton-Schulz 迭代替代 SVD 进行各向同性融合

    Args:
        lora_task_vectors: LoRA 任务向量列表
        pretrained_vector: 预训练模型向量 (state_dict 格式)
        config: 配置对象，包含:
            - device: 计算设备
            - K_ratio: 预训练核心空间比例 (默认 0.8)
            - lora_rank: LoRA 的秩 r
            - scaling_coeffs: 缩放系数
            - ns_iterations: Newton-Schulz 迭代次数 (默认 5)
            - ns_eps: 数值稳定性小量 (默认 1e-8)

    Returns:
        new_vector: 融合后的向量字典
    """
    device = config.device if hasattr(config, 'device') else 'cuda'
    if isinstance(device, int):
        device = f'cuda:{device}'

    K_ratio = getattr(config, 'K_ratio', 0.8)
    lora_rank = getattr(config, 'lora_rank', None)
    scaling_coeffs = getattr(config, 'scaling_coeffs', 1.0)
    ns_iterations = getattr(config, 'ns_iterations', 5)
    ns_eps = getattr(config, 'ns_eps', 1e-8)

    if isinstance(scaling_coeffs, float):
        scaling_coeffs = [scaling_coeffs] * len(lora_task_vectors)

    num_tasks = len(lora_task_vectors)

    print(f"PACT-ISOC-RANDSVD (LoRA) merging with {num_tasks} tasks")
    print(f"  K_ratio={K_ratio}, ns_iterations={ns_iterations}")

    with torch.no_grad():
        implicit_spaces = {}
        filtered_deltas = []

        print("Phase 1: Extracting feature bases and implicit spaces (randsvd)...")
        for task_idx, task_vector in enumerate(lora_task_vectors):
            task_filtered_delta = {}

            for base_name in task_vector.keys():
                A_t = task_vector[base_name]['A'].to(device)
                B_t = task_vector[base_name]['B'].to(device)

                W_pre_key = base_name
                if W_pre_key not in pretrained_vector:
                    print(f"Warning: Missing pretrain weights for {W_pre_key}")
                    task_filtered_delta[base_name] = {
                        'delta': (B_t @ A_t).cpu(),
                        'is_lora': True
                    }
                    del A_t, B_t
                    continue

                W_pre = pretrained_vector[W_pre_key].to(device)

                is_2d_matrix = len(W_pre.shape) == 2

                if not is_2d_matrix:
                    task_filtered_delta[base_name] = {
                        'delta': (B_t @ A_t).cpu(),
                        'is_lora': False
                    }
                    del A_t, B_t, W_pre
                    continue

                k = lora_rank if lora_rank is not None else A_t.shape[0]

                # 步骤 1 & 2: 使用随机SVD（固定比例）提取空间基底
                V_pre_K, K = extract_pretrained_core_space_rand(W_pre, K_ratio)
                V_t_k, k_actual = extract_task_explicit_space_rand_lora(A_t, k)

                if task_idx == 0:
                    print(f"  {base_name}: K={K}, k={k_actual} (randsvd)")

                del W_pre

                # 步骤 3: 计算任务的隐式依赖空间
                V_rel_t = compute_implicit_reliance_space(V_pre_K, V_t_k)
                del V_pre_K, V_t_k

                if base_name not in implicit_spaces:
                    implicit_spaces[base_name] = []
                implicit_spaces[base_name].append(V_rel_t.cpu())
                del V_rel_t

                task_filtered_delta[base_name] = {
                    'delta': (B_t @ A_t).cpu(),
                    'is_lora': True
                }
                del A_t, B_t

            torch.cuda.empty_cache()
            filtered_deltas.append(task_filtered_delta)

        print("Phase 2: Matrix reconstruction and PACT orthogonal filtering...")
        for task_idx, task_delta in enumerate(filtered_deltas):
            for base_name in task_delta:
                if not task_delta[base_name]['is_lora']:
                    continue

                if base_name not in implicit_spaces:
                    continue

                protect_spaces = []
                for other_idx in range(num_tasks):
                    if other_idx != task_idx:
                        V_rel = implicit_spaces[base_name][other_idx]
                        if V_rel.shape[1] > 0:
                            protect_spaces.append(V_rel.to(device))

                if len(protect_spaces) > 0:
                    V_protect_j = torch.cat(protect_spaces, dim=1)
                    del protect_spaces
                    V_protect_j, _ = torch.linalg.qr(V_protect_j)

                    Delta_j = task_delta[base_name]['delta'].to(device)
                    Delta_j_filtered = orthogonal_core_filtering(Delta_j, V_protect_j)
                    del Delta_j, V_protect_j
                    task_delta[base_name]['delta'] = Delta_j_filtered.cpu()
                    del Delta_j_filtered
                else:
                    del protect_spaces

            torch.cuda.empty_cache()

        print(f"Phase 3: Executing Newton-Schulz Isotropic Merging (ns_iterations={ns_iterations})...")
        new_vector = {}

        all_base_names = set()
        for task_delta in filtered_deltas:
            all_base_names.update(task_delta.keys())

        for base_name in all_base_names:
            deltas = []
            for task_idx, task_delta in enumerate(filtered_deltas):
                if base_name in task_delta:
                    delta = task_delta[base_name]['delta'].to(device)
                    if scaling_coeffs[task_idx] != 1.0:
                        delta = delta * scaling_coeffs[task_idx]
                    deltas.append(delta)

            if len(deltas) == 0:
                continue

            is_lora = filtered_deltas[0][base_name]['is_lora']

            if not is_lora:
                new_vector[base_name] = (sum(deltas) / len(deltas)).cpu()
                del deltas
                continue

            # 2D 矩阵：应用 Newton-Schulz 迭代进行各向同性融合
            W_sum = sum(deltas)
            del deltas

            m, n = W_sum.shape
            min_dim = min(m, n)

            norm_W = torch.norm(W_sum, p='fro')
            if norm_W < ns_eps:
                new_vector[base_name] = (W_sum / num_tasks).cpu()
                del W_sum
                continue

            X = W_sum / (norm_W + ns_eps)

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

            new_vector[base_name] = (mean_sigma * X).cpu()

            del X, W_sum, norm_W, sum_sigma, mean_sigma
            torch.cuda.empty_cache()

    return new_vector


class PactIsoRsvdMerger:
    """
    PACT-ISOC-RANDSVD Merger for LoRA-finetuned models.

    使用随机SVD和Newton-Schulz迭代实现PACT-ISOC融合，适用于大规模模型。

    关键特性：
    1. 使用 torch.svd_lowrank（随机SVD）替代完整SVD提取特征基底
    2. 使用 Newton-Schulz 迭代替代完整SVD进行各向同性融合
    3. 保持与原始 pact_isoc.py 完全兼容的接口

    适用场景：
    - 大规模模型（如LLaMA）的融合，其中完整SVD计算代价过高
    - 对融合速度有要求的场景
    """

    def __init__(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        self.device = device
        self.scaling_coeffs = [1.0] * len(finetuned_models)
        self.param_handler = param_handler
        self.finetuned_models = finetuned_models
        self.ftms_params = [param_handler(ft_model) for ft_model in finetuned_models]
        self.pretrained_model = pretrained_model.cpu()
        self.pt_params = self.pretrained_model.state_dict()
        self.merge_config = merge_config or {}
        self.num_tasks = len(finetuned_models)

        self.key_mapping = {}
        self._build_key_mapping()

    def _extract_core_key(self, lora_key):
        """
        从 LoRA key 中提取 core key

        例如：
        base_model.model.vision_model.encoder.layers.0.self_attn.q_proj.lora_A.default.weight
        -> vision_model.encoder.layers.0.self_attn.q_proj.weight
        """
        if 'lora_A' in lora_key:
            base_part = lora_key.split('.lora_A')[0]
        elif 'lora_B' in lora_key:
            base_part = lora_key.split('.lora_B')[0]
        else:
            return None

        parts = base_part.split('.')
        start_idx = 0

        for i, p in enumerate(parts):
            if p in ['vision_model', 'text_model', 'model']:
                start_idx = i
                break

        core_parts = parts[start_idx:]
        core_key = '.'.join(core_parts) + '.weight'

        return core_key

    def _build_key_mapping(self):
        """
        建立 canonical_key -> real_key 的映射
        通过尾缀匹配在 pretrained keys 中找到对应的真实 key
        """
        lora_keys = set()
        for ft_model in self.finetuned_models:
            sd = ft_model.state_dict()
            for key in sd.keys():
                if 'lora_A' in key:
                    lora_keys.add(key)

        pretrained_keys = list(self.pt_params.keys())

        matched = 0
        unmatched = 0
        unmatched_keys = []

        for lora_key in lora_keys:
            core_key = self._extract_core_key(lora_key)
            if core_key is None:
                continue

            found = False
            for pretrained_key in pretrained_keys:
                if pretrained_key.endswith(core_key):
                    self.key_mapping[core_key] = pretrained_key
                    found = True
                    matched += 1
                    break

            if not found:
                unmatched += 1
                unmatched_keys.append(core_key)

        print(f"Key mapping: {matched} matched, {unmatched} unmatched")
        if unmatched > 0 and len(unmatched_keys) <= 5:
            print(f"  Unmatched keys: {unmatched_keys}")
        elif unmatched > 0:
            print(f"  First 5 unmatched keys: {unmatched_keys[:5]}")

    def get_lora_task_vectors(self):
        """
        从 LoRA 模型中提取任务向量 (A 和 B 矩阵)
        返回的 key 为 canonical_key (core key)
        """
        lora_task_vectors = []

        for ft_model in self.finetuned_models:
            task_vector = OrderedDict()
            sd = ft_model.state_dict()

            for key, val in sd.items():
                if 'lora_A' in key:
                    core_key = self._extract_core_key(key)
                    if core_key is None:
                        continue
                    if core_key not in task_vector:
                        task_vector[core_key] = {}
                    task_vector[core_key]['A'] = val
                elif 'lora_B' in key:
                    core_key = self._extract_core_key(key)
                    if core_key is None:
                        continue
                    if core_key not in task_vector:
                        task_vector[core_key] = {}
                    task_vector[core_key]['B'] = val

            lora_task_vectors.append(task_vector)

        return lora_task_vectors

    def get_pretrained_weights(self):
        """
        获取预训练模型的权重
        返回 canonical_key -> tensor 的字典
        """
        canonical_weights = OrderedDict()

        for core_key, real_key in self.key_mapping.items():
            if real_key in self.pt_params:
                canonical_weights[core_key] = self.pt_params[real_key]

        return canonical_weights

    def set_scaling_coeffs(self, scaling_coeffs):
        if isinstance(scaling_coeffs, float) or len(scaling_coeffs) == 1:
            self.scaling_coeffs = [scaling_coeffs] * len(self.ftms_params)
        else:
            self.scaling_coeffs = list(scaling_coeffs)

    def transform(self, merge_config=None):
        """
        Transform method (required by the framework, but not used for PACT-ISOC-RANDSVD).
        """
        return

    def add_task_parameters(self, base_model, parameters, scaling_coeffs=1.0):
        """
        将融合后的参数真实地注入到基础模型中
        使用原地加法 .add_() 确保模型权重发生真实改变
        通过 canonical_key 映射回 real_key 再 add_
        """
        sd = base_model.state_dict()

        matched = 0
        unmatched = 0
        unmatched_keys = []

        for core_key, val in parameters.items():
            if core_key in self.key_mapping:
                real_key = self.key_mapping[core_key]
                if real_key in sd:
                    try:
                        sd[real_key].add_(val.to(sd[real_key].device).to(sd[real_key].dtype))
                        matched += 1
                    except Exception as e:
                        print(f"Warning: Could not add parameter {real_key}: {e}")
                        unmatched += 1
                        unmatched_keys.append(real_key)
                else:
                    unmatched += 1
                    unmatched_keys.append(real_key)
            else:
                unmatched += 1
                unmatched_keys.append(core_key)

        print(f"Parameter injection: {matched} matched, {unmatched} unmatched")
        if unmatched > 0 and len(unmatched_keys) <= 5:
            print(f"  Unmatched keys: {unmatched_keys}")
        elif unmatched > 0:
            print(f"  First 5 unmatched keys: {unmatched_keys[:5]}")

        return base_model

    def merge(self, merge_config=None):
        """
        Execute PACT-ISOC-RANDSVD merging.
        """
        if merge_config is None:
            merge_config = self.merge_config

        merge_config['scaling_coeffs'] = self.scaling_coeffs

        lora_task_vectors = self.get_lora_task_vectors()
        pretrained_weights = self.get_pretrained_weights()

        class Config:
            pass

        config = Config()
        config.device = self.device
        config.K_ratio = merge_config.get('K_ratio', 0.8)
        config.lora_rank = merge_config.get('lora_rank', None)
        config.scaling_coeffs = self.scaling_coeffs
        config.ns_iterations = merge_config.get('ns_iterations', 5)
        config.ns_eps = merge_config.get('ns_eps', 1e-8)

        merged_sd = pact_isoc_randsvd_merge(lora_task_vectors, pretrained_weights, config)

        merged_base = deepcopy(self.pretrained_model)

        merged_model = self.add_task_parameters(merged_base, merged_sd)

        return merged_model
