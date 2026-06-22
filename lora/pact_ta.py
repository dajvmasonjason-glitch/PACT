import torch
from collections import OrderedDict
from copy import deepcopy


def extract_pretrained_core_space_fixed(W_pre, K_ratio):
    """
    基于固定比例提取预训练核心空间
    对预训练权重矩阵进行 SVD，保留固定比例的维度

    Args:
        W_pre: 预训练权重矩阵 (d_out, d_in)
        K_ratio: 预训练核心空间比例，例如 0.8 表示保留 80% 的维度

    Returns:
        V_pre_K: 预训练核心特征子空间基底 (d_in, K)
        K: 确定的保留维度
    """
    _, S, V = torch.linalg.svd(W_pre, full_matrices=False)

    total_dims = len(S)
    K = int(total_dims * K_ratio)
    K = max(1, K)

    V_pre_K = V[:K, :].t()
    return V_pre_K, K


def extract_task_explicit_space_fixed(A_t, k):
    """
    基于固定维度提取任务显式变化空间
    直接对 LoRA 的 A 矩阵进行 SVD（速度优化核心）

    Args:
        A_t: LoRA 的 A 矩阵 (r, d_in)
        k: 保留的维度数

    Returns:
        V_t_k: 任务显式变化子空间基底 (d_in, k)
        k_actual: 实际保留的维度
    """
    _, S, V = torch.linalg.svd(A_t, full_matrices=False)

    k_actual = min(k, len(S))

    V_t_k = V[:k_actual, :].t()
    return V_t_k, k_actual


def compute_implicit_reliance_space(V_pre_K, V_t_k):
    """
    计算任务的隐式依赖空间
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
    无干涉正交过滤
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

    Args:
        state_dict: 模型的 state_dict

    Returns:
        lora_pairs: 字典，key 为 base 权重名，value 为 {'A': A矩阵, 'B': B矩阵}
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


def pact_ta_merge(lora_task_vectors, pretrained_vector, config):
    """
    PACT-TA 融合方法 (LoRA 版本)

    PACT-TA (Pretrained Core Avoidance + Task Arithmetic):
    结合了 PACT 的正交过滤机制和 Task Arithmetic 的简单加权求和。

    核心流程：
    1. 提取预训练核心空间和任务显式变化空间（SVD on LoRA A）
    2. 计算每个任务的隐式依赖空间（relies on pretrained core）
    3. 执行 PACT 正交过滤：每个任务的 delta 剔除其他任务隐式空间上的分量
    4. 使用 Task Arithmetic（加权求和/平均）合并过滤后的任务向量

    与 PACT-IsoC 的区别：
    - PACT-IsoC 在 Phase 3 使用 SVD 奇异值均衡化
    - PACT-TA 在 Phase 3 使用 Task Arithmetic，更加轻量且易于解释

    Args:
        lora_task_vectors: LoRA 任务向量列表，每个元素包含 {'A': A矩阵, 'B': B矩阵, 'base_name': 基础权重名}
        pretrained_vector: 预训练模型向量 (state_dict 格式)
        config: 配置对象，包含:
            - device: 计算设备
            - K_ratio: 预训练核心空间比例 (默认 0.8)
            - lora_rank: LoRA 的秩 r (默认从配置读取)
            - scaling_coeffs: 缩放系数
            - ta_merging_type: Task Arithmetic 聚合方式 ('sum' 或 'mean', 默认 'mean')

    Returns:
        new_vector: 融合后的向量字典，键为 base 模型的权重名
    """
    device = config.device if hasattr(config, 'device') else 'cuda'
    if isinstance(device, int):
        device = f'cuda:{device}'

    K_ratio = getattr(config, 'K_ratio', 0.8)
    lora_rank = getattr(config, 'lora_rank', None)
    scaling_coeffs = getattr(config, 'scaling_coeffs', 1.0)
    ta_merging_type = getattr(config, 'ta_merging_type', 'mean')  # 'sum' or 'mean'

    if isinstance(scaling_coeffs, float):
        scaling_coeffs = [scaling_coeffs] * len(lora_task_vectors)

    num_tasks = len(lora_task_vectors)

    print(f"PACT-TA (LoRA) merging with {num_tasks} tasks")
    print(f"  K_ratio={K_ratio}")
    print(f"  TA merging type={ta_merging_type}")

    with torch.no_grad():
        implicit_spaces = {}
        filtered_deltas = []

        print("Phase 1: Extracting feature bases and implicit spaces...")
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

                V_pre_K, K = extract_pretrained_core_space_fixed(W_pre, K_ratio)
                V_t_k, k_actual = extract_task_explicit_space_fixed(A_t, k)

                if task_idx == 0:
                    print(f"  {base_name}: K={K}, k={k_actual}")

                del W_pre

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

        print("Phase 2: PACT orthogonal filtering...")
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

        print(f"Phase 3: Executing Task Arithmetic ({ta_merging_type})...")
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

            if ta_merging_type == 'sum':
                new_vector[base_name] = sum(deltas).cpu()
            else:  # 'mean'
                new_vector[base_name] = (sum(deltas) / len(deltas)).cpu()

            del deltas
            torch.cuda.empty_cache()

    return new_vector


class PactTAMerger:
    """
    PACT-TA Merger for LoRA-finetuned models.

    PACT-TA (Pretrained Core Avoidance + Task Arithmetic):
    Combines the PACT orthogonal filtering mechanism with simple
    Task Arithmetic (weighted sum or average) for model merging.

    Key features:
    1. Directly SVD the small LoRA A matrix for efficiency
    2. Extract implicit reliance space from pretrained core
    3. Apply orthogonal filtering to protect other tasks' sacred spaces
    4. Use Task Arithmetic (sum/mean) for final aggregation

    Compared to PACT-IsoC:
    - PACT-IsoC uses SVD-based isotropic equalization in Phase 3
    - PACT-TA uses simpler Task Arithmetic, which is more interpretable
      and may better preserve task-specific characteristics.
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
        core_parts = []
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
        Transform method (required by the framework, but not used for PACT-TA).
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
        Execute PACT-TA merging.
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
        config.ta_merging_type = merge_config.get('ta_merging_type', 'mean')

        merged_sd = pact_ta_merge(lora_task_vectors, pretrained_weights, config)

        merged_base = deepcopy(self.pretrained_model)

        merged_model = self.add_task_parameters(merged_base, merged_sd)

        return merged_model
