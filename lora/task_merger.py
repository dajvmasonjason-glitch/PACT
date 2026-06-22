from collections import defaultdict, OrderedDict
import torch.nn.functional as F
from tqdm.auto import tqdm
from copy import deepcopy
from time import time
from torch import nn
import torch
import pdb

from utils import get_merging_fn, get_mask_fn
from masking_ops import masked_merge
from pact_isoc import PactIsoMerger
from pact_ta import PactTAMerger
from pact_isoc_rsvd import PactIsoRsvdMerger
from pact_ta_rsvd import PactTaRsvdMerger


class VectorOps(nn.Module):
    def directions_to_reps(self, directions):
        if isinstance(directions, list):
            return [self.directions_to_reps(direction) for direction in directions]
        return torch.nn.utils.parameters_to_vector(
            [value.reshape(-1) for key, value in directions.items()]
        )
        
    def rep_to_state_dict(self, vector, state_dict, remove_keys=[]):
        if isinstance(vector, list) or len(vector.shape) == 2:
            return [self.rep_to_state_dict(v, state_dict, remove_keys) for v in vector]
        # create a reference dict to define the order of the vector
        reference_dict = deepcopy(state_dict)
        for key in remove_keys:
            if key in reference_dict:
                del reference_dict[key]
        sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

        # create a shared state dict using the refence dict
        torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

        # add back the encoder and decoder embedding weights.
        if "transformer.shared.weight" in sorted_reference_dict:
            for key in remove_keys:
                sorted_reference_dict[key] = sorted_reference_dict[
                    "transformer.shared.weight"
                ]
        return sorted_reference_dict
    
    def mask_to_state_dict(self, mask, state_dict, remove_keys=[]):
        if isinstance(mask, list):
            return [self.mask_to_state_dict(m, state_dict, remove_keys) for m in mask]
        return self.rep_to_state_dict(mask, state_dict, remove_keys)
    
    def forward(self, directions, merging_fn, merge_config):
        vectors = self.directions_to_reps(directions)
        merged_vector,rows_to_keep, topk_mask = merging_fn(vectors)
        mask_sd = self.rep_to_state_dict(topk_mask, directions[0])
        
        ties_mask = [dict() for _ in range(len(rows_to_keep))]
        for idx in range(len(rows_to_keep)):
            ties_mask[idx] = self.rep_to_state_dict(rows_to_keep[idx], directions[0])
        sd = self.rep_to_state_dict(merged_vector, directions[0])
        
        return sd, ties_mask


class TaskMerger(nn.Module):
    def __init__(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        super().__init__()
        
        self.device = device
        self.scaling_coeffs = torch.tensor([1.] * len(finetuned_models))
        self.param_handler = param_handler
        self.finetuned_models = finetuned_models
        self.ftms_params = [param_handler(ft_model) for ft_model in finetuned_models]
        self.pretrained_model = pretrained_model.cpu()
        self.pt_params = self.pretrained_model.state_dict()
        self.merge_config = merge_config

    def randbin(self, M, N, P):
        P = 1-P
        return torch.randint(2, size=(M, N), dtype=torch.float32).bernoulli(P)
    
    def apply_dare(self, ftms_params, p, dare_seed = 0):
        print("DARE seed: ", dare_seed)
        torch.manual_seed(dare_seed)
        finetuned_directions = []
        for ftm_params in ftms_params:
            direction_sd = {}
            for key, finetuned_val in ftm_params.items():
                direction_sd[key] = finetuned_val * self.randbin(finetuned_val.shape[0], finetuned_val.shape[1], p) * (1/(1-p))
                # pdb.set_trace()
            finetuned_directions += [OrderedDict(sorted(direction_sd.items()))]
        return finetuned_directions

    def get_task_directions(self, ptm_params, ftms_params):
        finetuned_directions = []
        for ftm_params in ftms_params:
            direction_sd = {}
            
            for key, finetuned_val in ftm_params.items():
                if key not in ptm_params:
                    ptm_val = torch.zeros_like(finetuned_val)
                else:
                    ptm_val = ptm_params[key]
                direction_sd[key] = finetuned_val - ptm_val
            finetuned_directions += [OrderedDict(sorted(direction_sd.items()))]
        return finetuned_directions
    
    def set_scaling_coeffs(self, scaling_coeffs):
        if isinstance(scaling_coeffs, float) or len(scaling_coeffs) == 1:
            self.scaling_coeffs = torch.tensor([scaling_coeffs] * len(self.ftms_params))
        else:
            self.scaling_coeffs = torch.tensor(scaling_coeffs)
        
    def get_layer_names(self, state_dict):
        layer_names = defaultdict(lambda: dict())
        for key in state_dict:
            if ('.weight' in key) or ('_weight' in key):
                strip_key = key.replace('.weight', '').replace('_weight', '')
                layer_names[strip_key]['weight'] = key
            elif ('.bias' in key) or ('_bias' in key):
                strip_key = key.replace('.bias', '').replace('_bias', '')
                layer_names[strip_key]['bias'] = key
            else:
                layer_names[key]['other'] = key + ':other'
        return layer_names
    
    def add_task_parameters(self, base_model, parameters, concat_across_output = True, scaling_coeffs=1.):
        if isinstance(parameters, list):
            return [self.add_task_parameters(
                deepcopy(base_model), 
                parameter,
                concat_across_output=concat_across_output, 
                scaling_coeffs=scaling_coeffs
            ) for parameter in parameters]
        sd = base_model.state_dict()
        for key, val in parameters.items():
            try:
                if (concat_across_output):
                    sd[key].add_(val.cpu() * scaling_coeffs)
                else:
                    sd[key].add_(val.T.cpu() * scaling_coeffs)
            except:
                pdb.set_trace()
        return base_model
    
    def directions_to_matrices(self, directions, reference_layer_names=None):
        if isinstance(directions, list):
            return [self.directions_to_matrices(direction, reference_layer_names) for direction in directions]
        
        if reference_layer_names is None:
            layer_names = self.get_layer_names(directions)
        else:
            layer_names = reference_layer_names

        matrices = {}
        for layer_name, parameter_names in layer_names.items():
            if 'other' in parameter_names:
                other_parameter = directions[parameter_names['other'].replace(':other', '')].to(torch.float32)
                # Ensure parameters are always two dimensional
                if len(other_parameter.shape) == 1: # e.g., class token, positional embeddings
                    other_parameter = other_parameter[None, :]
                elif len(other_parameter.shape) > 2: # e.g., patch embeddings
                    other_parameter = other_parameter.flatten(1)
                matrices[layer_name + ':other'] = other_parameter
            elif 'weight' in parameter_names:
                weight_name = parameter_names['weight']
                weight = directions[weight_name]
                if 'norm' in layer_name or 'ln' in layer_name:
                    weight = torch.diag(weight)
                matrices[layer_name] = weight.flatten(1)
                if 'bias' in parameter_names:
                    bias = directions[parameter_names['bias']]
                    matrices[layer_name] = torch.concat((matrices[layer_name], bias.reshape(-1, 1)), dim=1)
        return matrices
    
    def matrix_to_state_dict(self, matrix, state_dict, remove_keys=[]):
        if isinstance(matrix, list):
            return [self.matrix_to_state_dict(m, state_dict) for m in matrix]
        
        reference_dict = deepcopy(state_dict)
        for key in remove_keys:
            if key in reference_dict:
                del reference_dict[key]
                
        layer_names = self.get_layer_names(reference_dict)
        merged_state_dict = {}
        for layer_name, value in matrix.items():
            try:
                parameter_types = layer_names[layer_name.replace(':other', '')]
                if 'other' in parameter_types:
                    name = parameter_types['other'].replace(':other', '')
                    merged_state_dict[name] = value.reshape(reference_dict[name].shape)
                else:
                    if 'bias' in parameter_types: 
                        bias_index = value.shape[1] - 1
                        value, bias = value[:, :bias_index], value[:, -1].flatten()
                        merged_state_dict[parameter_types['bias']] = bias
                    if 'norm' in layer_name or 'ln' in layer_name:
                        value = torch.diagonal(value)
                    name = parameter_types['weight']
                    merged_state_dict[name] = value.reshape(*(reference_dict[name].shape))
            except:
                pdb.set_trace()
                    
        # add back the encoder and decoder embedding weights.
        if "transformer.shared.weight" in merged_state_dict:
            for key in remove_keys:
                merged_state_dict[key] = merged_state_dict[
                    "transformer.shared.weight"
                ]
        return merged_state_dict
    
    def transform(self, *args, **kwargs):
        return

class VectorMerger(TaskMerger):
    def __init__(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        super().__init__(
            finetuned_models=finetuned_models, 
            pretrained_model=pretrained_model, 
            param_handler=param_handler, 
            device=device,
            merge_config=merge_config
        )
        
        self.representation_helper = VectorOps()
    
    def merge(self, merge_config={'merge_method': 'tv'}):
        print(merge_config['merge_method'])
        merging_fn = lambda x: get_merging_fn(merge_config['merge_method'])(
            x, **merge_config, weights=self.scaling_coeffs
        )

        ptm_reference_params = self.param_handler(self.pretrained_model).get_ft_parameters()
        ftms_relevant_params = [ftm.get_ft_parameters() for ftm in self.ftms_params]
        ftms_task_dirs = self.get_task_directions(ptm_reference_params, ftms_relevant_params)
        
        if merge_config.get('dare', False):
            ftms_task_dirs = self.apply_dare(
                ftms_task_dirs, merge_config['dare_pruning_coeffs'], merge_config['dare_seed']
            )
        
        merged_sd = self.representation_helper(ftms_task_dirs, merging_fn, merge_config)

        merged_base = deepcopy(self.pretrained_model)
        if len(merged_sd) == 2:
            merged_sd, mask = merged_sd
        merged_model = self.add_task_parameters(merged_base, merged_sd)
        
        return merged_model

class SVDMerger(TaskMerger):
    def __init__(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        super().__init__(
            finetuned_models=finetuned_models, 
            pretrained_model=pretrained_model, 
            param_handler=param_handler, 
            device=device,
            merge_config=merge_config
        )
        
        self.layer_names = self.get_layer_names(self.ftms_params[0].get_ft_parameters())
        self.representation_helper = VectorOps()
        self.ingredients = None
            
    def variable_extend_dim(self, elements, op_dim):
        if isinstance(elements, list):
            return [self.variable_extend_dim(element, op_dim) for element in elements]
        while len(elements.shape) < (op_dim+1):
            elements = elements.unsqueeze(-1)
        return elements
    
    def dict_of_concat_matrices(self, list_of_dictmatrices, dim=0, concat_across_output = True):
        dict2matrix_stack = defaultdict(lambda: list())
        for dict2matrix in list_of_dictmatrices:
            for key, val in dict2matrix.items():
                if(concat_across_output == True):
                    dict2matrix_stack[key] += [val.to(self.device)]
                else: 
                    dict2matrix_stack[key] += [val.T.to(self.device)]
                
        for key, list_of_vals in dict2matrix_stack.items():
            # Extend dim as necessary
            list_of_vals = self.variable_extend_dim(list_of_vals, op_dim=dim)
            dict2matrix_stack[key] = torch.concat(list_of_vals, dim=dim)
        return dict2matrix_stack
    
    def reconstruct_merged_sd(self, U_sd, sV_sd):
        if isinstance(sV_sd, list):
            if isinstance(U_sd, list):
                return [self.reconstruct_merged_sd(U, sV) for U, sV in zip(U_sd, sV_sd)]
            return [self.reconstruct_merged_sd(U_sd, sV) for sV in sV_sd]
        sd = {}
        for key, U in U_sd.items():
            sd[key] = (U @ sV_sd[key]).to(torch.float32)
        return sd
        
    def apply_svd(self, ft_params, concat_across_output = True):
        UsV_dict = {}
        basis_dict = {} # basis for reconstruction
        s_compositions_dict = [dict() for _ in range(len(ft_params))]
        V_compositions_dict = [dict() for _ in range(len(ft_params))] # basis composition information per task
        
        print(f'Calculating SVD over {len(ft_params)} models. S > 1e-5')
        concated_ft_params = self.dict_of_concat_matrices(ft_params, dim=1, concat_across_output = concat_across_output)
        for key, val in tqdm(concated_ft_params.items(), desc='Obtaining SVDs...'):
            U, s, V = torch.linalg.svd(val.to(torch.float64), full_matrices=False)
            # Keep only supported basis components
            U = U[:, s > 1e-5].type(torch.float32)
            V = V[s > 1e-5].type(torch.float32)
            s = s[s > 1e-5].type(torch.float32)
            UsV_dict[key] = {'U': deepcopy(U), 's':deepcopy(s), 'V':deepcopy(V) }
            # Set all s to be the same scale
            s[s <= 1e-5] = 0
            cat_hidden_dim = V.shape[1] // len(ft_params)

            basis_dict[key] = U.cpu()
            sV_concat = V
            Vs = list(torch.split(sV_concat, cat_hidden_dim, dim=1))
            for idx, V in enumerate(Vs):
                V = torch.diag(s) @ V # Simple and safe for all merging methods we use.
                s_model = s / s

                s_compositions_dict[idx][key] = s_model.cpu()
                V_compositions_dict[idx][key] = V.cpu()
        return basis_dict, s_compositions_dict, V_compositions_dict, UsV_dict
    
    def apply_Ss_on_Vs(self, task_Vs, task_Ss):
        task_sVs = [dict() for i in range(len(task_Vs))]
        for idx, (Vs, Ss) in enumerate(zip(task_Vs, task_Ss)):
            for key, V in Vs.items():
                if len(Ss[key].shape) == 2:
                    task_sVs[idx][key] = Ss[key] @ V
                else:
                    task_sVs[idx][key] = torch.diag(Ss[key]) @ V
        return task_sVs
    
    def remove_others(self, ftms_mats):
        other_mats = [dict() for i in range(len(ftms_mats))]
        transform_mats = [dict() for i in range(len(ftms_mats))]
        
        for m_idx, ftm_mats in enumerate(ftms_mats):
            for key, val in ftm_mats.items():
                if ':other' in key:
                    other_mats[m_idx][key] = val
                elif 'modules_to_save' in key:
                    other_mats[m_idx][key] = val
                else:
                    transform_mats[m_idx][key] = val
        print(f'Len other: {len(other_mats[0])}| len: transform: {len(transform_mats[0])}')
        return other_mats, transform_mats
    
    def add_others(self, ftms_mats, ftms_others):
        if isinstance(ftms_mats, list):
            return [self.add_others(ftms_mat, ftms_other) for ftms_mat, ftms_other in zip(ftms_mats, ftms_others)]
        
        for key, val in ftms_others.items():
            ftms_mats[key] = val
        return ftms_mats
    
    def transform(self, merge_config):
        # Setup parameters
        ptm_reference_params = deepcopy(self.param_handler(self.pretrained_model).get_ft_parameters())
        ftms_relevant_params = [ftm.get_ft_parameters() for ftm in self.ftms_params]
        ftms_task_dirs = self.get_task_directions(ptm_reference_params, ftms_relevant_params)
        
        ftms_task_mats = self.directions_to_matrices(ftms_task_dirs)
        ftms_others, ftms_mats = self.remove_others(ftms_task_mats)

        U, task_Ss, task_sVs, UsV_dict = self.apply_svd(
            ftms_mats,
            concat_across_output = merge_config.get('concat_across_output', True),
        )
            
        self.ingredients = {
            'ftms_relevant_params': ftms_relevant_params,
            'ftms_others': ftms_others,
            'ptm_reference_params': ptm_reference_params,
            'U': U,
            'task_Ss': task_Ss,
            'task_sVs': task_sVs,
            'UsV_dict': UsV_dict,
        }
        
        if merge_config.get('ingredients_path') is not None:
            torch.save(self.ingredients, merge_config['ingredients_path'])
    
    def merge(self, merge_config):
        if merge_config.get('ingredients_path') is not None:
            ingredients = torch.load(merge_config['ingredients_path'])
        else:
            ingredients = deepcopy(self.ingredients)
            
        ftms_others = ingredients['ftms_others']
        ptm_reference_params = ingredients['ptm_reference_params']
        U = ingredients['U']
        task_Ss = ingredients['task_Ss']
        task_sVs = ingredients['task_sVs']
        
        if merge_config.get('dare', False):
            print("Applying DARE")
            task_sVs = self.apply_dare(
                task_sVs, merge_config['dare_pruning_coeffs'], merge_config['dare_seed']
            )
        
        representations = self.representation_helper.directions_to_reps(task_sVs)
        ftms_reps = representations
        
        mask_fn = get_mask_fn(merge_config['merge_method'])
        masks = mask_fn(ftms_reps, **merge_config)
        ftms_reps = torch.vstack(ftms_reps).clone()
        masked_sVs = ftms_reps * masks
        pre_merge_sVs_dict = self.representation_helper.rep_to_state_dict(masked_sVs, task_sVs[0])
        rescaled_Vs = self.apply_Ss_on_Vs(pre_merge_sVs_dict, task_Ss)
        
        rescaled_Vs = torch.stack(self.representation_helper.directions_to_reps(rescaled_Vs), dim=0)
        merged_sV_ = masked_merge(
            merge_func=merge_config.get('merging_type'), vectors=rescaled_Vs, weights=self.scaling_coeffs
        )
        merged_sV_sd = self.representation_helper.rep_to_state_dict(merged_sV_, task_sVs[0])
        
        merged_sd = self.reconstruct_merged_sd(U, merged_sV_sd)
        
        merging_fn = lambda x: get_merging_fn(merge_config['merge_method'])(
            x, **merge_config, weights=self.scaling_coeffs
        )
        if merge_config.get('merge_other_params', False):
            merged_others,_ = self.representation_helper(ftms_others,  merging_fn=merging_fn)
            merged_sd = self.add_others(merged_sd, merged_others)
        
        merged_sd = self.matrix_to_state_dict(merged_sd, ptm_reference_params)
        # Add merged sd to the ptm
        merged_base = deepcopy(self.pretrained_model)
        merged_model = self.add_task_parameters(merged_base, merged_sd,  concat_across_output = merge_config.get('concat_across_output', True))
        return merged_model
    

def get_merge_handler(rep_type):
    if rep_type == 'svd-vector':
        return SVDMerger
    elif rep_type == 'vector':
        return VectorMerger
    elif rep_type == 'iso':
        return IsoMerger
    elif rep_type == 'pact-isoc':
        return PactIsoMerger
    elif rep_type == 'pact-ta':
        return PactTAMerger
    elif rep_type == 'pact-isoc-rsvd':
        return PactIsoRsvdMerger
    elif rep_type == 'pact-ta-rsvd':
        return PactTaRsvdMerger


class IsoMerger(TaskMerger):
    def __init__(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        super().__init__(
            finetuned_models=finetuned_models, 
            pretrained_model=pretrained_model, 
            param_handler=param_handler, 
            device=device,
            merge_config=merge_config
        )
        self.num_tasks = len(finetuned_models)

    def iso_c_merge(self, task_dirs):
        """
        ISO-C: Isotropic Merging via SVD with singular value equalization.
        
        Core idea: Average task vectors, apply SVD, then set all singular values 
        to their mean to achieve isotropy in the parameter space.
        """
        print("Computing ISO-C merging...")
        new_vector = {}
        
        with torch.no_grad():
            for key in task_dirs[0].keys():
                tvs = [td[key].to(self.device) for td in task_dirs]
                new_vector[key] = sum(tvs) / len(tvs)
                
                if len(tvs[0].shape) == 2 and "text_projection" not in key:
                    new_vector[key] *= len(tvs)
                    U, S, V = torch.linalg.svd(new_vector[key], full_matrices=False)
                    S_mean = torch.ones_like(S) * S.mean()
                    
                    new_vector[key] = torch.linalg.multi_dot(
                        (
                            U,
                            torch.diag(S_mean),
                            V,
                        )
                    )
        
        return new_vector

    def isoc_ns_merge(self, task_dirs, num_iterations=5, eps=1e-8, use_bfloat16=True):
        """
        ISO-C with Newton-Schulz iteration instead of SVD.
        
        Memory-efficient isotropic merging using Newton-Schulz iteration to approximate
        the SVD-based singular value equalization, suitable for large language models.
        
        Args:
            task_dirs: List of task direction dictionaries
            num_iterations: Number of Newton-Schulz iterations (default: 5)
            eps: Numerical stability epsilon (default: 1e-8)
            use_bfloat16: Whether to use BFloat16 for computation (default: True)
        """
        print(f"Computing ISO-C with Newton-Schulz (iterations={num_iterations})...")
        new_vector = {}
        
        device = torch.device(self.device) if isinstance(self.device, (str, int)) else self.device
        
        with torch.no_grad():
            for key in task_dirs[0].keys():
                tvs = [td[key].to(device) for td in task_dirs]
                
                W_sum = sum(tvs)
                new_vector[key] = W_sum
                
                is_2d_matrix = (len(tvs[0].shape) == 2) and ("text_projection" not in key)
                
                if not is_2d_matrix:
                    new_vector[key] = W_sum / len(task_dirs)
                    continue
                
                W = W_sum
                m, n = W.shape
                min_dim = min(m, n)
                
                original_dtype = W.dtype
                compute_dtype = torch.bfloat16 if use_bfloat16 and device.type == 'cuda' else original_dtype
                W = W.to(compute_dtype)
                
                norm_W = torch.norm(W, p='fro')
                if norm_W < eps:
                    new_vector[key] = new_vector[key].to(original_dtype)
                    continue
                
                X = W / (norm_W + eps)
                
                for _ in range(num_iterations):
                    if m >= n:
                        A = X.t() @ X
                        A.mul_(-1).add_(3.0 * torch.eye(n, device=device, dtype=compute_dtype))
                        X = torch.matmul(X, A).mul_(0.5)
                        del A
                    else:
                        A = X @ X.t()
                        A.mul_(-1).add_(3.0 * torch.eye(m, device=device, dtype=compute_dtype))
                        X = torch.matmul(A, X).mul_(0.5)
                        del A
                
                sum_sigma = torch.sum(X * W)
                mean_sigma = sum_sigma / min_dim
                
                W_isoc = mean_sigma * X
                
                new_vector[key] = W_isoc.to(original_dtype)
                
                del X, W, norm_W, sum_sigma, mean_sigma, W_isoc
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        return new_vector

    def iso_cts_merge(self, task_dirs, common_space_fraction=0.8):
        """
        ISO-CTS: Isotropic Merging with Common and Task-Specific space separation.
        
        Core idea: 
        1. Extract common space from sum of all task vectors
        2. For each task, compute task-specific space by removing common space projection
        3. Concatenate common space and task-specific spaces
        4. Orthogonalize the combined spaces
        5. Set singular values to mean for isotropy
        """
        print(f"Computing ISO-CTS merging with common_space_fraction={common_space_fraction}...")
        new_vector = {}
        num_tasks = len(task_dirs)
        
        with torch.no_grad():
            for key in task_dirs[0].keys():
                shape_ = task_dirs[0][key].shape
                
                is_2d_matrix = (len(shape_) == 2) and ("text_projection" not in key)
                
                if not is_2d_matrix:
                    print(f"Combining by avg {key}...")
                    for i, td in enumerate(task_dirs):
                        vec = td[key].to(self.device)
                        if i == 0:
                            new_vector[key] = vec.clone()
                        else:
                            new_vector[key] += (vec - new_vector[key]) / (i + 1)
                    continue
                
                print(f"Computing common space using sum for {key}...")
                combined_w = sum([td[key].to(self.device) for td in task_dirs])
                
                common_space_index_s = int(min(shape_) * common_space_fraction)
                _task_specific_total_space_index_s = round((min(shape_) - common_space_index_s) / num_tasks) * num_tasks
                common_space_index_s = min(shape_) - _task_specific_total_space_index_s
                
                u, s, v = torch.linalg.svd(combined_w, full_matrices=False)
                common_space_u = u[:, :common_space_index_s]
                common_space_s = s[:common_space_index_s]
                common_space_v = v[:common_space_index_s, :]
                
                n_dims_per_task = int((min(shape_) - common_space_index_s) / num_tasks)
                
                for i, td in enumerate(task_dirs):
                    w = td[key].to(self.device)
                    
                    w_ts = w - common_space_u @ common_space_u.T @ w
                    u_ts, s_ts, v_ts = torch.linalg.svd(w_ts, full_matrices=False)
                    
                    if i == 0:
                        combined_space_u = torch.zeros_like(u_ts, device=self.device)
                        combined_space_s = torch.zeros_like(s_ts, device=self.device)
                        combined_space_v = torch.zeros_like(v_ts, device=self.device)
                    
                    combined_space_u[:, i * n_dims_per_task : (i + 1) * n_dims_per_task] = u_ts[:, :n_dims_per_task]
                    combined_space_s[i * n_dims_per_task : (i + 1) * n_dims_per_task] = s_ts[:n_dims_per_task]
                    combined_space_v[i * n_dims_per_task : (i + 1) * n_dims_per_task, :] = v_ts[:n_dims_per_task, :]
                
                combined_space_u[:, num_tasks * n_dims_per_task : num_tasks * n_dims_per_task + common_space_index_s] = common_space_u
                combined_space_s[num_tasks * n_dims_per_task : num_tasks * n_dims_per_task + common_space_index_s] = common_space_s
                combined_space_v[num_tasks * n_dims_per_task : num_tasks * n_dims_per_task + common_space_index_s, :] = common_space_v
                
                u_combined_space_u, s_combined_space_u, v_combined_space_u = torch.linalg.svd(combined_space_u, full_matrices=False)
                u_combined_space_v, s_combined_space_v, v_combined_space_v = torch.linalg.svd(combined_space_v, full_matrices=False)
                combined_space_u = u_combined_space_u @ v_combined_space_u
                combined_space_v = u_combined_space_v @ v_combined_space_v
                
                combined_space_s = torch.ones_like(combined_space_s) * combined_space_s.mean()
                
                new_vector[key] = torch.linalg.multi_dot(
                    (
                        combined_space_u,
                        torch.diag(combined_space_s),
                        combined_space_v,
                    )
                )
        
        return new_vector

    def merge(self, merge_config):
        """
        Main merge function for ISO methods.
        """
        print(f"ISO Merge method: {merge_config['merge_method']}")
        
        ptm_reference_params = self.param_handler(self.pretrained_model).get_ft_parameters()
        ftms_relevant_params = [ftm.get_ft_parameters() for ftm in self.ftms_params]
        task_dirs = self.get_task_directions(ptm_reference_params, ftms_relevant_params)
        
        task_dirs_dict = [OrderedDict(td) for td in task_dirs]
        
        if merge_config['merge_method'] == 'iso_c':
            merged_sd = self.iso_c_merge(task_dirs_dict)
        elif merge_config['merge_method'] == 'isoc_ns':
            num_iterations = merge_config.get('ns_iterations', 5)
            eps = merge_config.get('ns_eps', 1e-8)
            use_bfloat16 = merge_config.get('use_bfloat16', True)
            merged_sd = self.isoc_ns_merge(task_dirs_dict, num_iterations, eps, use_bfloat16)
        elif merge_config['merge_method'] == 'iso_cts':
            common_fraction = merge_config.get('common_space_fraction', 0.8)
            merged_sd = self.iso_cts_merge(task_dirs_dict, common_fraction)
        else:
            raise ValueError(f"Unknown ISO method: {merge_config['merge_method']}")
        
        merged_base = deepcopy(self.pretrained_model)
        scaling_coeffs = merge_config.get('scaling_coeffs', 1.0)
        merged_model = self.add_task_parameters(merged_base, merged_sd, scaling_coeffs=scaling_coeffs)

        return merged_model
