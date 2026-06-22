import torch
from typing import Dict, List

from .ties_utils import resolve_sign, disjoint_merge
from .utils import topk_values_mask


def ties_merging_layerwise(
    task_vectors_dict: Dict[str, torch.Tensor],
    reset_thresh=None,
    merge_func=""
):
    """
    Perform TIES merging layer-by-layer instead of on flattened parameters.

    Args:
        task_vectors_dict: Dictionary mapping parameter names to stacked task vectors
                          Shape: (num_tasks, *param_shape) for each parameter
        reset_thresh: Top-k threshold for sparsification (percentage to keep)
        merge_func: Merge function to use (e.g., "dis-mean")

    Returns:
        merged_dict: Dictionary mapping parameter names to merged parameters
    """
    merged_dict = {}

    for param_name, task_vectors in task_vectors_dict.items():
        print(f"Processing layer: {param_name}, shape: {task_vectors.shape}")

        # Flatten each task vector for this layer
        original_shape = task_vectors.shape
        num_tasks = original_shape[0]
        param_shape = original_shape[1:]

        # Flatten to (num_tasks, num_params)
        flat_task_vectors = task_vectors.reshape(num_tasks, -1)

        # Apply TIES algorithm on this layer
        all_checks = flat_task_vectors.clone()
        updated_checks, *_ = topk_values_mask(all_checks, K=reset_thresh, return_mask=False)

        print(f"  RESOLVING SIGN for {param_name}")
        final_signs = resolve_sign(updated_checks)
        assert final_signs is not None

        print(f"  Disjoint AGGREGATION: {merge_func} for {param_name}")
        merged_flat = disjoint_merge(updated_checks, merge_func, final_signs)

        # Reshape back to original parameter shape
        merged_param = merged_flat.reshape(param_shape)
        merged_dict[param_name] = merged_param

    return merged_dict
