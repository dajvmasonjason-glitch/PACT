"""Shared utilities for the PACT insight experiments.

This module intentionally lives under ``analysis`` so the experiment pipeline can
be added without modifying the training/evaluation code in ``src``.
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.models.heads import get_classification_head
from src.models.modeling import ImageClassifier, ImageEncoder
from src.utils import utils
from src.utils.variables_and_paths import get_finetuned_path, get_zeroshot_path


TensorDict = Dict[str, torch.Tensor]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def task_to_val_name(task: str) -> str:
    return task if task.endswith("Val") else f"{task}Val"


def task_to_base_name(task: str) -> str:
    return task[:-3] if task.endswith("Val") else task


def build_args(
    model: str,
    model_location: str,
    data_location: str,
    batch_size: int,
    device: str,
    num_workers: int,
) -> Namespace:
    model_location = os.path.expanduser(model_location)
    data_location = os.path.expanduser(data_location)
    return Namespace(
        model=model,
        model_location=model_location,
        data_location=data_location,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        save_dir=os.path.join(model_location, model),
    )


def checkpoint_paths(model_location: str, model: str, task_a: str, task_b: str) -> Dict[str, str]:
    task_a_val = task_to_val_name(task_a)
    task_b_val = task_to_val_name(task_b)
    return {
        "pre": get_zeroshot_path(model_location, "MNISTVal", model=model),
        "A": get_finetuned_path(model_location, task_a_val, model=model),
        "B": get_finetuned_path(model_location, task_b_val, model=model),
    }


def load_checkpoint(path: str) -> TensorDict:
    state = torch.load(path, map_location="cpu")
    if "model_name" in state:
        state = dict(state)
        state.pop("model_name")
    return state


def load_task_checkpoints(model_location: str, model: str, task_a: str, task_b: str) -> Tuple[TensorDict, TensorDict, TensorDict, Dict[str, str]]:
    paths = checkpoint_paths(model_location, model, task_a, task_b)
    missing = [name for name, path in paths.items() if not os.path.exists(path)]
    if missing:
        details = ", ".join(f"{name}={paths[name]}" for name in missing)
        raise FileNotFoundError(f"Missing checkpoint(s): {details}")
    return load_checkpoint(paths["pre"]), load_checkpoint(paths["A"]), load_checkpoint(paths["B"]), paths


def git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def make_output_dir(output_root: str, task_a: str, task_b: str) -> Path:
    out = Path(output_root) / f"{task_to_base_name(task_a)}-{task_to_base_name(task_b)}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def tensor_dict_difference(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> TensorDict:
    out: TensorDict = {}
    for key, value in left.items():
        if key not in right:
            continue
        if not torch.is_floating_point(value):
            continue
        out[key] = value.detach().cpu() - right[key].detach().cpu()
    return out


def is_analyzable(name: str, tensor: torch.Tensor) -> bool:
    lname = name.lower()
    if tensor.dim() < 2:
        return False
    if "head" in lname or "classification_head" in lname:
        return False
    if "norm" in lname or "ln_" in lname or "logit_scale" in lname:
        return False
    return torch.is_floating_point(tensor)


def analyzable_keys(tensors: Mapping[str, torch.Tensor], include_regex: Optional[str] = None) -> List[str]:
    import re

    pattern = re.compile(include_regex) if include_regex else None
    keys = [k for k, v in tensors.items() if is_analyzable(k, v)]
    if pattern is not None:
        keys = [k for k in keys if pattern.search(k)]
    return keys


def image_encoder_from_state(model_name: str, state_dict: Mapping[str, torch.Tensor], device: str) -> ImageEncoder:
    encoder = ImageEncoder(model_name)
    encoder.load_state_dict(state_dict, strict=True)
    encoder.to(device)
    encoder.eval()
    return encoder


def classifier_from_state(model_name: str, state_dict: Mapping[str, torch.Tensor], dataset_name: str, args: Namespace) -> ImageClassifier:
    encoder = image_encoder_from_state(model_name, state_dict, args.device)
    head = get_classification_head(args, task_to_val_name(dataset_name)).to(args.device)
    model = ImageClassifier(encoder, head)
    model.to(args.device)
    model.eval()
    return model


def get_loader(
    dataset_name: str,
    preprocess,
    args: Namespace,
    split: str,
    max_samples: Optional[int],
    seed: int,
    shuffle: bool = False,
) -> DataLoader:
    dataset = get_dataset(
        dataset_name,
        preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    base = dataset.train_dataset if split == "train" else dataset.test_dataset
    if max_samples is not None and max_samples > 0 and max_samples < len(base):
        gen = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(base), generator=gen)[:max_samples].tolist()
        base = Subset(base, indices)
    return DataLoader(base, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)


@torch.no_grad()
def evaluate_state_dict(state_dict: Mapping[str, torch.Tensor], dataset_name: str, args: Namespace) -> float:
    model = classifier_from_state(args.model, state_dict, dataset_name, args)
    loader = get_loader(task_to_val_name(dataset_name), model.image_encoder.val_preprocess, args, "test", None, 0)
    correct = 0
    total = 0
    for batch in loader:
        batch = maybe_dictionarize(batch)
        images = batch["images"].to(args.device)
        labels = batch["labels"].to(args.device)
        logits = utils.get_logits(images, model)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(correct / max(total, 1))


def _named_image_encoder_parameters(model: ImageClassifier) -> Dict[str, torch.nn.Parameter]:
    return {
        name.removeprefix("image_encoder."): param
        for name, param in model.named_parameters()
        if name.startswith("image_encoder.") and param.requires_grad
    }


def compute_fisher_and_aux(
    state_a: Mapping[str, torch.Tensor],
    model_name: str,
    task_a: str,
    args: Namespace,
    fisher_type: str,
    num_mc_samples: int,
    max_samples: int,
    seed: int,
) -> Tuple[TensorDict, TensorDict]:
    set_seed(seed)
    model = classifier_from_state(model_name, state_a, task_a, args)
    loader = get_loader(task_to_val_name(task_a), model.image_encoder.train_preprocess, args, "train", max_samples, seed)
    params = _named_image_encoder_parameters(model)
    fisher = {name: torch.zeros_like(param, device="cpu") for name, param in params.items()}
    grad_abs = {name: torch.zeros_like(param, device="cpu") for name, param in params.items()}
    n_seen = 0

    for batch in loader:
        batch = maybe_dictionarize(batch)
        images = batch["images"].to(args.device)
        labels = batch["labels"].to(args.device)
        batch_size = labels.numel()

        logits = model(images)
        losses: List[torch.Tensor] = []
        if fisher_type == "empirical":
            losses.append(F.cross_entropy(logits, labels, reduction="mean"))
        elif fisher_type == "squared_grad":
            pseudo = logits.detach().argmax(dim=1)
            losses.append(F.cross_entropy(logits, pseudo, reduction="mean"))
        elif fisher_type == "true":
            probs = F.softmax(logits.detach(), dim=1)
            if num_mc_samples == 0:
                log_probs = F.log_softmax(logits, dim=1)
                losses.append(-(probs * log_probs).sum(dim=1).mean())
            else:
                for _ in range(num_mc_samples):
                    sampled = torch.multinomial(probs, 1).squeeze(1)
                    losses.append(F.cross_entropy(logits, sampled, reduction="mean"))
        else:
            raise ValueError(f"Unknown fisher_type: {fisher_type}")

        for loss in losses:
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=len(losses) > 1)
            with torch.no_grad():
                for name, param in params.items():
                    if param.grad is None:
                        continue
                    g = param.grad.detach()
                    fisher[name] += g.pow(2).cpu() * batch_size / len(losses)
                    grad_abs[name] += g.abs().cpu() * batch_size / len(losses)
        n_seen += batch_size

    denom = max(n_seen, 1)
    grad_times_weight = {}
    for name in fisher:
        fisher[name] /= denom
        grad_abs[name] /= denom
        grad_times_weight[name] = grad_abs[name] * state_a[name].detach().cpu().abs()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return fisher, grad_times_weight


def flatten(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().reshape(-1).cpu()


def quantile(t: torch.Tensor, q: float, max_samples: int = 10_000_000) -> torch.Tensor:
    """计算张量的分位数，对大张量使用采样以避免内存问题"""
    flat = flatten(t)
    n = flat.numel()

    # 如果张量较小，直接计算
    if n <= max_samples:
        return torch.quantile(flat, q)

    # 对大张量进行采样
    gen = torch.Generator().manual_seed(0)
    indices = torch.randperm(n, generator=gen)[:max_samples]
    sampled = flat[indices]
    return torch.quantile(sampled, q)


def equalize_masks(raw_masks: Mapping[str, torch.Tensor], seed: int, k: Optional[int] = None) -> Dict[str, torch.Tensor]:
    counts = {name: int(mask.sum().item()) for name, mask in raw_masks.items()}
    if k is None:
        positive = [count for count in counts.values() if count > 0]
        k = min(positive) if positive else 0
    gen = torch.Generator().manual_seed(seed)
    out: Dict[str, torch.Tensor] = {}
    for name, mask in raw_masks.items():
        flat_mask = mask.reshape(-1).bool()
        idx = flat_mask.nonzero(as_tuple=False).flatten()
        keep = torch.zeros_like(flat_mask)
        if k > 0 and idx.numel() > 0:
            selected = idx[torch.randperm(idx.numel(), generator=gen)[: min(k, idx.numel())]]
            keep[selected] = True
        out[name] = keep.reshape_as(mask)
    return out


def random_mask_like(shape: torch.Size, k: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    flat = torch.zeros(int(np.prod(tuple(shape))), dtype=torch.bool)
    if k > 0:
        flat[torch.randperm(flat.numel(), generator=gen)[:k]] = True
    return flat.reshape(shape)


def parse_task_pairs(value: str) -> List[Tuple[str, str]]:
    pairs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            a, b = item.split(":", 1)
        elif "-" in item:
            a, b = item.split("-", 1)
        else:
            raise ValueError(f"Task pair must look like A:B or A-B, got {item}")
        pairs.append((a.strip(), b.strip()))
    return pairs


def save_json(path: Path, data: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
