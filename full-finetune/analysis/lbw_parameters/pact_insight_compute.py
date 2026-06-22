"""Module 1: compute task vectors and Fisher-like sensitivity tensors."""

from __future__ import annotations

import argparse
from datetime import datetime

import torch

from analysis.lbw_parameters.pact_insight_common import (
    build_args,
    compute_fisher_and_aux,
    git_commit,
    load_task_checkpoints,
    make_output_dir,
    tensor_dict_difference,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute PACT insight module-1 tensors.")
    p.add_argument("--task-a", required=True)
    p.add_argument("--task-b", required=True)
    p.add_argument("--model", default="ViT-B-16")
    p.add_argument("--model-location", default="models/ckpts")
    p.add_argument("--data-location", default="datasets")
    p.add_argument("--output-root", default="analysis/outputs")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--fisher-type", choices=["true", "empirical", "squared_grad"], default="true")
    p.add_argument("--num-mc-samples", type=int, default=1)
    p.add_argument("--fisher-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    args = build_args(cli.model, cli.model_location, cli.data_location, cli.batch_size, cli.device, cli.num_workers)
    state_pre, state_a, state_b, paths = load_task_checkpoints(cli.model_location, cli.model, cli.task_a, cli.task_b)
    tau_a = tensor_dict_difference(state_a, state_pre)
    tau_b = tensor_dict_difference(state_b, state_pre)
    fisher, grad_times_weight = compute_fisher_and_aux(
        state_a,
        cli.model,
        cli.task_a,
        args,
        cli.fisher_type,
        cli.num_mc_samples,
        cli.fisher_samples,
        cli.seed,
    )
    out_dir = make_output_dir(cli.output_root, cli.task_a, cli.task_b)
    payload = {
        "tau_A": tau_a,
        "tau_B": tau_b,
        "F_A": fisher,
        "grad_times_weight_A": grad_times_weight,
        "meta": {
            "module": "mod1_compute",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "task_A": cli.task_a,
            "task_B": cli.task_b,
            "model": cli.model,
            "paths": paths,
            "fisher_type": cli.fisher_type,
            "num_mc_samples": cli.num_mc_samples,
            "fisher_samples": cli.fisher_samples,
            "seed": cli.seed,
        },
    }
    torch.save(payload, out_dir / "mod1_tensors.pt")
    print(f"Saved {out_dir / 'mod1_tensors.pt'}")


if __name__ == "__main__":
    main()
