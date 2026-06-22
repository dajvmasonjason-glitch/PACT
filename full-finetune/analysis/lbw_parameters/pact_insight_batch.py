"""Batch runner for the PACT insight pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from analysis.lbw_parameters.pact_insight_common import make_output_dir, parse_task_pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run mod1 -> mod2 -> mod3 for task pairs.")
    p.add_argument("--task-pairs", required=True, help="Comma-separated pairs, e.g. SUN397:Cars,EuroSAT:SVHN")
    p.add_argument("--model", default="ViT-B-16")
    p.add_argument("--model-location", default="models/ckpts")
    p.add_argument("--data-location", default="datasets")
    p.add_argument("--output-root", default="analysis/outputs")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--fisher-type", default="true", choices=["true", "empirical", "squared_grad"])
    p.add_argument("--num-mc-samples", type=int, default=1)
    p.add_argument("--fisher-samples", type=int, default=1000)
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-mod3", action="store_true")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    cli = parse_args()
    pairs = parse_task_pairs(cli.task_pairs)
    for task_a, task_b in pairs:
        out_dir = make_output_dir(cli.output_root, task_a, task_b)
        common = [
            "--task-a", task_a,
            "--task-b", task_b,
            "--model", cli.model,
            "--model-location", cli.model_location,
            "--data-location", cli.data_location,
            "--device", cli.device,
            "--batch-size", str(cli.batch_size),
            "--num-workers", str(cli.num_workers),
            "--seed", str(cli.seed),
        ]
        run([
            sys.executable, "-m", "analysis.lbw_parameters.pact_insight_compute",
            *common,
            "--output-root", cli.output_root,
            "--fisher-type", cli.fisher_type,
            "--num-mc-samples", str(cli.num_mc_samples),
            "--fisher-samples", str(cli.fisher_samples),
        ])
        mod1 = out_dir / "mod1_tensors.pt"
        run([
            sys.executable, "-m", "analysis.lbw_parameters.pact_insight_masks",
            "--mod1-path", str(mod1),
            "--n-permutations", str(cli.n_permutations),
            "--seed", str(cli.seed),
        ])
        if not cli.skip_mod3:
            run([
                sys.executable, "-m", "analysis.lbw_parameters.pact_insight_ablation",
                *common,
                "--mod1-path", str(mod1),
                "--mod2-path", str(out_dir / "mod2_masks.pt"),
                "--num-seeds", str(cli.num_seeds),
            ])


if __name__ == "__main__":
    main()
