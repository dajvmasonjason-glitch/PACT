"""Hot-layer sacred similarity & hidden interference analysis.

Only computes on layers where E_in (cross-task mean) exceeds a threshold,
avoiding the dilution effect from cold layers that contribute ~1.0 similarity
and ~0 interference by construction.

Outputs per-layer matrices AND the hot-layer average.
Plots have no title and no colorbar for clean paper figures.

Usage:
    python -m analysis.lbw_subspaces.analyze_hot_layers \
        --model ViT-B-16 --model_location models/ckpts \
        --K 15 --k 8 --top_layers 6
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.variables_and_paths import DATASETS_8  # noqa: E402

from analysis.lbw_subspaces.analyze_pact_motivation import (  # noqa: E402
    _is_2d_weight,
    compute_sacred_basis,
    extract_pretrained_core_basis,
    extract_task_explicit_basis,
    load_pretrained_and_task_states,
)


def _short_layer_name(name: str) -> str:
    import re
    m = re.search(r"resblocks\.(\d+)\.(.+)$", name)
    if m:
        b = int(m.group(1))
        suffix = m.group(2)
        mapping = {
            "attn.in_proj_weight": "attn.in_proj",
            "attn.out_proj.weight": "attn.out_proj",
            "mlp.c_fc.weight": "mlp.c_fc",
            "mlp.c_proj.weight": "mlp.c_proj",
        }
        return f"b{b:02d}.{mapping.get(suffix, suffix)}"
    name = name.replace("model.visual.", "").replace("model.", "")
    return name


# ---------------------------------------------------------------------------
# Core computation: per-layer sim & interference
# ---------------------------------------------------------------------------


def compute_per_layer_sim_interf(
    pretrained_state: Mapping[str, torch.Tensor],
    task_states: Mapping[str, Mapping[str, torch.Tensor]],
    task_order: List[str],
    K: int = 15,
    k: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> Dict[str, object]:
    """Compute per-layer sacred similarity and hidden interference.

    Returns dict with:
        sim_per_layer   : list of (T, T) ndarrays
        interf_per_layer: list of (T, T) ndarrays
        e_in_per_layer  : list of floats (cross-task mean E_in for each layer)
        layer_names     : list of str
    """
    n_tasks = len(task_order)
    sim_per_layer: List[np.ndarray] = []
    interf_per_layer: List[np.ndarray] = []
    e_in_per_layer: List[float] = []
    layer_names: List[str] = []

    for name, W_pre_cpu in pretrained_state.items():
        if not _is_2d_weight(name, W_pre_cpu):
            continue
        if any(name not in task_states[t] for t in task_order):
            continue

        W_pre = W_pre_cpu.to(device=device, dtype=torch.float32)
        V_pre_K = extract_pretrained_core_basis(W_pre, K=K)

        task_bases: List[torch.Tensor] = []
        sacreds: List[torch.Tensor] = []
        e_in_vals: List[float] = []

        for t in task_order:
            W_t = task_states[t][name].to(device=device, dtype=torch.float32)
            Delta_t = W_t - W_pre
            delta_sq = float(torch.linalg.norm(Delta_t, ord="fro") ** 2)
            if delta_sq > 1e-12:
                projected = Delta_t @ V_pre_K
                in_energy = float(torch.linalg.norm(projected, ord="fro") ** 2)
                e_in_vals.append(in_energy / delta_sq)
            else:
                e_in_vals.append(0.0)
            V_t_k = extract_task_explicit_basis(Delta_t, k=k)
            V_sacred_t = compute_sacred_basis(V_pre_K, V_t_k)
            task_bases.append(V_t_k)
            sacreds.append(V_sacred_t)
            del W_t, Delta_t

        sim_mat = np.zeros((n_tasks, n_tasks), dtype=np.float64)
        interf_mat = np.zeros((n_tasks, n_tasks), dtype=np.float64)

        for i in range(n_tasks):
            V_a = sacreds[i]
            r_a = V_a.shape[1]
            for j in range(n_tasks):
                V_b = sacreds[j]
                r_b = V_b.shape[1]
                if r_a == 0 or r_b == 0:
                    sim_mat[i, j] = 0.0
                else:
                    overlap = V_a.t() @ V_b
                    sim_mat[i, j] = float(
                        torch.linalg.norm(overlap, ord="fro") ** 2
                    ) / float(min(r_a, r_b))

                V_B_k = task_bases[i]
                V_sacred_A = sacreds[j]
                k_b = V_B_k.shape[1]
                if V_sacred_A.shape[1] == 0 or k_b == 0:
                    interf_mat[i, j] = 0.0
                else:
                    proj = V_B_k.t() @ V_sacred_A
                    interf_mat[i, j] = float(
                        torch.linalg.norm(proj, ord="fro") ** 2
                    ) / float(k_b)

        sim_per_layer.append(sim_mat)
        interf_per_layer.append(interf_mat)
        e_in_per_layer.append(float(np.mean(e_in_vals)))
        layer_names.append(name)

        if verbose and len(layer_names) % 10 == 0:
            print(f"  processed {len(layer_names)} layers (last: {name})")

        del W_pre, V_pre_K, task_bases, sacreds
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return {
        "sim_per_layer": sim_per_layer,
        "interf_per_layer": interf_per_layer,
        "e_in_per_layer": e_in_per_layer,
        "layer_names": layer_names,
    }


def select_hot_layers(
    e_in_per_layer: List[float], layer_names: List[str], top_n: int = 6
) -> List[int]:
    order = np.argsort(e_in_per_layer)[::-1]
    return order[:top_n].tolist()


# ---------------------------------------------------------------------------
# Visualization (no title, no colorbar)
# ---------------------------------------------------------------------------


def plot_heatmap_clean(
    matrix: np.ndarray,
    task_names: List[str],
    save_path: str,
    vmin: float = 0.0,
    vmax: float | None = None,
    cmap_name: str = "rocket",
    fmt: str = ".2f",
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.linewidth": 0.0,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })

    if vmax is None:
        vmax = float(np.max(matrix))

    cmap = sns.color_palette(cmap_name, as_cmap=True)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    sns.heatmap(
        matrix, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
        annot=True, fmt=fmt, annot_kws={"size": 9, "weight": "bold"},
        xticklabels=task_names, yticklabels=task_names,
        cbar=False, linewidths=0.6, linecolor="white", square=True,
    )
    ax.set_xticklabels(task_names, rotation=45, ha="right", fontsize=10)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_yticklabels(task_names, rotation=0, fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    fig.savefig(os.path.splitext(save_path)[0] + ".pdf")
    plt.close(fig)


def save_matrix_csv(
    matrix: np.ndarray, task_names: List[str], out_path: str
) -> None:
    header = "," + ",".join(task_names)
    rows = [
        task_names[i] + "," + ",".join(f"{v:.6f}" for v in matrix[i])
        for i in range(matrix.shape[0])
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hot-layer sacred similarity & hidden interference"
    )
    parser.add_argument("--model", type=str, default="ViT-B-16")
    parser.add_argument(
        "--model_location", type=str,
        default=str(REPO_ROOT / "models_complete" / "models" / "checkpoints"),
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(Path(__file__).resolve().parent / "ouput_hot_layers"),
    )
    parser.add_argument("--K", type=int, default=15)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--top_layers", type=int, default=6)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=DATASETS_8,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isabs(args.model_location):
        args.model_location = str((REPO_ROOT / args.model_location).resolve())
    if not os.path.isabs(args.output_dir):
        args.output_dir = str(Path(args.output_dir).resolve())
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 72)
    print("Hot-layer Sacred Similarity & Hidden Interference")
    print(f"  Model       : {args.model}")
    print(f"  Tasks ({len(args.tasks)}): {args.tasks}")
    print(f"  K={args.K}, k={args.k}, top_layers={args.top_layers}")
    print(f"  Device      : {args.device}")
    print(f"  Output dir  : {args.output_dir}")
    print("=" * 72)

    pre_state, task_states = load_pretrained_and_task_states(
        args.model, args.model_location, args.tasks
    )

    res = compute_per_layer_sim_interf(
        pretrained_state=pre_state,
        task_states=task_states,
        task_order=args.tasks,
        K=args.K,
        k=args.k,
        device=args.device,
    )

    sim_per_layer = res["sim_per_layer"]
    interf_per_layer = res["interf_per_layer"]
    e_in_per_layer = res["e_in_per_layer"]
    layer_names = res["layer_names"]

    hot_indices = select_hot_layers(
        e_in_per_layer, layer_names, top_n=args.top_layers
    )

    print(f"\n[Hot layers selected (top {args.top_layers} by mean E_in)]")
    for li in hot_indices:
        print(f"  {_short_layer_name(layer_names[li]):<22} "
              f"mean E_in = {e_in_per_layer[li]:.4f}")

    # Compute hot-layer average
    hot_sim = np.mean([sim_per_layer[i] for i in hot_indices], axis=0)
    hot_interf = np.mean([interf_per_layer[i] for i in hot_indices], axis=0)

    # Save average matrices
    save_matrix_csv(hot_sim, args.tasks,
                    os.path.join(args.output_dir, "hot_sacred_similarity_avg.csv"))
    save_matrix_csv(hot_interf, args.tasks,
                    os.path.join(args.output_dir, "hot_hidden_interference_avg.csv"))
    np.save(os.path.join(args.output_dir, "hot_sacred_similarity_avg.npy"), hot_sim)
    np.save(os.path.join(args.output_dir, "hot_hidden_interference_avg.npy"), hot_interf)

    # Save per-layer matrices
    per_layer_dir = os.path.join(args.output_dir, "per_layer")
    os.makedirs(per_layer_dir, exist_ok=True)
    for li in hot_indices:
        short = _short_layer_name(layer_names[li])
        save_matrix_csv(
            sim_per_layer[li], args.tasks,
            os.path.join(per_layer_dir, f"sim_{short}.csv"),
        )
        save_matrix_csv(
            interf_per_layer[li], args.tasks,
            os.path.join(per_layer_dir, f"interf_{short}.csv"),
        )

    # Print summary
    print(f"\n[Hot-layer avg sacred similarity]")
    off_diag = hot_sim[~np.eye(hot_sim.shape[0], dtype=bool)]
    print(f"  off-diag mean: {off_diag.mean():.4f}")
    print(f"  off-diag min : {off_diag.min():.4f}")
    print(f"  off-diag max : {off_diag.max():.4f}")

    print(f"\n[Hot-layer avg hidden interference]")
    off_diag_i = hot_interf[~np.eye(hot_interf.shape[0], dtype=bool)]
    print(f"  off-diag mean: {off_diag_i.mean():.4f}")
    print(f"  off-diag min : {off_diag_i.min():.4f}")
    print(f"  off-diag max : {off_diag_i.max():.4f}")

    # Plot average heatmaps
    plot_heatmap_clean(
        hot_sim, args.tasks,
        os.path.join(args.output_dir, "hot_sacred_similarity_avg.png"),
        vmin=0.0, vmax=1.0,
    )
    plot_heatmap_clean(
        hot_interf, args.tasks,
        os.path.join(args.output_dir, "hot_hidden_interference_avg.png"),
        vmin=0.0, vmax=None,
    )

    # Plot per-layer heatmaps
    for li in hot_indices:
        short = _short_layer_name(layer_names[li])
        plot_heatmap_clean(
            sim_per_layer[li], args.tasks,
            os.path.join(per_layer_dir, f"sim_{short}.png"),
            vmin=0.0, vmax=1.0,
        )
        plot_heatmap_clean(
            interf_per_layer[li], args.tasks,
            os.path.join(per_layer_dir, f"interf_{short}.png"),
            vmin=0.0, vmax=None,
        )

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
