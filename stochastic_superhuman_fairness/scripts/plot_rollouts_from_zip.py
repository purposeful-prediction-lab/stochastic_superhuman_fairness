import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from stochastic_superhuman_fairness.core.qp_solver import solve_stochastic_subdom_coupling
from stochastic_superhuman_fairness.core.models.model_io_utils import load_model_from_archive
from stochastic_superhuman_fairness.core.plotting.rollout_plots import plot_rollouts_vs_demos, plot_zero_one_vs_features_subplots
from stochastic_superhuman_fairness.core.plotting.aux_plots import plot_subdominance_heatmap, plot_optimal_transport_solution, plot_ot_solution_heatmaps
from stochastic_superhuman_fairness.core.fairness.subdominance import (
    compute_subdominance_matrix,
)

def _parse_pairs(s: str):
    """
    "0,1;0,3;2,4" -> [(0,1),(0,3),(2,4)]
    """
    if s is None:
        return None
    out = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        i, j = chunk.split(",")
        out.append((int(i), int(j)))
    return out


def _choose_demos(demos, n: int, seed: int = 0):
    """
    If n <= len(demos): sample without replacement.
    If n > len(demos): sample with replacement.
    """
    rng = random.Random(seed)
    D = len(demos)
    if n <= D:
        idx = rng.sample(range(D), k=n)
        return [demos[i] for i in idx]
    return [demos[rng.randrange(D)] for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, help="Path to run.zip")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--n_rollouts", type=int, default=None,
                    help="Number of rollouts to collect (default: number of demos in split).")
    ap.add_argument("--dist_mode", default="per_param_diag",
                    help="Bayesian dist mode (if model supports bayesian rollouts).")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Decision threshold for action sampling.")
    ap.add_argument("--pairs", default=None,
                    help='Optional pairs like "0,1;0,3;2,4"')
    ap.add_argument("--save_plot_dir", default=None,
                    help="If set, save plot here; otherwise saves next to archive.")
    ap.add_argument("--feat_vs_feats_name", default="all_features_vs_features.png")
    ap.add_argument("--zero_one_vs_feats_name", default="zero_one_vs_features.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Load
    model, cfg, demo = load_model_from_archive(
        archive_path=args.archive,
        device=args.device,
        phase_idx=args.phase,
        strict=False,
        use_safe_load=True,
        overwrite_demos = False,
    )
    import ipdb;ipdb.set_trace()
    demos_all = demo.train_demos if args.split == "train" else demo.test_demos
    if demos_all is None or len(demos_all) == 0:
        raise RuntimeError(f"No demos found for split={args.split}")

    n_rollouts = args.n_rollouts if args.n_rollouts is not None else len(demos_all)
    demos_sel = _choose_demos(demos_all, n=n_rollouts, seed=args.seed)
    pcfg = cfg.phase_cfg

    # Collect rollouts (bayesian if available, otherwise deterministic)
    if hasattr(model, "collect_bayesian_rollouts"):
        rb = model.collect_bayesian_rollouts(
            demo,
            demos=demos_sel,
            dist_mode= pcfg.train.bayesian.dist_mode,
            decision_threshold=args.threshold,
        )
        rollout_feats = rb.feats.detach().cpu().numpy()
    else:
        # Deterministic fallback: use the base collector if you exposed it, else minimal inline.
        if hasattr(model, "collect_eval_rollouts"):
            rb = model.collect_eval_rollouts(demo, demos=demos_sel, decision_threshold=args.threshold)
            rollout_feats = rb.feats.detach().cpu().numpy()
        else:
            raise RuntimeError("Model has neither collect_bayesian_rollouts nor collect_eval_rollouts.")

    # Demo feats
    demo_feats = np.stack([d["fairness_feats"] for d in demos_all])  # [D, K]

    # Plot
    pairs = _parse_pairs(args.pairs)
    feature_names = getattr(model, "metrics_list", None)
    title = f"{os.path.basename(args.archive)} | split={args.split} | R={n_rollouts} D={len(demos_all)}"

    alpha = model.compute_alpha(rollout_feats, demo.train_demo_means_sorted, mode = model.subdom_mode)
    import ipdb;ipdb.set_trace()
    fig, _axes = plot_rollouts_vs_demos(
        rollouts=rollout_feats,
        demos=demo_feats,
        feature_names=feature_names,
        pairs=pairs,
        title=title,
        alpha=alpha,
        beta=None,
        return_artists=False,
    )

    # Save
    save_dir = args.save_plot_dir or os.path.dirname(os.path.abspath(args.archive))
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, args.feat_vs_feats_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    fig, axes = plot_zero_one_vs_features_subplots(
        rb,
        demos_all,
        feature_names=feature_names,
        alpha=alpha,          # (K,) or (K+1,) if you want y-alpha too
    )
    # Save
    out_path = os.path.join(save_dir, args.zero_one_vs_feats_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved plot:", out_path)

    S = compute_subdominance_matrix(
            rollout_feats,
            demo_feats,
            mode=model.subdom_mode,
            alpha=alpha,
            beta=model.compute_beta(),
        )
    fig, axes = plot_subdominance_heatmap(S)

    # Save
    out_path = os.path.join(save_dir, "Subdominance_heatmap.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    out = solve_stochastic_subdom_coupling(
            S,
            solver="sinkhorn",
            weight_method="primal",
            normalize_subdom=True,
    )
    fig, axes = plot_ot_solution_heatmaps(out)
    # Save
    out_path = os.path.join(save_dir, "ot_solution.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
