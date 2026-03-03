import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from stochastic_superhuman_fairness.core.qp_solver import solve_stochastic_subdom_coupling
from stochastic_superhuman_fairness.core.models.model_io_utils import load_model_from_archive
from stochastic_superhuman_fairness.core.utils_io import load_metrics_jsonl
from stochastic_superhuman_fairness.core.plotting.rollout_plots import  plot_zero_one_vs_features_subplots, plot_rollouts_vs_demos_pairs, plot_zero_one_vs_features
from stochastic_superhuman_fairness.core.plotting.loss_plots import  plot_loss_and_subdom
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
    ap.add_argument("--split", choices=["train", "eval"], default="eval")
    ap.add_argument("--n_rollouts", type=int, default=100, help="Number of rollouts to collect (default: number of demos in split).")
    ap.add_argument("--decision_threshold", type=float, default=0.5, help="Decision threshold for action sampling.")
    ap.add_argument("--pairs", default=None, help='Optional pairs like "0,1;0,3;2,4"')
    ap.add_argument("--stochastic", action="store_true", help="Enable stochastic label sampling for visualization.")
    ap.add_argument("--use_demos_as_gtruth", action="store_true", help="Use Demo labels for metric compuation instead of ground truth.")
    ap.add_argument("--save_plot_dir", default=None, help="If set, save plot here; otherwise saves next to archive.")
    ap.add_argument("--feat_vs_feats_name", default="all_features_vs_features.png")
    ap.add_argument("--zero_one_vs_feats_name", default="zero_one_vs_features.png")
    ap.add_argument("--losses_name", default="losses.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r_opacity", type=float, default=.35)
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
    demos_all = demo.train_demos if args.split == "train" else demo.eval_demos
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
            rb = model.collect_eval_rollouts(demo,
                                             demos=demos_sel, 
                                             decision_threshold = args.decision_threshold,
                                             n_rollouts = args.n_rollouts,
                                             stochastic=args.stochastic,
                                             use_demos_as_gtruth = args.use_demos_as_gtruth,
                                             )
            rollout_feats = rb.feats.detach().cpu().numpy()
        else:
            raise RuntimeError("Model has neither collect_bayesian_rollouts nor collect_eval_rollouts.")

    #  import ipdb;ipdb.set_trace()
    # Demo feats
    demo_feats = np.stack([d["fairness_feats"] for d in demos_all])  # [D, K]

    # Plot
    pairs = _parse_pairs(args.pairs)
    feature_names = getattr(model, "metrics_list", None)
    rtype = 'Stochastic' if args.stochastic else 'Deterministic'
    title = f"{os.path.basename(args.archive)} | split={args.split} | R={n_rollouts}, {rtype} D={len(demos_all)}"

    #  alpha = model.compute_alpha(rollout_feats, demo.train_demo_means_sorted, mode = model.subdom_mode)
    alpha = 1.
    #  import ipdb;ipdb.set_trace()
    # Plot All Feature vs Feature Pairs
    # =====================================================================================
    fig, axes = plot_rollouts_vs_demos_pairs(
        #  rollout_feats,
        rb.feats_by_mode(as_numpy=True),
        demo_feats,
        feature_names=feature_names,
        pairs=pairs,
        title=title,
        baselines = demo.meta['baseline_fairness_features'],
        alpha_rollouts = args.r_opacity,
    )

    # Save
    if args.save_plot_dir is not None:
        save_dir = args.save_plot_dir 
    else:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(args.archive)),'..',  'plots')

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, args.feat_vs_feats_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot Zero One vs All Other Features
    # =====================================================================================
    fig, axes = plot_zero_one_vs_features(
        #  rb,
        rb.feats_by_mode(as_numpy=True),
        demo_feats,
        feature_names=feature_names,
        baselines = demo.meta['baseline_fairness_features'],
        alpha_rollouts = args.r_opacity,
    )

    # Save
    out_path = os.path.join(save_dir, args.zero_one_vs_feats_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved plot:", out_path)
    # Plot Subdominance as heatmap
    # =====================================================================================
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
    # Plot Optimal Transport Coupling
    # =====================================================================================
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

    # Plot Losses
    # =====================================================================================
    log_dir = os.path.join(os.path.dirname(os.path.abspath(args.archive)), '..', 'metrics_log.jsonl')
    logs = load_metrics_jsonl(log_dir)
    fig = plot_loss_and_subdom(logs)
    # Save
    out_path = os.path.join(save_dir, "losses.png")
    print(f'Saving losses.png to {out_path}')
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    raise SystemExit(main())
