import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from stochastic_superhuman_fairness.core.models.model_io_utils import load_model_from_archive
from stochastic_superhuman_fairness.core.plotting.rollout_plots import plot_rollouts_vs_demos


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
    ap.add_argument("--out_name", default="rollouts_vs_demos.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Load
    model, cfg, demo = load_model_from_archive(
        archive_path=args.archive,
        device=args.device,
        phase_idx=args.phase,
        strict=False,
        use_safe_load=True,
    )

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

    fig, _axes = plot_rollouts_vs_demos(
        rollouts=rollout_feats,
        demos=demo_feats,
        feature_names=feature_names,
        pairs=pairs,
        title=title,
        alpha=1,
        beta=None,
        return_artists=False,
    )

    # Save
    save_dir = args.save_plot_dir or os.path.dirname(os.path.abspath(args.archive))
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, args.out_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved plot:", out_path)


if __name__ == "__main__":
    raise SystemExit(main())
