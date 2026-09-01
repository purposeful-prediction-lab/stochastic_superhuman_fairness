import argparse
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from stochastic_superhuman_fairness.core.qp_solver import solve_stochastic_subdom_coupling
from stochastic_superhuman_fairness.core.models.model_io_utils import load_model_from_archive
from stochastic_superhuman_fairness.core.utils_io import load_metrics_jsonl, classify_arg, get_base_path, split_path, save_dict_text
from stochastic_superhuman_fairness.core.utils import trajectory_logprob, ensemble_scores
from stochastic_superhuman_fairness.core.plotting.demonstration_plots import(
        plot_label_agreement_stats
        )
from stochastic_superhuman_fairness.core.plotting.rollout_plots import(
        plot_rollouts_vs_demos_pairs, plot_zero_one_vs_features,
        plot_zero_one_vs_features_mode_coupling,
        plot_zero_one_vs_features_demo_logprobs
        )
from stochastic_superhuman_fairness.core.plotting.rollout_movies import animate_zero_one_vs_features, export_html_player
from stochastic_superhuman_fairness.core.plotting.loss_plots import plot_loss_and_subdom
from stochastic_superhuman_fairness.core.plotting.gamma_plots import plot_gamma_diagnostics_dashboard
from stochastic_superhuman_fairness.core.plotting.plot_utils import add_cfg_text_to_figure
from stochastic_superhuman_fairness.core.plotting.plotting_palettes import MODE_PALETTE_100_PAPERSAFE, MODE_PALETTE_100_MAXVAR, generate_paper_safe_mode_palette
from stochastic_superhuman_fairness.core.plotting.aux_plots import (
        plot_paired_subdominance_curve,
        plot_subdominance_heatmap, plot_optimal_transport_solution,
        plot_ot_solution_heatmaps, plot_indicator_matrix,
        plot_dominance_counts,
        plot_cfg_string,
        plot_policy_probs,
        )
from stochastic_superhuman_fairness.core.fairness.subdominance import (
    compute_subdominance_matrix,
    compute_subdominance_matrix_grouped,
)

def compute_intrademo_scale(logs: list) -> float:
    """
    Scale demo baseline loss to match rollout loss magnitude.

    Assumes losses are SUM-reduced over samples.

    Args:
        logs (list)

    Returns:
        scale factor
    """

    for l in logs:
        valid_log = l.get("train/R", None)
        if valid_log is not None:
            num_rollouts = l['train/R']
            num_demos = l['train/D']
            break
    if num_demos == 0:
        #  raise ValueError("num_demos must be > 0")
        return 1.
    return num_rollouts / num_demos

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
    ap.add_argument("--ref_model", type=str, default = None, help="Path to a reference model to be printed in the feature plots")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--split", choices=["train", "eval"], default="train")
    ap.add_argument("--no_blue_palette", action="store_true", help="Exclude blue from plots, as demos are blue to avoid confusion.")
    ap.add_argument("--ot_solver", choices=["mosek", "sinkhorn", None], default=None, help="OT solver. Default is whatever the model has in their config.")
    ap.add_argument("--per_policy_rollouts", type=int, default=10, help="Number of rollouts to collect, per policy.")
    #  ap.add_argument("--n_rollouts", type=int, default=100, help="Number of rollouts to collect (default: number of demos in split).")
    ap.add_argument("--decision_threshold", type=float, default=0.5, help="Decision threshold for action sampling.")
    ap.add_argument("--pairs", default=None, help='Optional pairs like "0,1;0,3;2,4"')
    ap.add_argument("--stochastic", action="store_true", help="Enable stochastic label sampling for visualization.")
    ap.add_argument("--use_demos_as_gtruth", action="store_true", help="Use Demo labels for metric compuation instead of ground truth.")
    ap.add_argument("--save_plot_dir", default=None, help="If set, save plot here; otherwise saves next to archive.")
    ap.add_argument("--feat_vs_feats_name", default="all_features_vs_features.png")
    ap.add_argument("--zero_one_vs_feats_name", default="zero_one_vs_features.png")
    ap.add_argument("--plot_freq", type=int, default=1, help="Plot frequency for all the x vs epoch plots.")
    ap.add_argument("--losses_name", default="losses.png")
    ap.add_argument("--no_anim", action="store_true", help="Skip building the zero_one_vs_features training movie.")
    ap.add_argument("--anim_name", default="zero_one_vs_features_movie.gif")
    ap.add_argument("--anim_fps", type=float, default=2, help="Frames per second for the training movie.")
    ap.add_argument("--anim_match", choices=["gamma", "l2", "none"], default="gamma",
                     help="Draw a line from each policy's aggregate point to its matched demo in the movie, "
                          "matched by highest OT coupling mass (gamma) or nearest L2 distance. 'none' disables it.")
    ap.add_argument("--top3_video", action="store_true",
                     help="In the training movie, draw lines to each policy's top-3 matches (ranked by "
                          "--anim_match) instead of just the best one. Rank 1 is solid, rank 2 dashed and "
                          "half as opaque, rank 3 dotted and fainter still.")
    ap.add_argument("--movie_html", action="store_true",
                     help="Also export the training movie as a self-contained interactive HTML player "
                          "(video-like scrub bar + play/pause + speed dropdown), alongside the gif/mp4. "
                          "No ffmpeg or internet needed -- every frame is embedded inline.")
    ap.add_argument("--anim_html_name", default="zero_one_vs_features_player.html")
    ap.add_argument("--anim_speeds", default="0.25,0.5,1,2,4,8",
                     help="Comma-separated playback-speed multipliers offered in the HTML player's dropdown.")
    ap.add_argument("--no_gamma_matrix", action="store_true",
                     help="Hide the per-policy gamma-matrix panel (mass heatmap + perplexity/dispersion "
                          "per policy) in the training movie. Shown by default.")
    ap.add_argument("--no_demo_indices", action="store_true",
                     help="Hide the per-demo index labels (matching the subdominance/gamma heatmaps' demo "
                          "columns) on the training movie's zero_one_vs_features points. Shown by default "
                          "when there are <= 50 demos.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r_opacity", type=float, default=.35)
    args = ap.parse_args()

    annotation_keywords = [
        "lr", "batch_size", "solver", "train", "stochastic", 
        'init_noise_std', 'normalize_s_matrix', 'n_models', 'gamma_temperature',
        'policy_sources', 'policy_mixture', 'row_constraints',  'col_constraints',
        'ot_temperature', 'gamma_temperature', 't_function',

        ]
    ignore_keywords = ['learner']

    # Load
    model, cfg, demo = load_model_from_archive(
        archive_path=args.archive,
        device=args.device,
        phase_idx=args.phase,
        strict=False,
        use_safe_load=True,
        overwrite_demos = False,
    )
    #  import ipdb;ipdb.set_trace()
    model_name = print(args.archive.rsplit(os.path.sep,1)[-1]) if classify_arg(args.archive)  != 'name' else args._archive
    ot_solver = cfg.learner.default.train.stochastic.solver if args.ot_solver is None else args.ot_solver

    if args.ref_model is not None:
        if classify_arg(args.ref_model)  == 'name':
            ref_path = os.path.join(get_base_path(args.archive), args.ref_model)
            ref_model_name = args.ref_model
        else:
            ref_path = args.ref_model
            _, ref_model_name = split_path(args.ref_model)
        # Load reference model if required
        ref_model, _, _ = load_model_from_archive(
            archive_path=ref_path,
            demonstrator = demo,
            device=args.device,
            phase_idx=args.phase,
            strict=False,
            use_safe_load=True,
            overwrite_demos = False,
        )
    # Choose mode plotting colors
    num_policies = len(model.policies)
    mode_palette = generate_paper_safe_mode_palette(n=100, include_blue= not args.no_blue_palette)
    # Get Demos and Demo Feats
    s_train_demo_feats, s_eval_demo_feats = demo.get_rank_sorted_demo_feats()
    #  import ipdb;ipdb.set_trace()
    if args.split == "train":
        demos_all = demo.train_demos 
        demo_feats = s_train_demo_feats
    else:
        demos_all = demo.eval_demos
        demo_feats = s_eval_demo_feats

    if demos_all is None or len(demos_all) == 0:
        raise RuntimeError(f"No demos found for split={args.split}")
    g_truth_labels = [d['y'] for d in demos_all]
    demo_labels = [d['y_demo'] for d in demos_all]
    per_policy_rollouts = args.per_policy_rollouts if args.per_policy_rollouts is not None else 10
    demos_sel = _choose_demos(demos_all, n=len(demos_all), seed=args.seed)
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
        rbs = []
        if hasattr(model, "collect_eval_rollouts"):
            models = [model]
            if args.ref_model is not None:
                models += [ref_model]
            for m in models:
                rbs.append(m.collect_eval_rollouts(
                        demo,
                        demos=demos_sel, 
                        shared_x = demo.shared_x,
                        decision_threshold = args.decision_threshold,
                        n_rollouts = per_policy_rollouts,
                        stochastic=args.stochastic,
                        use_demos_as_gtruth = args.use_demos_as_gtruth,
                        )
                    )
            rb = rbs[0]
            rollout_feats = rb.feats.detach().cpu().numpy()
            if args.ref_model is not None:
                rb_ref = rbs[1]
                rollout_feats_ref = rb_ref.feats.detach().cpu().numpy()
                # Get rollout groupings
                rollout_groupings_ref = rb_ref.get_rollout_groupings_by_policy()
                feats_by_mode_ref = rb_ref.feats_by_mode(as_numpy=True)
        else:
            raise RuntimeError("Model has neither collect_bayesian_rollouts nor collect_eval_rollouts.")

    # We have made predictions on p * per policy rollouts times for our trajectories.
    # Each per_policy_rollouts traj demarks a new policy. We want the logits of each demo from each policy.
    # TODO: Demos are sorted now. Adddress that
    #  demos_traj_logporobs = trajectory_logprob(rb.logits[::per_policy_rollouts], demo_labels)

    # Get rollout groupings
    rollout_groupings = rb.get_rollout_groupings_by_policy()
    feats_by_mode = rb.feats_by_mode(as_numpy=True)

    # Plot
    pairs = _parse_pairs(args.pairs)
    feature_names = getattr(model, "metrics_list", None)
    rtype = 'Stochastic' if args.stochastic else 'Deterministic'
    title = f"{os.path.basename(args.archive)} | split={args.split} | R/Policy={per_policy_rollouts}, {rtype} D={len(demos_all)}"

    #  alpha = model.compute_alpha(rollout_feats, demo.train_demo_means_sorted, mode = model.subdom_mode)
    alpha = 1.

    # Plot All Feature vs Feature Pairs
    # =====================================================================================
    fig, axes = plot_rollouts_vs_demos_pairs(
        feats_by_mode,
        #  s_eval_demo_feats,
        demo_feats,
        feature_names=feature_names,
        pairs=pairs,
        title=title,
        mode_colors = mode_palette,
        baselines = demo.meta['baseline_fairness_features'],
        alpha_rollouts = args.r_opacity,
        alpha_baselines = 0.6,
    )
    text = add_cfg_text_to_figure(fig, cfg, annotation_keywords, ignore_keywords = ignore_keywords, x=0.01, y=0.5, fontsize=9)
    plt.tight_layout(rect=(0.18, 0.0, 1.0, 1.0))  # leave space on the left
    # Save
    if args.save_plot_dir is not None:
        save_dir = args.save_plot_dir 
    else:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(args.archive)),'..',  'plots')

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, args.feat_vs_feats_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot Subdominance as heatmap
    # =====================================================================================
    S = compute_subdominance_matrix_grouped(
            feats_by_mode,
            demo_feats,
            #  s_eval_demo_feats,
            mode=model.subdom_mode,
            alpha=alpha,
            beta=model.compute_beta(),
        )
    sorted_rollout_groupings = []
    idx_offset = 0
    for m, mode in enumerate(feats_by_mode):
        sorted_rollout_groupings.append([idx_offset+ i for i in range(len(mode))])
        idx_offset += len(mode)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig, _ = plot_subdominance_heatmap(S, ax = axes[0], row_groups = sorted_rollout_groupings, group_colors = mode_palette, group_strip_width = 2.18)
    # Plot OT Coupling
    out = solve_stochastic_subdom_coupling(
                S,
                num_policies,
                solver= ot_solver,
                normalize_s_matrix=False,
        )
    gamma_temp = cfg.learner.get('default').get('train').get('stochastic').get('gamma_temperature', 1.0)
    gamma_temp = 1.0
    fig, _ = plot_ot_solution_heatmaps(out, ax= axes[1], gamma_temperature = gamma_temp, row_groups = sorted_rollout_groupings, 
                                          group_strip_width = 2.18,
                                          group_colors = mode_palette,
                                          )
    # Save
    out_path = os.path.join(save_dir, "subdominance_and_ot_solution.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot Coupling for each mode and demo
    # =====================================================================================
    num_labels = 10 if 10 < len(demo_feats) else len(demo_feats)
    demo_text_labels = np.arange(num_labels).tolist() + [None] * (len(demo_feats)- num_labels)
    mode_colors = mode_palette[:len(feats_by_mode)]

    fig, axes = plot_zero_one_vs_features_mode_coupling(
        rollout_feats=feats_by_mode,         # list of (r_m, K)
        demo_feats=demo_feats,               # (D, K)
        gamma=out['gamma_np'],                         # (sum_r_m, D) or list[(r_m,D)]
        feature_names=feature_names,
        annotate_sidebar = [f"Highest\nBeat Rate", "Lowest"],
        mode_colors = mode_palette,
        demo_text_labels=demo_text_labels,         # e.g. [None, 2, None, 5, ...]
        cmap="viridis",
        coupling_norm="per_mode",
    )
    # Save
    out_path = os.path.join(save_dir, "per_policy_coupling.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved plot:", out_path)

    # Plot Zero One vs All Other Features
    # =====================================================================================
    if args.ref_model is not None:
        #  demo_text_labels = np.arange(20).tolist() + [None] * (len(demo_feats)-20)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        _, _ = plot_zero_one_vs_features(
                feats_by_mode_ref,
                demo_feats,
                ax = axes[1],
                plot_demos_as_text = False,
                demo_text_labels = None,
                feature_names=feature_names,
                title = f"{ref_model_name} Zero-one vs Features",
                #baselines = demo.meta['baseline_fairness_features'],
                mode_colors = mode_palette,
                alpha_rollouts = args.r_opacity,
            )
        main_ax = axes[0]
    else:
        main_ax = None
    #  import ipdb;ipdb.set_trace()
    fig2, _ = plot_zero_one_vs_features(
        feats_by_mode,
        demo_feats,
        ax = main_ax,
        feature_names=feature_names,
        #  baselines = demo.meta['baseline_fairness_features'],
        mode_colors = mode_palette,
        alpha_rollouts = args.r_opacity,
    )
    fig = fig2 if args.ref_model is  None else fig
    # Save
    out_path = os.path.join(save_dir, args.zero_one_vs_feats_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved plot:", out_path)

    
    # Plot  Subdominance Indicator Matrix
    # =====================================================================================
    S_demo_roll = compute_subdominance_matrix_grouped(
            [demo_feats],
            rollout_feats,
            mode=model.subdom_mode,
            alpha=alpha,
            beta=model.compute_beta(),
        )  # [D,R]
    S_rev_ji = S_demo_roll.T  # [R,D]

    reverse_subdom_beat_indicator = S <= S_rev_ji
    beat_indicator = S <= 0.

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig, ax = plot_indicator_matrix(
        reverse_subdom_beat_indicator,
        ax = axes[0],
        row_groups= sorted_rollout_groupings,
        group_colors = mode_palette,
        title="S <= S_rev_ji Indicator Matrix with Row Groups",
        group_strip_width = 2.18,
        )

    fig, ax = plot_indicator_matrix(
        beat_indicator,
        ax = axes[1],
        row_groups=sorted_rollout_groupings,
        group_colors = mode_palette,
        title="Dominance Indicator Matrix with Row Groups",
        group_strip_width = 2.18,
        )
    out_path = os.path.join(save_dir, "indicators.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot Optimal Transport Coupling + Subdominance Heat map
    # =====================================================================================

    #  out = solve_stochastic_subdom_coupling(
    #          S,
    #          solver="sinkhorn",
    #          weight_method="primal",
    #          normalize_subdom=False,
    #  )
    #  gamma_temp = cfg.learner.get('default').get('train').get('stochastic').get('gamma_temperature', 1.0)
    #  gamma_temp = 1.0
    #
    #  fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    #  fig, axes = plot_ot_solution_heatmaps(out, ax= axes, gamma_temperature = gamma_temp, row_groups = sorted_rollout_groupings,
    #                                        group_strip_width = 2.18,
    #                                        group_colors = mode_palette,
    #                                        )
    #  # Save
    #  out_path = os.path.join(save_dir, "ot_solution.png")
    #  fig.savefig(out_path, dpi=200, bbox_inches="tight")
    #  plt.close(fig)
    #
    # =====================================================================================
    # Log Plots
    # =====================================================================================

    # Plot Losses
    # =====================================================================================
    log_dir = os.path.join(os.path.dirname(os.path.abspath(args.archive)), '..', 'metrics_log.jsonl')
    logs = load_metrics_jsonl(log_dir)

    # Plot Zero One vs Features Movie
    # =====================================================================================
    if not args.no_anim:
        anim_out_path = os.path.join(save_dir, args.anim_name)
        try:
            _, _ = animate_zero_one_vs_features(
                logs,
                demo_feats,
                #  demo.train_demo_feats,
                feature_names=feature_names,
                mode_colors=mode_palette[:num_policies],
                alpha_rollouts=args.r_opacity,
                save_path=anim_out_path,
                fps=args.anim_fps,
                show_gamma_matrix=not args.no_gamma_matrix,
                show_demo_indices=not args.no_demo_indices,
                match=None if args.anim_match == "none" else args.anim_match,
                top_k=3 if args.top3_video else 1,
            )
        except ValueError as e:
            print(f"[plot_all] Skipping training movie: {e}")

        if args.movie_html:
            html_out_path = os.path.join(save_dir, args.anim_html_name)
            speeds = [float(s) for s in args.anim_speeds.split(",")]
            try:
                export_html_player(
                    logs,
                    demo_feats,
                    feature_names=feature_names,
                    mode_colors=mode_palette[:num_policies],
                    alpha_rollouts=args.r_opacity,
                    html_path=html_out_path,
                    base_fps=args.anim_fps,
                    speeds=speeds,
                    show_gamma_matrix=not args.no_gamma_matrix,
                    show_demo_indices=not args.no_demo_indices,
                    match=None if args.anim_match == "none" else args.anim_match,
                    top_k=3 if args.top3_video else 1,
                )
            except ValueError as e:
                print(f"[plot_all] Skipping interactive training movie: {e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    scale = compute_intrademo_scale(logs)
    fig, ax = plot_loss_and_subdom(logs,ax=axes[0], intrademo_scale = scale,  log_loss_scale = False,
                                   plot_freq=args.plot_freq)
    #  try:
    fig, ax = plot_paired_subdominance_curve(logs, ax = axes[1])
    #  except:
        #  print('Not paired subdominance')
    # Save
    out_path = os.path.join(save_dir, "losses.png")
    print(f'Saving losses.png to {out_path}')
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    # Plot Demo Log Probs
    # =====================================================================================
    logits = np.array(rb.logits[::per_policy_rollouts])
    demo_logprobs = ensemble_scores(logits, torch.stack(demo_labels), mode = 'prob', temperature  = 10., require_grad = False)
    fig, ax = plot_zero_one_vs_features_demo_logprobs(
                    feats_by_mode,
                    demo_feats,
                    demo_logprobs,
                    #  ax = axes[1],
                    feature_names=feature_names,
                    title = f"{model_name} Demo log Probablity",
                    mode_colors = mode_palette,
                    alpha_rollouts = args.r_opacity,
                    logprob_norm = 'per_mode',
                )
    # Save
    out_path = os.path.join(save_dir, "demo_logprobs.png")
    print(f'Saving demo_logprobs.png to {out_path}')
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot Loss Term Activations
    # =====================================================================================

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig, _ = plot_dominance_counts(
        logs,
        ax = axes[0],
        per_mode="train/l_terms/per_mode_dominant_rollouts",
        title = 'Dominance and Rollout Indicator Counts per epoch',
        x_timesteps= [i for i in range(0, len(logs), 40)],
        palette= mode_palette,
    )
    #  try:
    fig, _ = plot_policy_probs(
        logs,
        ax=axes[1],
        title = 'Dominance and Rollout Indicator Counts per epoch',
        palette= mode_palette,
    )
    #  except:
        #  print("No policy probs found")
    # Save
    out_path = os.path.join(save_dir, "loss_term_activation_counts.png")
    print(f'Saving loss_term_activation_counts.png to {out_path}')
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


    # Plot Configs
    # =====================================================================================
    #  fig, ax = plot_cfg_string(cfg)
    # Save
    out_path = os.path.join(save_dir, "configs.txt")
    save_dict_text(cfg, path = out_path)
    print(f'Saving configs.txt to {out_path}')
    #  fig.savefig(out_path, dpi=200, bbox_inches="tight")
    #  plt.close(fig)

    # Plot Demonstration Info
    # =====================================================================================
    # Save
    # This segment requires a 2x2 grid
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 5))
        out_path = os.path.join(save_dir, "demos_info.png")
        print(f'Saving demonstration info plot to {out_path}')
        fig, _ = plot_label_agreement_stats(demo.label_agreement_dict, axes = (axes[0,0], axes[0,1]))
        fig, _ = plot_label_agreement_stats(demo.label_agreement_dict, axes = (axes[1,0], axes[1,1]), split = 'eval')
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Count not draw demo info plot, error: {e}.\n Skipping...")

    # Plot all gamma diagnostics
    #  try:
    fig, out = plot_gamma_diagnostics_dashboard(logs, palette = mode_palette)
    out_path = os.path.join(save_dir, "gamma_plots.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    #  except:
        #  print(f"\n Failed plotting Gamma diagnostics\n")

if __name__ == "__main__":
    raise SystemExit(main())
