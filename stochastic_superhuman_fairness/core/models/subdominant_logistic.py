# Torch Libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import stack_module_state, functional_call, vmap
from typing import Union
# numpy and computation
import numpy as np
#  Stochastic Superhuman Fairness Libs
from stochastic_superhuman_fairness.core.models.logistic import LogisticRegressionModel
from stochastic_superhuman_fairness.core.qp_solver import solve_stochastic_subdom_coupling, bj_from_beatrates_nocollapse
from stochastic_superhuman_fairness.core.fairness.fairness_metrics import compute_directional_cost
from stochastic_superhuman_fairness.core.fairness.subdominance import (
        subdominant_logloss_shared_X_multi_rollout,  # the [R,N] logits version, reference
        subdominant_weighted_logloss_shared_X_multi_rollout,  # the [R,N] logits version, reference
        compute_subdominance_matrix,
        compute_subdominance_matrix,
        )
from stochastic_superhuman_fairness.core.fairness.fairness_metrics import compute_fairness_features, compute_fairness_features_batched, compute_fairness_features_batched_fast
from stochastic_superhuman_fairness.core.utils import sample_binary_from_probs, sample_rollouts_from_probs
from stochastic_superhuman_fairness.core.rollout_utils import collect_rollouts
from stochastic_superhuman_fairness.core.baselines.logistic_regression import (
        train_logistic_from_demos, load_saved_sklearn_policy_into_torch,  sklearn_logistic_to_torch_clone,
        logistic_training_settings, sklearn_model_to_torch_module,
        )
# Path and system libs
from pathlib import Path
import os
CURRENT_FILE = Path(__file__).resolve()
CURRENT_DIR = CURRENT_FILE.parent
# General Libs
import copy

class MultiSubdominantLogisticRegressionModel(LogisticRegressionModel):
    """
    Maintains an ensemble of n logistic models (or small MLPs).

    Each epoch:
      1) Forward shared X through each model -> logits_i
      2) Make yhat_i decisions per model
      3) Compute fairness features per model (rollout_feats)
      4) Compute S = subdom(rollout_feats, demo_feats)
      5) Solve OT -> gamma
      6) Compute reverse subdom and indicator I_ij = 1[S_ij <= Srev_ji]
      7) Compute custom loss and do a GD step (on all models)
    """

    def __init__(self, cfg, demonstrator):
        super().__init__(cfg, demonstrator)


        # --- ensemble size ---
        self.n_models = int(self.model_cfg.get("n_models", 8))

        # --- create ensemble ---
        # Reuse the base LogisticRegressionModel policy architecture by cloning it.
        source_policies = self._build_source_policies_from_cfg(demonstrator)
        mixtures = self.model_cfg.get("policy_mixture", None)
        if mixtures is None:
            raise ValueError("model_cfg.policy_mixture must be provided.")

        self.policies = self.init_policy_mixture(
            source_policies=source_policies,
            mixtures=mixtures,
            n_total_policies=self.n_models,
        )
        #  import ipdb;ipdb.set_trace()
        # Assumes self.policy is an nn.Module created in LogisticRegressionModel.__init__.
        #  log_policy = train_logistic_from_demos(demonstrator.train_demos)
        #  import ipdb;ipdb.set_trace()
        #  logistic_policy = load_saved_sklearn_policy_into_torch(
        #      os.path.join(CURRENT_DIR, "../../data/checkpoints/logistic_model_adult_performance_0p1884_0p1289_0p0559_0p0903_0p1419.pkl"),
        #      device=self.device,
        #  )
        #  self.policies = self.init_policy_mixture([self.policy, logistic_policy], [1, 1], 2)
        #  self.policies = self.init_policy_mixture([self.policy], [8], 8)
        #  self.policies = self.init_policy_mixture([logistic_policy], [3], 3)
        #  self.policies = self.init_policy_mixture([logistic_policy], [100], 100)
        #  import ipdb;ipdb.set_trace()
        #  self.policies = nn.ModuleList([self._clone_policy() for _ in range(self.n_models)])

        # optional: initialize ensemble members slightly differently
        init_noise = float(self.model_cfg.get("init_noise_std", 0.05))
        if init_noise > 0:
            self._perturb_ensemble(init_noise)
        # optimizer (can be replaced externally)
        lr = float(self.cfg.get("train", {}).get("lr", 1e-3))
        wd = float(self.cfg.get("train", {}).get("weight_decay", 0.0))
        if self.cfg.get("train", {}).get('optimizer', 'sgd') == 'sgd':
            self.optimizer = torch.optim.SGD(self.policies.parameters(), lr=lr, weight_decay=wd)
        else:
            self.optimizer = torch.optim.Adam(self.policies.parameters(), lr=lr, weight_decay=wd)
        #  import ipdb;ipdb.set_trace()

    def _perturb_ensemble(self, std: float):
        with torch.no_grad():
            for m in self.policies:
                for p in m.parameters():
                    p.add_(std * torch.randn_like(p))

    @torch.no_grad()
    def _decisions_from_logits(self, logits: torch.Tensor, threshold: float | None, stochastic_if_threshold: bool):
        probs = torch.sigmoid(logits)
        if threshold is None:
            # sample Bernoulli
            yhat = torch.bernoulli(probs)
        else:
            if stochastic_if_threshold:
                yhat = torch.bernoulli(probs)
            else:
                yhat = (probs > threshold).float()
        return probs, yhat
    # ------------------------------------------------------------
    # Full vectorized training epoch block
    # ------------------------------------------------------------
    def _train_one_epoch_vec(
        self,
        demonstrator,
        batch_size: int = 1,
        n_dir: int = 20,
        n_rollouts: int = None,
        decision_threshold: float | None = None,
        solver: str = "mosek",
        row_constraints: bool = False,
        col_constraints: str = None, # None, rev_ranking
        ot_temperature: float = 1.0,
        gamma_temperature: float = 1.0,
        normalize_s_matrix: bool = True,
        stochastic_if_threshold: bool = False,
        alpha_updates: str = "analytical",
        no_update: bool = False,
        use_demos_as_gtruth: bool = False,
        loss_fn: str = None,
        loss_fn_kwargs: dict = {},
        **kwargs,
    ):
        '''This function assumes shared x among demos.'''
        demos = demonstrator.train_demos
        device = next(self.policies.parameters()).device

        # ---- shared data ----
        X = demos[0]["X"].to(device)
        A = demos[0]["A"]
        A = A.to(device) if torch.is_tensor(A) else torch.as_tensor(A, device=device)
        y_true = demos[0]["y"].to(device).view(-1).float()

        y_demo = torch.stack(
            [demonstrator.get_targets(d).to(device).view(-1).float() for d in demos],
            dim=0
        )  # [D, N]

        policies = self.policies
        P = len(policies)
        M = int(n_rollouts) if n_rollouts is not None else 1
        D = len(demos)
        R = P * M

        metric_weights = torch.ones(len(self.metrics_list), device=device, dtype=torch.float32)

        logits_per_policy = []
        yhat_per_policy = []
        feat_per_policy = []

        for pol in policies:
            logits = pol(X).squeeze(-1)                      # [N]
            probs = torch.sigmoid(logits)                    # [N]

            if (decision_threshold is not None) and (not stochastic_if_threshold):
                yhat_batch = (probs.unsqueeze(0).expand(M, -1) >= decision_threshold).float()   # [M, N]
            else:
                probs_batch = probs.unsqueeze(0).expand(M, -1)                                   # [M, N]
                yhat_batch = torch.bernoulli(probs_batch)                                         # [M, N]
            #  import ipdb;ipdb.set_trace()
            feat_batch = compute_fairness_features_batched_fast(
                y_true=y_true,
                y_pred_batch=yhat_batch,   # [M, N]
                a=A,
                metrics=self.metrics_list,
                weights=metric_weights,
            )  # [M, K]

            logits_per_policy.append(logits)        # each [N]
            yhat_per_policy.append(yhat_batch)      # each [M, N]
            feat_per_policy.append(feat_batch)      # each [M, K]

        logits_policies = torch.stack(logits_per_policy, dim=0)   # [P, N]
        yhat_all = torch.stack(yhat_per_policy, dim=0)            # [P, M, N]
        feat_all = torch.stack(feat_per_policy, dim=0)            # [P, M, K]

        N = logits_policies.shape[-1]
        K = feat_all.shape[-1]

        logits_rollouts = (
            logits_policies.unsqueeze(1)
            .expand(-1, M, -1)
            .reshape(R, N)
        )  # [R, N]

        yhat_rollouts = yhat_all.reshape(R, N)     # [R, N]
        rollout_feats = feat_all.reshape(R, K)     # [R, K]
        policy_ids = torch.arange(P, device=device).repeat_interleave(M)  # [R]

        demo_feats = np.stack([d["fairness_feats"] for d in demos])  # [D, K]
        # ----------------------------------------------------
        # 2) Subdominance matrix S[R,D]
        # ----------------------------------------------------
        if alpha_updates == 'analytical':
            self.compute_alpha(
                rollout_feats,
                demonstrator.train_demo_means_sorted,
                mode=self.subdom_mode,
                reduce='mean'
            )

        S = compute_subdominance_matrix(
            rollout_feats,
            demo_feats,
            mode=self.subdom_mode,
            alpha=self.alpha if self.alpha is not None else 1.0,
            beta=self.beta,
        )  # [R,D]

        # ----------------------------------------------------
        # 3) Directional cost
        # ----------------------------------------------------
        dir_cost = compute_directional_cost(
            rollout_feats.detach().cpu().numpy(),
            demo_feats,
            n_dir=n_dir,
        )

        # ----------------------------------------------------
        # 4) OT solve -> gamma
        # ----------------------------------------------------
        if col_constraints == 'rev_ranking':
            bj_priors = bj_from_beatrates_nocollapse(demonstrator.beat_rates_train)
        else:
            bj_priors  = None
        #  import ipdb;ipdb.set_trace()
        out = solve_stochastic_subdom_coupling(
            S,
            solver=solver,
            weight_method="primal",
            normalize_subdom=normalize_s_matrix,
            demo_marginals=bj_priors,
            row_constraints=row_constraints,
            tau=ot_temperature,
        )

        gamma = gamma_temperature * torch.tensor(
            out["gamma_np"], device=device, dtype=torch.float32
        )  # [R,D]

        S_demo_roll = compute_subdominance_matrix(
            demo_feats,
            rollout_feats.detach(),
            mode=self.subdom_mode,
            alpha=self.alpha if self.alpha is not None else 1.0,
            beta=self.beta,
        )  # [D,R]

        S_rev_ji = self.apply_S_temperature(S_demo_roll.T, beta=ot_temperature)  # [R,D]

        win_indicator = (
            torch.as_tensor(S, device=device).float()
            <= torch.as_tensor(S_rev_ji, device=device).float()
        ).float()
        #  S0_indicator = S == 0
        #  S0_indicator = S <= 0.1
        #  indicator = torch.logical_and(S0_indicator, win_indicator).float()
        #  indicator = (S == 0.0).float()
        indicator = (S <= S.mean()).float()
        #  indicator = (S <= 0.2).float()
        #  import ipdb;ipdb.set_trace()
        indicator_rev = (
            torch.as_tensor(S_rev_ji, device=device).float()
            <= torch.as_tensor(S, device=device).float()
        ).float()
        # ----------------------------------------------------
        # 5) Loss + GD step
        # ----------------------------------------------------
        if loss_fn == 'weighted':
            loss_out = subdominant_weighted_logloss_shared_X_multi_rollout(
                    S,
                    torch.tensor(S_rev_ji).to(S.device),
                    logits_rollouts,  # [R,N]
                    yhat_rollouts,    # [R,N]
                    y_demo,           # [D,N]
                    gamma,            # [R,D]
                    indicator,        # [R,D]
                    **loss_fn_kwargs,
                )
        else:
            loss_out = subdominant_logloss_shared_X_multi_rollout(
                logits_rollouts=logits_rollouts,  # [R,N]
                yhat_rollouts=yhat_rollouts,      # [R,N]
                y_demo=y_demo,                    # [D,N]
                gamma=gamma,                      # [R,D]
                indicator_win=indicator,          # [R,D]
            )
        loss_term_dict = {'dominant_rollouts': int(indicator.sum().item()), 'dominant_demos': int(indicator_rev.sum()),
                          'per_mode_dominant_rollouts': indicator.sum(axis=1).reshape(P, n_rollouts).tolist(),
                          'per_mode_dominant_demos': indicator_rev.sum(axis=1).reshape(P, n_rollouts).tolist(),
                          'S_mean': loss_out.info['S_mean'],'norm_paired_subdom': loss_out.info['norm_paired_subdom']
                          }

        if not no_update:
            self.optimizer.zero_grad(set_to_none=True)
            loss_out.loss.backward()
            self.optimizer.step()

        # optional per-policy stats
        z1 = (yhat_rollouts != y_true.unsqueeze(0)).float().mean(dim=1)  # [R]
        per_policy = {}
        for p_idx in range(P):
            mask = (policy_ids == p_idx)
            per_policy[p_idx] = {
                "zero_one": float(z1[mask].mean().detach().cpu()) if mask.any() else np.nan,
                "fairness": rollout_feats[mask].mean(dim=0).detach().cpu().tolist() if mask.any() else [],
                "n_rollouts": int(mask.sum().item()),
            }
        return {
                    "train/loss": float(loss_out.loss.detach().cpu()),
                    "train/mean_subdom": float(S.mean()),
                    "train/std_subdom": float(S.std()),
                    "train/fairness": rollout_feats.mean(dim=0).detach().cpu().tolist(),
                    "train/directional_cost": dir_cost,
                    "train/ot_temperature": float(ot_temperature),
                    "train/indicator_mean": float(indicator.mean().detach().cpu()),
                    "train/R": int(R),
                    "train/P": int(P),
                    "train/D": int(D),
                    "train/l_terms": loss_term_dict,
                    "gamma_matrix": gamma,
                }


    #--------------------------------------------------------------------------------------------------------
    def _train_one_epoch_ref(
        self,
        demonstrator,
        batch_size: int = 1,
        n_dir: int = 20,
        n_rollouts: int = None,              # now: number of rollouts per policy
        decision_threshold: float | None = None,
        solver: str = "mosek",
        row_constraints: bool = False,
        col_constraints: bool = True,
        ot_temperature: float = 1.0,
        gamma_temperature: float = 1.0,
        normalize_s_matrix: bool = True,
        stochastic_if_threshold: bool = False,
        alpha_updates: str = 'analytical',
        no_update: bool = False,
        use_demos_as_gtruth: bool = False,
        **kwargs,
    ):
        demos = demonstrator.train_demos
        device = next(self.policies.parameters()).device

        # ---- shared data ----
        X = demos[0]["X"].to(device)
        A = demos[0]["A"]
        A = A.to(device) if torch.is_tensor(A) else A
        y_true = demos[0]["y"].to(device).view(-1).float()

        # demo labelings over shared X
        y_demo = torch.stack(
            [demonstrator.get_targets(d).to(device).view(-1).float() for d in demos],
            dim=0
        )  # [D,N]

        policies = self.policies
        P = len(policies)
        n_rollouts_per_policy = int(n_rollouts) if n_rollouts is not None else 1
        D = len(demos)

        # ----------------------------------------------------
        # 1) Forward each policy once, sample n_rollouts per policy
        # ----------------------------------------------------
        logits_list = []
        yhat_list = []
        feat_list = []
        policy_ids = []

        for p_idx, pol in enumerate(policies):
            logits = pol(X).squeeze(-1)      # [N]
            probs = torch.sigmoid(logits)    # [N]

            for _ in range(n_rollouts_per_policy):
                yhat = sample_binary_from_probs(probs)  # [N]

                logits_list.append(logits)   # same logits reused
                yhat_list.append(yhat)
                policy_ids.append(p_idx)

                f = compute_fairness_features(
                    y_true,
                    yhat,
                    A,
                    self.metrics_list,
                    weights=[1, 1, 1, 1, 1],
                )  # [K]
                feat_list.append(f)

        logits_rollouts = torch.stack(logits_list, dim=0)   # [R,N]
        yhat_rollouts   = torch.stack(yhat_list, dim=0)     # [R,N]
        rollout_feats   = torch.stack(feat_list, dim=0)     # [R,K]
        policy_ids      = torch.tensor(policy_ids, device=device)  # [R]
        R = logits_rollouts.shape[0]

        demo_feats = np.stack([d["fairness_feats"] for d in demos])  # [D,K]

        # ----------------------------------------------------
        # 2) Subdominance matrix S[R,D]
        # ----------------------------------------------------
        if alpha_updates == 'analytical':
            self.compute_alpha(
                rollout_feats,
                demonstrator.train_demo_means_sorted,
                mode=self.subdom_mode,
                reduce='mean'
            )

        S = compute_subdominance_matrix(
            rollout_feats,
            demo_feats,
            mode=self.subdom_mode,
            alpha=self.alpha if self.alpha is not None else 1.0,
            beta=self.beta,
        )  # [R,D]

        # ----------------------------------------------------
        # 3) Directional cost
        # ----------------------------------------------------
        dir_cost = compute_directional_cost(
            rollout_feats.detach().cpu().numpy(),
            demo_feats,
            n_dir=n_dir,
        )

        # ----------------------------------------------------
        # 4) OT solve -> gamma
        # ----------------------------------------------------
        if col_constraints == 'rev_ranking':
            bj_priors = bj_from_beatrates_nocollapse(demonstrator.beat_rates_train)
        else:
            bj_priors  = None
        out = solve_stochastic_subdom_coupling(
            S,
            solver=solver,
            weight_method="primal",
            normalize_subdom=normalize_s_matrix,
            demo_marginals=bj_priors,
            row_constraints=row_constraints,
            tau=ot_temperature,
        )

        gamma = gamma_temperature * torch.tensor(
            out["gamma_np"], device=device, dtype=torch.float32
        )  # [R,D]

        S_demo_roll = compute_subdominance_matrix(
            demo_feats,
            rollout_feats.detach(),
            mode=self.subdom_mode,
            alpha=self.alpha if self.alpha is not None else 1.0,
            beta=self.beta,
        )  # [D,R]

        S_rev_ji = self.apply_S_temperature(S_demo_roll.T, beta=ot_temperature)  # [R,D]

        indicator = (
            torch.as_tensor(S, device=device).float()
            <= torch.as_tensor(S_rev_ji, device=device).float()
        ).float()

        indicator_rev = (
            torch.as_tensor(S_rev_ji, device=device).float()
            <= torch.as_tensor(S, device=device).float()
        ).float()
        loss_term_dict = {'dominant_rollouts': int(indicator.sum().item()), 'dominant_demos': int(indicator_rev.sum())}
        #  import ipdb;ipdb.set_trace()
        # ----------------------------------------------------
        # 5) Loss + GD step
        # ----------------------------------------------------
        loss_out = subdominant_logloss_shared_X_multi_rollout(
            logits_rollouts=logits_rollouts,  # [R,N]
            yhat_rollouts=yhat_rollouts,      # [R,N]
            y_demo=y_demo,                    # [D,N]
            gamma=gamma,                      # [R,D]
            indicator_win=indicator,          # [R,D]
        )

        if not no_update:
            self.optimizer.zero_grad(set_to_none=True)
            loss_out.loss.backward()
            self.optimizer.step()

        # optional per-policy stats
        z1 = (yhat_rollouts != y_true.unsqueeze(0)).float().mean(dim=1)  # [R]
        per_policy = {}
        for p_idx in range(P):
            mask = (policy_ids == p_idx)
            per_policy[p_idx] = {
                "zero_one": float(z1[mask].mean().detach().cpu()) if mask.any() else np.nan,
                "fairness": rollout_feats[mask].mean(dim=0).detach().cpu().tolist() if mask.any() else [],
                "n_rollouts": int(mask.sum().item()),
            }
        return {
                    "train/loss": float(loss_out.loss.detach().cpu()),
                    "train/mean_subdom": float(S.mean()),
                    "train/std_subdom": float(S.std()),
                    "train/fairness": rollout_feats.mean(dim=0).detach().cpu().tolist(),
                    "train/directional_cost": dir_cost,
                    "train/ot_temperature": float(ot_temperature),
                    "train/indicator_mean": float(indicator.mean().detach().cpu()),
                    "train/R": int(R),
                    "train/P": int(P),
                    "train/D": int(D),
                    "train/l_terms": loss_term_dict,
                }

        #----------------------------------------------------------------------------------------------------------
        #  return {
        #      "train/loss": float(loss_out.loss.detach().cpu()),
        #      "train/mean_subdom": float(S.mean()),
        #      "train/std_subdom": float(S.std()),
        #      "train/fairness": rollout_feats.mean(dim=0).detach().cpu().tolist(),
        #      "train/directional_cost": dir_cost,
        #      "train/ot_temperature": float(ot_temperature),
        #      "train/indicator_mean": float(indicator.mean().detach().cpu()),
        #      "train/R": int(R),
        #      "train/P": int(P),
        #      "train/D": int(D),
        #      "train/n_rollouts_per_policy": int(n_rollouts_per_policy),
        #      "per_policy": per_policy,
        #  }

    def _train_one_epoch_collect(
        self,
        demonstrator,
        batch_size: int = 1,
        n_dir: int = 20,
        n_rollouts: int = None,              # interpreted here as rollouts per policy
        decision_threshold: float | None = None,
        solver: str = "mosek",
        row_constraints: bool = False,
        col_constraints: bool = True,
        ot_temperature: float = 1.0,
        gamma_temperature: float = 1.0,
        normalize_s_matrix: bool = True,
        stochastic_if_threshold: bool = False,
        alpha_updates: str = "analytical",
        no_update: bool = False,
        use_demos_as_gtruth: bool = False,
        **kwargs,
    ):
        demos = demonstrator.train_demos
        device = next(self.policies.parameters()).device
        shared_x = demonstrator.shared_x

        # shared-X assumption
        X = demos[0]["X"].to(device)
        A = demos[0]["A"]
        A = A.to(device) if torch.is_tensor(A) else A
        y_true = demos[0]["y"].to(device).view(-1).float()

        # demo labelings over shared X
        y_demo = torch.stack(
            [demonstrator.get_targets(d).to(device).view(-1).float() for d in demos],
            dim=0,
        )  # [D, N]

        P = len(self.policies)
        n_rollouts_per_policy = int(n_rollouts) if n_rollouts is not None else 1
        R = P * n_rollouts_per_policy
        D = len(demos)

        # cycle demos so we collect exactly R rollouts, one policy chosen internally per rollout
        #  demo_idxs = sample_demo_idxs(len(demos), n_rollouts=R, mode="cycle")
        demo_idxs = None # This will cycle through all demos
        #  import ipdb;ipdb.set_trace()
        rb = collect_rollouts(
            policy=self.policies,
            demonstrator=demonstrator,
            demos=demos,
            shared_x = shared_x,
            metrics_list=self.metrics_list,
            n_rollouts= n_rollouts_per_policy,
            demo_idxs=demo_idxs,
            decision_threshold=decision_threshold,
            require_grad=True,
            detach_outputs=False,
            stochastic=True,
            return_logits=True,
            use_demos_as_gtruth=use_demos_as_gtruth,
        )

        logits_rollouts = torch.stack(rb.logits, dim=0)   # [R, N]
        yhat_rollouts   = torch.stack(rb.y_hat, dim=0)    # [R, N]
        rollout_feats   = rb.feats                        # [R, K]

        demo_feats = np.stack([d["fairness_feats"] for d in demos])  # [D, K]

        if alpha_updates == "analytical":
            self.compute_alpha(
                rollout_feats,
                demonstrator.train_demo_means_sorted,
                mode=self.subdom_mode,
                reduce="mean",
            )

        S = compute_subdominance_matrix(
            rollout_feats,
            demo_feats,
            mode=self.subdom_mode,
            alpha=self.alpha if self.alpha is not None else 1.0,
            beta=self.beta,
        )  # [R, D]

        dir_cost = compute_directional_cost(
            rollout_feats.detach().cpu().numpy(),
            demo_feats,
            n_dir=n_dir,
        )
        if col_constraints == 'rev_ranking':
            bj_priors = bj_from_beatrates_nocollapse(demonstrator.beat_rates_train)
        else:
            bj_priors  = None
        out = solve_stochastic_subdom_coupling(
            S,
            solver=solver,
            weight_method="primal",
            normalize_subdom=normalize_s_matrix,
            demo_marginals=bj_priors,
            row_constraints=row_constraints,
            tau=ot_temperature,
        )

        gamma = gamma_temperature * torch.tensor(
            out["gamma_np"], device=device, dtype=torch.float32
        )  # [R, D]

        S_demo_roll = compute_subdominance_matrix(
            demo_feats,
            rollout_feats.detach(),
            mode=self.subdom_mode,
            alpha=self.alpha if self.alpha is not None else 1.0,
            beta=self.beta,
        )  # [D, R]

        S_rev_ji = self.apply_S_temperature(S_demo_roll.T, beta=ot_temperature)  # [R, D]

        S_t = torch.as_tensor(S, device=device).float()
        S_rev_t = torch.as_tensor(S_rev_ji, device=device).float()

        indicator = (S_t <= S_rev_t).float()
        indicator_rev = (S_rev_t <= S_t).float()

        loss_out = subdominant_logloss_shared_X_multi_rollout(
            logits_rollouts=logits_rollouts,  # [R, N]
            yhat_rollouts=yhat_rollouts,      # [R, N]
            y_demo=y_demo,                    # [D, N]
            gamma=gamma,                      # [R, D]
            indicator_win=indicator,          # [R, D]
        )

        if not no_update:
            self.optimizer.zero_grad(set_to_none=True)
            loss_out.loss.backward()
            self.optimizer.step()

        return {
            "train/loss": float(loss_out.loss.detach().cpu()),
            "train/mean_subdom": float(S.mean()),
            "train/std_subdom": float(S.std()),
            "train/fairness": rollout_feats.mean(dim=0).detach().cpu().tolist(),
            "train/directional_cost": dir_cost,
            "train/ot_temperature": float(ot_temperature),
            "train/indicator_mean": float(indicator.mean().detach().cpu()),
            "train/R": int(R),
            "train/P": int(P),
            "train/D": int(D),
            #  "train/n_rollouts_per_policy": int(n_rollouts_per_policy),
            "train/batch_zero_one": float(rb.batch_zero_one_loss),
            #  "per_policy": rb.aux_list[-1]["policy_perf"] if rb.aux_list else {},
        }

    # -----------------------------------------------------------------------------------------------------------

    def _train_one_epoch_old(
        self,
        demonstrator,
        # Train specific
        batch_size: int = 1,
        n_dir: int = 20,
        n_rollouts: int = None,              # if provided, you can use only first n models
        decision_threshold: float | None = None,  # (ignored here; we sample stochastically)
        # OT specific
        solver: str = "mosek",
        row_constraints: bool = False,
        col_constraints: bool = True,
        ot_temperature: float = 1.0,
        gamma_temperature: float = 1.0,
        normalize_s_matrix: bool = True,
        stochastic_if_threshold: bool = False,    # (ignored here; we sample stochastically)
        alpha_updates: str = 'analytical',   # analytical, None
        # Grad update specific
        no_update: bool = False,
        use_demos_as_gtruth: bool = False,
        **kwargs,
    ):
        demos = demonstrator.train_demos
        device = next(self.policies.parameters()).device

        # ---- shared data ----
        X = demos[0]["X"].to(device)
        A = demos[0]["A"]
        A = A.to(device) if torch.is_tensor(A) else A
        y_true = demos[0]["y"].to(device).view(-1).float()  # ground truth (shared)

        # demo labelings over X (distinct per demo)
        y_demo = torch.stack(
            [demonstrator.get_targets(d).to(device).view(-1).float() for d in demos],
            dim=0
        )  # [D,N]

        # ---- choose how many rollout-modelsrollouts to use ----
        policies = self.policies
        if n_rollouts is not None:
            policies = policies[: int(n_rollouts)]
        R = len(policies)
        D = len(demos)

        # ----------------------------------------------------
        # 1) Forward each model on shared X -> logits_i -> probs -> sample yhat_i
        # ----------------------------------------------------
        logits_list, demo_logits_list = [], []
        yhat_list = []
        feat_list = []
        #  import ipdb;ipdb.set_trace()
        for pol in policies:
            logits = pol(X).squeeze(-1)          # [N]
            probs = torch.sigmoid(logits)         # [N]
            yhat = sample_binary_from_probs(probs)  # [N] sampled decisions

            logits_list.append(logits)
            #  demo_logits_list.append(logits)
            yhat_list.append(yhat)
            # fairness feats use ground truth
            f = compute_fairness_features(y_true, yhat, A, self.metrics_list, weights = [1,1,1,1,1])  # [K]
            feat_list.append(f)

        logits_rollouts = torch.stack(logits_list, dim=0)    # [R,N]
        #  logits_demos    = torch.stack(demo_logits_list, dim=0)    # [R,N]
        yhat_rollouts   = torch.stack(yhat_list, dim=0)      # [R,N]
        rollout_feats   = torch.stack(feat_list, dim=0)      # [R,K]
        #  import ipdb;ipdb.set_trace()

        # demo fairness matrix
        demo_feats = np.stack([d["fairness_feats"] for d in demos])  # [D,K]
        # ----------------------------------------------------
        # 2) Subdominance matrix S[R,D]
        # ----------------------------------------------------
        if alpha_updates == 'analytical':
            self.compute_alpha(rollout_feats, demonstrator.train_demo_means_sorted, mode = self.subdom_mode, reduce = 'mean')
        S = compute_subdominance_matrix(
            rollout_feats,
            demo_feats,
            mode=self.subdom_mode,
            alpha=self.alpha if self.alpha is not None else 1.0,
            beta=self.beta,
        )

        #  import ipdb;ipdb.set_trace()
        # ----------------------------------------------------
        # 3) Directional cost (diagnostic)
        # ----------------------------------------------------
        dir_cost = compute_directional_cost(
            rollout_feats.detach().cpu().numpy(),
            demo_feats,
            n_dir=n_dir,
        )

        # ----------------------------------------------------
        # 4) OT solve -> gamma
        # ----------------------------------------------------
        if col_constraints == 'rev_ranking':
            bj_priors = bj_from_beatrates_nocollapse(demonstrator.beat_rates_train)
        else:
            bj_priors  = None
        out = solve_stochastic_subdom_coupling(
            S,
            solver=solver,
            weight_method="primal",
            normalize_subdom=normalize_s_matrix,
            demo_marginals = bj_priors,
            row_constraints=row_constraints,
            tau = ot_temperature, # Scales subdom matrix before OT computation
        )

        #  import ipdb;ipdb.set_trace()
        gamma = gamma_temperature * torch.tensor(out["gamma_np"], device=device, dtype=torch.float32)  # [R,D]
        #  weights = torch.tensor(gamma.sum(axis=1), device=device, dtype=torch.float32)  # [R]

        # reverse subdom (demo -> rollout), aligned to [R,D]
        S_demo_roll = compute_subdominance_matrix(
            demo_feats,
            rollout_feats.detach(),
            mode=self.subdom_mode,
            alpha=self.alpha if self.alpha is not None else 1.0,
            beta=self.beta,
        )  # [D,R]
        S_rev_ji = S_demo_roll.T  # [R,D]
        S_rev_ji = self.apply_S_temperature(S_rev_ji, beta=ot_temperature)

        indicator = (torch.as_tensor(S, device=device).float() <= torch.as_tensor(S_rev_ji, device=device).float()).float()
        indicator_rev = (torch.as_tensor(S_rev_ji, device=device).float() <= torch.as_tensor(S, device=device).float()).float()

        loss_term_dict = {'dominant_rollouts': int(indicator.sum().item()), 'dominant_demos': int(indicator_rev.sum())}
        #  import ipdb;ipdb.set_trace()
        # ----------------------------------------------------
        # 5) Loss + GD step
        # ----------------------------------------------------
        # THis works when X is shared.
        loss_out = subdominant_logloss_shared_X_multi_rollout(
            logits_rollouts=logits_rollouts,  # [R,N]
            yhat_rollouts=yhat_rollouts,      # [R,N]
            y_demo=y_demo,                    # [D,N]
            gamma=gamma,                      # [R,D]
            indicator_win=indicator,          # [R,D]
        )

        if not no_update:
            self.optimizer.zero_grad(set_to_none=True)
            loss_out.loss.backward()
            self.optimizer.step()

        #  import ipdb;ipdb.set_trace()
        return {
            "train/loss": float(loss_out.loss.detach().cpu()),
            "train/mean_subdom": float(S.mean()),
            "train/std_subdom": float(S.std()),
            "train/fairness": rollout_feats.mean(dim=0).detach().cpu().tolist(),
            "train/directional_cost": dir_cost,
            "train/ot_temperature": float(ot_temperature),
            "train/l_terms": loss_term_dict,
            "train/indicator_mean": float(indicator.mean().detach().cpu()),
            "train/R": int(R),
            "train/D": int(D),
        }

    #--------------------------------------------------------------------------------------------------------

    def train_one_epoch_subdominant_logistic_genericX(
        self,
        demonstrator,
        batch_size: int = 1,
        n_dir: int = 20,
        n_rollouts: int = None,
        decision_threshold: float | None = None,  # ignored; we sample stochastically
        stochastic_if_threshold: bool = False,
        normalize_s_matrix: bool = True,
        solver: str = "mosek",
        row_constraints: bool = False,
        col_constraints: bool = True,
        ot_temperature: float = 1.0,
        no_update: bool = False,
        **kwargs,
    ):
        demos = demonstrator.train_demos
        device = next(self.policies.parameters()).device

        policies = self.policies
        if n_rollouts is not None:
            policies = policies[: int(n_rollouts)]
        R = len(policies)
        D = len(demos)

        # -----------------------------------------
        # 1) Per-demo forward for each model
        #   - store logits_i(X_j) and yhat_i(X_j)
        #   - compute fairness feats using GT y_j
        # -----------------------------------------
        logits_by_demo = []      # list length D, each [R, N_j]
        yhat_by_demo = []        # list length D, each [R, N_j]
        y_demo_list = []         # list length D, each [N_j]
        demo_feats = []          # [D,K] (from stored fairness_feats)
        rollout_feats_accum = [] # list length D, each [R,K]

        for d in demos:
            Xj = d["X"].to(device)
            Aj = d["A"]
            Aj = Aj.to(device) if torch.is_tensor(Aj) else Aj
            y_true_j = d["y"].to(device).view(-1).float()               # GT for this demo data
            y_demo_j = demonstrator.get_targets(d).to(device).view(-1).float()  # demo labeling on Xj

            y_demo_list.append(y_demo_j)
            demo_feats.append(d["fairness_feats"])

            logits_list = []
            yhat_list = []
            feats_list = []

            for pol in policies:
                logits = pol(Xj).squeeze(-1)            # [N_j]
                probs = torch.sigmoid(logits)
                yhat = sample_binary_from_probs(probs)  # [N_j]

                logits_list.append(logits)
                yhat_list.append(yhat)

                f = compute_fairness_features(Xj, y_true_j, yhat, Aj, self.metrics_list)  # [K]
                feats_list.append(f)

            logits_by_demo.append(torch.stack(logits_list, dim=0))   # [R,N_j]
            yhat_by_demo.append(torch.stack(yhat_list, dim=0))       # [R,N_j]
            rollout_feats_accum.append(torch.stack(feats_list, dim=0))  # [R,K]

        demo_feats = np.stack(demo_feats)  # [D,K]

        # Aggregate rollout feats across demos -> one [R,K]
        rollout_feats = torch.stack(rollout_feats_accum, dim=0).mean(dim=0)  # [R,K]

        # ----------------------------------------------------
        # 2) Subdominance matrix S[R,D]
        # ----------------------------------------------------
        S = compute_subdominance_matrix(
            rollout_feats,
            demo_feats,
            mode=self.subdom_mode,
            alpha=self.alpha if self.alpha is not None else 1.0,
            beta=0,
        )
        S = self.apply_S_temperature(S, beta=ot_temperature)

        # ----------------------------------------------------
        # 3) Directional cost (diagnostic) on aggregated feats
        # ----------------------------------------------------
        dir_cost = compute_directional_cost(
            rollout_feats.detach().cpu().numpy(),
            demo_feats,
            n_dir=n_dir,
        )

        # ----------------------------------------------------
        # 4) OT solve -> gamma
        # ----------------------------------------------------
        out = solve_stochastic_subdom_coupling(
            S,
            solver=solver,
            weight_method="primal",
            normalize_subdom=normalize_s_matrix,
            row_constraints=row_constraints,
        )
        gamma = torch.tensor(out["gamma_np"], device=device, dtype=torch.float32)  # [R,D]

        # reverse subdom aligned [R,D]
        S_demo_roll = compute_subdominance_matrix(
            torch.tensor(demo_feats, device=device, dtype=rollout_feats.dtype),
            rollout_feats.detach(),
            mode=self.subdom_mode,
            alpha=self.alpha if self.alpha is not None else 1.0,
            beta=0,
        )  # [D,R]
        S_rev_ji = S_demo_roll.T  # [R,D]
        indicator = (torch.as_tensor(S, device=device).float() <= torch.as_tensor(S_rev_ji, device=device).float()).float()

        # ----------------------------------------------------
        # 5) Loss + GD step (generic X)
        #   Loss = sum_j loss_on_demo_j using logits_by_demo[j], yhat_by_demo[j], y_demo_list[j]
        # ----------------------------------------------------
        total_loss = torch.zeros((), device=device)
        # compute loss per demo and average (you can weight by demo size later)
        for j in range(D):
            loss_out_j = subdominant_logloss_shared_X_multi_rollout(
                logits_rollouts=logits_by_demo[j],      # [R,N_j]
                yhat_rollouts=yhat_by_demo[j],          # [R,N_j]
                y_demo=torch.stack(y_demo_list, dim=0)[:, : y_demo_list[j].shape[0]]  # not ideal; see note below
                if False else torch.stack([y_demo_list[j]], dim=0),  # placeholder: demo-only (see note)
                gamma=gamma[:, j:j+1],                   # [R,1] coupling to this demo
                indicator_win=indicator[:, j:j+1],       # [R,1]
            )
            total_loss = total_loss + loss_out_j.loss

        total_loss = total_loss / D

        if not no_update:
            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            self.optimizer.step()

        return {
            "train/loss": float(total_loss.detach().cpu()),
            "train/mean_subdom": float(np.asarray(S).mean()),
            "train/std_subdom": float(np.asarray(S).std()),
            "train/fairness": rollout_feats.mean(dim=0).detach().cpu().tolist(),
            "train/directional_cost": dir_cost,
            "train/ot_temperature": float(ot_temperature),
            "train/indicator_mean": float(indicator.mean().detach().cpu()),
            "train/R": int(R),
            "train/D": int(D),
        }

    #--------------------------------------------------------------------------------------

    def _policy_asset_dir(self):
        # relative to this class file
        return Path(__file__).resolve().parent / "../../data/checkpoints"

    #--------------------------------------------------------------------------------------

    def _resolve_policy_spec(self, spec, demonstrator):
        """
        spec can be:
          - "random"
          - "lr"          -> train sklearn logistic on demonstrator.train_demos, convert to torch
          - "something.pkl" / "something.zip"
          - full/relative path to .pkl or .zip

        Returns:
          torch nn.Module
        """
        if not isinstance(spec, str):
            raise TypeError(f"Policy spec must be a string. Got {type(spec)}")

        s = spec.strip().lower()

        # 1) random
        if s == "random":
            return self._clone_policy().to(self.device)

        # 2) train logistic from demos
        if s == "lr":
            preset = 'medium'
            print(f"Training logistic a --{preset}-- model as part of start policy mixture...\n")
            settings = logistic_training_settings("medium")
            sk_model = train_logistic_from_demos(demonstrator.train_demos, **settings)
            return sklearn_model_to_torch_module(
                sk_model,
                device=self.device,
            )

        # 3) file / checkpoint
        raw_spec = Path(spec)

        # if no parent/path given, look in default asset dir
        if raw_spec.parent == Path("."):
            candidate = (self._policy_asset_dir() / raw_spec).resolve()
        else:
            candidate = raw_spec.expanduser().resolve()

        if not candidate.exists():
            raise FileNotFoundError(f"Could not find policy file: {candidate}")

        suffix = candidate.suffix.lower()

        # sklearn logistic saved checkpoint
        if suffix == ".pkl":
            return load_saved_sklearn_policy_into_torch(
                candidate,
                device=self.device,
            )

        # torch / sb3 / custom zip loader
        if suffix == ".zip":
            # replace with your actual zip/bootstrap loader
            return load_saved_bootstrap_policy_into_torch(
                candidate,
                base_policy=self.policy,
                device=self.device,
            )

        raise ValueError(f"Unsupported policy spec/file type: {spec}")

    #--------------------------------------------------------------------------------------

    def _build_source_policies_from_cfg(self, demonstrator):
        """
        Example expected cfg:
          model_cfg.policy_sources = ["random", "lr", "my_model.pkl"]
        """
        specs = self.model_cfg.get("policy_sources", None)
        if specs is None:
            raise ValueError("model_cfg.policy_sources must be provided.")

        if isinstance(specs, str):
            specs = [specs]

        return [self._resolve_policy_spec(spec, demonstrator) for spec in specs]
