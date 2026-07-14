# Torch Libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import stack_module_state, functional_call, vmap
from typing import Union
# numpy and computation
import numpy as np
#  Stochastic Superhuman Fairness Libs
from stochastic_superhuman_fairness.core.dataclasses.model_dataclasses import CouplingConfig
from stochastic_superhuman_fairness.core.models.logistic import LogisticRegressionModel
from stochastic_superhuman_fairness.core.qp_solver import solve_stochastic_subdom_coupling, bj_from_beatrates_nocollapse
from stochastic_superhuman_fairness.core.fairness.fairness_metrics import compute_directional_cost
from stochastic_superhuman_fairness.core.fairness.subdominance import (
        subdominant_logloss_shared_X_multi_rollout,  # the [R,N] logits version, reference
        subdominant_weighted_logloss_shared_X_multi_rollout,  # the [R,N] logits version, reference
        behavior_guided_subdominant_spring_loss,
        compute_subdominance_matrix,
        compute_subdominance_matrix,
        )
from stochastic_superhuman_fairness.core.fairness.fairness_metrics import compute_fairness_features, compute_fairness_features_batched, compute_fairness_features_batched_fast
from stochastic_superhuman_fairness.core.utils import sample_binary_from_probs, sample_rollouts_from_probs, compute_sample_distribution, rollout_scores
from stochastic_superhuman_fairness.core.models.ensemble_utils import mix_policies_, params_changed, reset_optimizer_state_
from stochastic_superhuman_fairness.core.gamma_utils import  smooth_gamma, compute_subdom_policy_weights_torch
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
from dataclasses import asdict

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
        self.old_gamma = None
        self.old_top1 = None

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


        # optional: initialize ensemble members slightly differently
        init_noise = float(self.model_cfg.get("init_noise_std", 0.05))
        if init_noise > 0:
            self._perturb_ensemble(init_noise)
        self.init_policies = [p.clone() for p in self.policies.parameters()]
        # optimizer (can be replaced externally)
        lr = float(self.cfg.get("train", {}).get("lr", 1e-3))
        wd = float(self.cfg.get("train", {}).get("weight_decay", 0.0))
        if self.cfg.get("train", {}).get('optimizer', 'sgd') == 'sgd':
            self.optimizer = torch.optim.SGD(self.policies.parameters(), lr=lr, weight_decay=wd)
        else:
            self.optimizer = torch.optim.Adam(self.policies.parameters(), lr=lr, weight_decay=wd)
        # Other Parameters
        self.post_eval_transfer = int(self.model_cfg.get("post_eval_transfer", False))

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
        n_mini_epochs: int = 10,
        indicator_func: str = None,
        weighted_S_matrix: bool = False,
        row_constraints: bool = False,
        col_constraints: str = None, # None, rev_ranking
        ot_temperature: float = 1.0,
        gamma_temperature: float = 1.0,
        sparse_gamma: bool = False,
        normalize_s_matrix: bool = True,
        stochastic_if_threshold: bool = False,
        alpha_updates: str = "analytical",
        no_update: bool = False,
        use_demos_as_gtruth: bool = False,
        compute_intrademo_ot_loss: bool = True,
        loss_fn: str = None,
        loss_fn_kwargs: dict = {},
        epoch: int = None,
        coupling_cfg: CouplingConfig = CouplingConfig(),
        **kwargs,
    ):
        '''This function assumes shared x among demos.'''
        demos = demonstrator.train_demos
        device = next(self.policies.parameters()).device
        compute_intrademo_ot_loss = compute_intrademo_ot_loss and (epoch % 100 == 0)
        # ---- shared data ----
        X = demos[0]["X"].to(device)
        A = demos[0]["A"]
        A = A.to(device) if torch.is_tensor(A) else torch.as_tensor(A, device=device)
        y_true = demos[0]["y"].to(device).view(-1).float()

        y_demo = torch.stack(
            [demonstrator.get_targets(d).to(device).view(-1).float() for d in demos],
            dim=0
        )  # [D, N]

        #  import ipdb;ipdb.set_trace()
        policies = self.policies
        P = len(policies)
        M = int(n_rollouts) if n_rollouts is not None else 1
        D = len(demos)
        R = P * M
        # Init gamma to uniform for the first run
        #  old_gamma = torch.ones((R, D)) / (R*D) if self.old_gamma is None else self.old_gamma
        metric_weights = torch.ones(len(self.metrics_list), device=device, dtype=torch.float32)
        logits_per_policy = []
        yhat_per_policy = []
        feat_per_policy = []

        for pol in policies:
            logits = pol(X).squeeze(-1)                      # [N]
            probs = torch.sigmoid(logits)                    # [N]

            #  import ipdb;ipdb.set_trace()
            if (decision_threshold is not None) and (not stochastic_if_threshold):
                yhat_batch = (probs.unsqueeze(0).expand(M, -1) >= decision_threshold).float()   # [M, N]
            else:
                probs_batch = probs.unsqueeze(0).expand(M, -1)                                   # [M, N]
                yhat_batch = torch.bernoulli(probs_batch)                                         # [M, N]

            #  yhat_batch = torch.zeros_like(probs_batch)
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
                #  reduce='mean'
                reduce = getattr(self.cfg.get('subdominance', {}), "alpha_reduce", 'mean'),
            )

        S = compute_subdominance_matrix(
            rollout_feats,
            demo_feats,
            mode=self.subdom_mode,
            alpha=self.alpha if self.alpha is not None else 1.0,
            beta=self.beta,
            feat_reduce=self.feat_reduce,
        )  # [R,D]

        # ----------------------------------------------------
        # 3) Directional cost
        # ----------------------------------------------------
        dir_cost = compute_directional_cost(
            rollout_feats.detach().cpu().numpy(),
            demo_feats,
            n_dir=n_dir,
        )

        if row_constraints:
            s_probs = compute_sample_distribution(logits_policies, y_demo)
        else:
            if self.old_gamma is None:
                s_probs = (torch.ones((R,D)) / (R*D)).sum(axis=1)
            else:
                s_probs = self.old_gamma.sum(axis=1)
            s_probs = s_probs.reshape(-1, M).sum(axis=1) / s_probs.sum()

        # ----------------------------------------------------
        # 4) OT solve -> gamma
        # ----------------------------------------------------
        # None = uniiform | rev_ranking ~ -e(T*beat_rate) | ranking ~ e(T*beat_rate)
        if col_constraints == 'policy_probs':
            bj_priors = s_probs.repeat_interleave(M) / M
        else:
            bj_priors = bj_from_beatrates_nocollapse(demonstrator.beat_rates_train, rank_type = col_constraints)

        S_OT = S * torch.tensor(bj_priors, device=S.device) if weighted_S_matrix else S
        out = solve_stochastic_subdom_coupling(
            S_OT,
            P,   # num of policies in ensemple
            prev_gamma = None if self.old_gamma is None else self.old_gamma.detach().cpu().numpy(),
            demo_marginals=bj_priors,
            **asdict(coupling_cfg),
        )
        gamma = gamma_temperature * torch.tensor(
            out["gamma_np"], device=device, dtype=torch.float32
        )  # [R,D]
        if sparse_gamma:
            max_idxs = gamma.argmax(dim=1, keepdim=True)
            gamma.zero_()
            gamma.scatter_(1, max_idxs, 1.0)
        S_demo_roll = compute_subdominance_matrix(
            demo_feats,
            rollout_feats.detach(),
            mode=self.subdom_mode,
            alpha=self.alpha if self.alpha is not None else 1.0,
            beta=self.beta,
        )  # [D,R]

        S_rev_ji = self.apply_S_temperature(S_demo_roll.T, beta=ot_temperature)  # [R,D]

        lower_subdom_indicator = (
            torch.as_tensor(S, device=device).float()
            <= torch.as_tensor(S_rev_ji, device=device).float()
        ).float()
        # Select rollouts whose subdom is less than the weighted subdom of demos.
        indicator, criterion = self.get_outperforming_rollout_idxs(S, demonstrator, gamma, indicator_func = indicator_func)
        #  indicator = (S <= demonstrator.train_intrademo_subdom_dict['median']).float()
        indicator_rev = (
            torch.as_tensor(S_rev_ji, device=device).float()
            <= torch.as_tensor(S, device=device).float()
        ).float()
        # ----------------------------------------------------
        # 5) Loss + GD step
        # ----------------------------------------------------
        for mini_ep in range(n_mini_epochs):

            old_gamma = self.old_gamma.detach() if self.old_gamma is not None else gamma.detach()
            effective_gamma = smooth_gamma(gamma.detach(), old_gamma)
            # recompute under current model params
            logits_rollouts = self.ensemble_logits_shared_X(
                X,              # or shared X
                require_grad=True,
            )  # [P, N] 
            logits_rollouts = logits_rollouts.repeat_interleave(M, dim=0) # [P*M,N] = [R,N]

            #  import ipdb;ipdb.set_trace()
            if loss_fn == 'weighted' or loss_fn == 'bc_guided':
                if loss_fn == 'weighted':
                    loss_func = subdominant_weighted_logloss_shared_X_multi_rollout
                else:
                    loss_func = behavior_guided_subdominant_spring_loss

                loss_out = loss_func(
                        S.detach(),
                        torch.tensor(S_rev_ji).detach().to(S.device),
                        logits_rollouts,  # [R,N]
                        yhat_rollouts.detach(),    # [R,N]
                        y_demo.detach(),           # [D,N]
                        effective_gamma.detach(),            # [R,D]
                        indicator.detach(),        # [R,D]
                        criterion.detach(),        # [R]
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
            #  if not no_update:
            if epoch < 2001:
                self.optimizer.zero_grad(set_to_none=True)
                loss_out.loss.backward()
                self.optimizer.step()
        # Mini Pochs end
        # --------------

        loss_term_dict = {'dominant_rollouts': int(lower_subdom_indicator.sum().item()),
                          'dominant_demos': int(indicator_rev.sum()),
                          'per_mode_dominant_rollouts': indicator.sum(axis=1).reshape(P, n_rollouts).tolist(),
                          'per_mode_dominant_demos': indicator_rev.sum(axis=1).reshape(P, n_rollouts).tolist(),
                          'S_mean': loss_out.info['S_mean'],
                          'norm_paired_subdom': loss_out.info['norm_paired_subdom'],
                          'paired_subdom': loss_out.info['paired_subdom'],
                          'demo_logprobs':[loss_out.info['demo_logprobs'][m* n_rollouts].sum(dim=1).cpu().tolist() for m in range(P)],
                          'loss_terms': [loss_out.info['term1'], loss_out.info['term2']]
                          }

        if compute_intrademo_ot_loss:
            demo_baseline_dict = self.compute_demo_logloss_from_matching(
                    demonstrator,
                    labels=y_demo,      # [D, N]
                    logits=loss_out.info['demo_logprobs'][:: n_rollouts],      # [D, N]
                    indicator_func = indicator_func,
                    loss_fn_kwargs = loss_fn_kwargs,
                )

        
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
        #  import ipdb;ipdb.set_trace()
        returns = {
                    "train/loss": float(loss_out.loss.detach().cpu()),
                    "train/mean_subdom": float(S.mean()),
                    "train/std_subdom": float(S.std()),
                    "train/fairness": rollout_feats.mean(dim=0).detach().cpu().tolist(),
                    "train/directional_cost": dir_cost,
                    "train/ot_temperature": float(ot_temperature),
                    "train/indicator_mean": float(indicator.mean().detach().cpu()),
                    "train/policy_probs":s_probs.detach().cpu().tolist(),
                    "train/R": int(R),
                    "train/P": int(P),
                    "train/D": int(D),
                    "train/l_terms": loss_term_dict,
                    "gamma_matrix": gamma,
                    "gamma_diagnostics": out['gamma_diagnostics'],
                    "S_diagnostics": out['S_diagnostics'],
                }

        if compute_intrademo_ot_loss:
            returns.update({"train/demo_baseline_dict": {'loss':demo_baseline_dict['loss_float'], 
                                                         'loss_per_model': demo_baseline_dict['loss_per_model_float']}})
        # Next iter assignments
        #  import ipdb;ipdb.set_trace()
        self.old_gamma = gamma.detach()

        return returns

    #--------------------------------------------------------------------------------------------------------

    def get_outperforming_rollout_idxs(self, S, demonstrator, gamma, indicator_func = 'median_intrademo_subdom',
                                       indicator_temp: float = 0.05):
        ''' Get idxs of well performing rollouts. THese have their likelihood increased by  the loss func
            The rest (underperforming) are instead ignored and demo likelihood under parameters ins increased.
        '''
        soft_indicators = False
        if 'soft' in indicator_func:
            soft_indicators = True
            indicator_func = indicator_func.split('_',1)[-1]
            
        is_torch = isinstance(S, torch.Tensor)
        if indicator_func == 'median_intrademo_subdom':
            crit = demonstrator.train_intrademo_subdom_dict['median']
        elif indicator_func == 'median':
            crit = S.median()
        elif indicator_func == 'mean':
            crit = S.mean()
        elif indicator_func == None:
            crit = 0.0
        elif indicator_func == 'match_sum':
            crit = (gamma * S).sum(axis=1) / (gamma.sum(axis=1) + 1e-5)

        crit = torch.as_tensor(crit, device=S.device) if is_torch else np.asarray(crit)
        #  import ipdb;ipdb.set_trace()
        if soft_indicators:
            out = torch.sigmoid((crit[:, None] - S) / indicator_temp).float()
        else:
            out = (S < crit) if crit.ndim == 0 else (S < crit[:, None])
        return (out.float() if is_torch else out.astype(float), crit)

    #--------------------------------------------------------------------------------------------------------

    def post_eval(self, eval_stats: dict):
        ''' This function will perform a convex combination with parameter lambda between the best performing
            ensemble module and the 50% worst performing ones.
        '''
        if self.post_eval_transfer:
            zero_one_losses = np.array([es['eval/zero_one_loss'] for es in eval_stats['per_policy']])
            low_perf_idxs = np.where(zero_one_losses > np.median(zero_one_losses))[0]
            best_policy_idx = np.argmin(zero_one_losses).item()
            #  changed, details = params_changed(
            #      self,
            #      mix_policies,          # your function
            #      (self, best_policy_idx, low_perf_idxs),                  # fn args start here
            #      return_details=True,
            #  )

            #  print("Changed:", changed)
            #  print(details)  # {(policy_idx, param_name): max_abs_diff}
            mix_policies_(self, best_policy_idx, low_perf_idxs)
            # Good practise to reset optimizers buffers for strongly altered params, especially with Adam.
            modified_params = []
            for b_idx in low_perf_idxs:
                modified_params += list(self.policies[b_idx].parameters())

            reset_optimizer_state_(self.optimizer, modified_params)
        return
    #--------------------------------------------------------------------------------------------------------

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

    def compute_demo_logloss_from_matching(
        self,
        demonstrator,
        labels,
        logits,
        indicator_win=None,
        indicator_func: str = None,
        loss_fn_kwargs: dict = {},
        reduction: str = "sum",   # "sum" | "mean" | "none"
        require_grad: bool = False,
    ):
        """
        Compute demo-demo loss after OT matching is already done.

        gamma  : [D, D] matching/coupling matrix
        labels : [D, N] demo decisions/labels
        logits : [M, D, N] ensemble logits, or [D, N] for one model

        Matching is fixed and shared across all M models.
        """
        gamma = demonstrator.intrademo_gamma_torch
        S = demonstrator.train_intrademo_S_torch
        ctx = torch.enable_grad() if require_grad else torch.no_grad()
        with ctx:
            gamma = torch.as_tensor(gamma, dtype=torch.float32)
            device = gamma.device

            labels = torch.as_tensor(labels, device=device, dtype=torch.float32)
            logits = torch.as_tensor(logits, device=device, dtype=torch.float32)

            if logits.ndim == 2:
                logits = logits.unsqueeze(0)  # [1, D, N]

            assert gamma.ndim == 2 and gamma.shape[0] == gamma.shape[1], (
                f"Expected gamma [D,D], got {gamma.shape}"
            )
            assert labels.ndim == 2, f"Expected labels [D,N], got {labels.shape}"
            assert logits.ndim == 3, f"Expected logits [M,D,N], got {logits.shape}"

            D = gamma.shape[0]
            M, D_logits, N = logits.shape

            assert labels.shape == (D, N), (
                f"Expected labels {(D, N)}, got {labels.shape}"
            )
            assert D_logits == D, (
                f"logits second dim must match D={D}, got {D_logits}"
            )

            if indicator_win is None:
                indicator_win = (gamma > 0).float()
            else:
                indicator_win = torch.as_tensor(
                    indicator_win, device=device, dtype=torch.float32
                )
                assert indicator_win.shape == gamma.shape, (
                    f"indicator_win shape {indicator_win.shape} != gamma shape {gamma.shape}"
                )

            losses = []
            infos = []

            #  import ipdb;ipdb.set_trace()
            indicator, criterion = self.get_outperforming_rollout_idxs(S, demonstrator, gamma, indicator_func = indicator_func)
            for m in range(M):
                loss_out = subdominant_weighted_logloss_shared_X_multi_rollout(
                    S,
                    S,
                    logits[m],  # [R,N]
                    labels,    # [R,N]
                    labels,           # [D,N]
                    gamma,            # [R,D]
                    indicator_win,        # [R,D]
                    criterion,        # [R]
                    **loss_fn_kwargs,
                )
                losses.append(loss_out.loss)
                infos.append(getattr(loss_out, "info", {}))

            loss_per_model = torch.stack(losses)

            if reduction == "sum":
                loss = loss_per_model.sum()
            elif reduction == "mean":
                loss = loss_per_model.mean()
            elif reduction == "none":
                loss = loss_per_model
            else:
                raise ValueError(f"Unknown reduction: {reduction}")

            return {
                "loss": loss,
                "loss_float": (
                    float(loss.detach().cpu())
                    if reduction != "none"
                    else loss.detach().cpu().tolist()
                ),
                "loss_per_model": loss_per_model,
                "loss_per_model_float": loss_per_model.detach().cpu().tolist(),
                "gamma": gamma,
                "indicator_win": indicator_win,
                "info_per_model": infos,
            }
