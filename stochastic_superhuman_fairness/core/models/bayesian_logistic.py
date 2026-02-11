import torch
import torch.nn as nn
import numpy as np

from stochastic_superhuman_fairness.core.models.logistic import LogisticRegressionModel
from stochastic_superhuman_fairness.core.models.stochastic_mixin import StochasticParamDistMixin
from stochastic_superhuman_fairness.core.qp_solver import solve_stochastic_subdom_coupling
from stochastic_superhuman_fairness.core.fairness.fairness_metrics import compute_directional_cost
from stochastic_superhuman_fairness.core.models.utils import (
    phi_features,
    phi_mean,
    phi_mean_per_demo,
)
from stochastic_superhuman_fairness.core.fairness.subdominance import (
    subdominance_loss_from_features,
    compute_subdominance_matrix,
)
from stochastic_superhuman_fairness.core.fairness.compute_fairness_utils import compute_fairness_features
from stochastic_superhuman_fairness.core.rollout_utils import collect_bayesian_rollouts


class BayesianLogisticRegressionModel(StochasticParamDistMixin, LogisticRegressionModel):
    """
    Logistic regression with full fairness + subdominance loop.

    Core update:
        1. Predict y_hat
        2. Compute φ_rollout and φ_demos
        3. Compute fairness metrics
        4. Compute subdominance matrix and weights
        5. Update weights by subdominance-weighted φ difference
    """

    def __init__(self, cfg, demonstrator):
        super().__init__(cfg, demonstrator)
        self.dist_initialized = 0
        # --- bayesian configuration ---
        scfg = cfg.get('subdominance', {})
        self.subdom_mode = scfg.get("mode", "absolute")
        self.subdom_type = scfg.get("type", "standard")
        self.subdom_agg = scfg.get("rollout_aggregate", "mean")
        self.subdom_weight_mode = scfg.get("weight_mode", "softmax")  # or "linear"
        self.alpha = scfg.get("alpha")
        self.beta = scfg.get("beta")

    def train_one_epoch_stochastic_bayesian(
        self,
        demonstrator,
        batch_size: int = 1,
        n_dir: int = 20,
        decision_threshold: float | None = None,
        stochastic_if_threshold: bool = False,
        # --- distribution selection ---
        dist_mode: str = "per_param_diag",   # "per_param_diag" or "full_param_mvn"
        # --- OT temperature ---
        beta: float = 1.0,
        # --- update knobs (shared) ---
        ema: float = 0.2,
        max_mean_delta: float | None = None,
        # --- per_param_diag knobs ---
        var_floor: float = 1e-8,
        var_ratio_clip: tuple[float, float] = (0.5, 2.0),
        # --- full_param_mvn knobs ---
        cov_floor: float = 1e-8,
        diag_ratio_clip: tuple[float, float] = (0.5, 2.0),
        shrinkage: float = 0.0,
        set_policy_to_mean_after: bool = True,
        **kwargs
    ):
        demos = demonstrator.train_demos
        device = next(self.policy.parameters()).device

        rollout_feats = []
        eps_list = []                 # for per_param_diag
        theta_vec_list = []           # for full_param_mvn

        if self.dist_initialized == 0:
            self.init_dist(dist_mode, init_var = 1e-4)
        # ----------------------------------------------------
        # 1) Collect rollouts: sample params -> decisions -> feats
        # ----------------------------------------------------
        rb = self.collect_bayesian_rollouts(
            demonstrator,
            demos=demonstrator.train_demos,
            dist_mode=dist_mode,
            decision_threshold=decision_threshold,
            restore_policy_to_mean=False,  # we’ll do it after update_dist like before
        )
        #  for d in demos:
        #      Xd = d["X"].to(device)
        #      yd = demonstrator.get_targets(d).to(device)
        #      Ad = d["A"]
        #      Ad = Ad.to(device) if torch.is_tensor(Ad) else Ad
        #
        #      # sample parameters from chosen distribution
        #      theta, aux = self.sample_dist(dist_mode)
        #      self.load_params_into_policy(theta)
        #
        #      # forward + decisions (uses current live policy)
        #      probs = torch.sigmoid(self.policy(Xd).squeeze(-1))
        #      y_hat = self._sample_actions_from_probs(
        #          probs,
        #          threshold=decision_threshold,
        #      )
        #
        #      f_r = compute_fairness_features(Xd, yd, y_hat, Ad, self.metrics_list)
        #      rollout_feats.append(f_r)
        #      if dist_mode == "per_param_diag":
        #          eps_list.append(aux["eps"])
        #      elif dist_mode == "full_param_mvn":
        #          theta_vec_list.append(aux["theta_vec"])
        #      else:
        #          raise ValueError(f"Unknown dist_mode: {dist_mode}")
        #
        #  rollout_feats = torch.stack(rollout_feats, dim=0)  # [R,K]

        rollout_feats = rb.feats # [R, K]
        # Aux for later updates (same semantics as your original code)
        eps_list = None
        theta_vec_list = None
        if dist_mode == "per_param_diag":
            eps_list = [a["eps"] for a in rb.aux_list]
        elif dist_mode == "full_param_mvn":
            theta_vec_list = [a["theta_vec"] for a in rb.aux_list]
        else:
            raise ValueError(f"Unknown dist_mode: {dist_mode}")
        demo_feats = np.stack([d["fairness_feats"] for d in demos])  # [D,K]

        # ----------------------------------------------------
        # 2) Subdominance matrix
        # ----------------------------------------------------
        self.compute_alpha(rollout_feats, demonstrator.train_demo_means_sorted, mode = self.subdom_mode, reduce = 'mean')
        #  import ipdb;ipdb.set_trace()
        S = compute_subdominance_matrix(
            rollout_feats,
            demo_feats,
            mode=self.subdom_mode,
            alpha=self.alpha,
            beta=self.compute_beta(),
        )

        # OT temperature
        S = self.apply_ot_temperature(S, beta=beta)

        # ----------------------------------------------------
        # 3) Directional cost (diagnostic)
        # ----------------------------------------------------
        dir_cost = compute_directional_cost(
            rollout_feats.detach().cpu().numpy(),
            demo_feats,
            n_dir=n_dir,
        )

        # ----------------------------------------------------
        # 4) OT solve -> weights
        # ----------------------------------------------------
        out = solve_stochastic_subdom_coupling(
            S,
            solver="mosek",
            weight_method="primal",
            debias_rowcol=True,
            normalize=True,
        )

        if "gamma_np" in out:
            gamma = out["gamma_np"]  # [R,D]
            weights = torch.tensor(gamma.sum(axis=1), device=device, dtype=torch.float32)  # [R]
        else:
            weights = torch.tensor(out["weights_np"], device=device, dtype=torch.float32)  # [R]

        # ----------------------------------------------------
        # 5) Update distribution
        # ----------------------------------------------------
        #  import ipdb;ipdb.set_trace()
        if dist_mode == "per_param_diag":
            self.update_dist(
                "per_param_diag",
                eps_list=eps_list,
                weights=weights,
                ema=ema,
                var_floor=var_floor,
                max_mean_delta=max_mean_delta,
                var_ratio_clip=var_ratio_clip,
            )
            if set_policy_to_mean_after:
                self.load_params_into_policy(self.per_param_mean)

        elif dist_mode == "full_param_mvn":
            self.update_dist(
                "full_param_mvn",
                theta_vec_list=theta_vec_list,
                weights=weights,
                ema=ema,
                cov_floor=cov_floor,
                max_mean_delta=max_mean_delta,
                diag_ratio_clip=diag_ratio_clip,
                shrinkage=shrinkage,
            )
            if set_policy_to_mean_after:
                # set live policy to full_mu
                theta_mean = self._unflatten_to_theta_dict(self.full_mu, self.full_template, self.full_names)
                self.load_params_into_policy(theta_mean)

        # ----------------------------------------------------
        # 6) Metrics
        # ----------------------------------------------------
        return {
            "train/mean_subdom": float(S.mean()),
            "train/std_subdom": float(S.std()),
            "train/fairness": rollout_feats.mean(dim=0).detach().cpu().tolist(),
            "train/directional_cost": dir_cost,
            "train/beta": float(beta),
            "train/dist_mode": dist_mode,
        }

    #------------------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def collect_bayesian_rollouts(self, demonstrator, demos=None, dist_mode="per_param_diag", restore_policy_to_mean = False, decision_threshold=None):
        # use the wrapper we wrote earlier (with before_demo sampling)
        return collect_bayesian_rollouts(
            self,
            demonstrator,
            demos=demos,
            dist_mode=dist_mode,
            restore_policy_to_mean = restore_policy_to_mean,
            decision_threshold=decision_threshold,
        )
