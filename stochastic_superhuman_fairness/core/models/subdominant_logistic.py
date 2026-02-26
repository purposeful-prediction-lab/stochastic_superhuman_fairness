import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from stochastic_superhuman_fairness.core.models.logistic import LogisticRegressionModel
from stochastic_superhuman_fairness.core.qp_solver import solve_stochastic_subdom_coupling
from stochastic_superhuman_fairness.core.fairness.fairness_metrics import compute_directional_cost
from stochastic_superhuman_fairness.core.fairness.subdominance import (
        subdominant_logloss_shared_X_multi_rollout,  # the [R,N] logits version
        compute_subdominance_matrix,
        )
from stochastic_superhuman_fairness.core.fairness.fairness_metrics import compute_fairness_features
from stochastic_superhuman_fairness.core.utils import sample_binary_from_probs

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
        # Assumes self.policy is an nn.Module created in LogisticRegressionModel.__init__.
        self.policies = nn.ModuleList([self._clone_policy() for _ in range(self.n_models)])

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

    def _clone_policy(self) -> nn.Module:
        # Create a new instance with the same architecture and copy weights.
        # This is the most robust approach if your policy isn't a simple Linear.
        pol = copy.deepcopy(self.policy)
        return pol

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

    #--------------------------------------------------------------------------------------------------------

    #  def train_one_epoch_subdominant_logistic_sharedX(
    def _train_one_epoch(
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
        ot_temperature: float = 1.0,
        normalize_s_matrix: bool = True,
        stochastic_if_threshold: bool = False,    # (ignored here; we sample stochastically)
        alpha_updates: str = 'analytical',   # analytical, None
        # Grad update specific
        no_update: bool = False,
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

        # ---- choose how many rollout-models to use ----
        policies = self.policies
        if n_rollouts is not None:
            policies = policies[: int(n_rollouts)]
        R = len(policies)
        D = len(demos)

        # ----------------------------------------------------
        # 1) Forward each model on shared X -> logits_i -> probs -> sample yhat_i
        # ----------------------------------------------------
        logits_list = []
        yhat_list = []
        feat_list = []

        for pol in policies:
            logits = pol(X).squeeze(-1)          # [N]
            probs = torch.sigmoid(logits)         # [N]
            yhat = sample_binary_from_probs(probs)  # [N] sampled decisions

            logits_list.append(logits)
            yhat_list.append(yhat)
            # fairness feats use ground truth
            f = compute_fairness_features(y_true, yhat, A, self.metrics_list, weights = [1,1,1,1,10])  # [K]
            feat_list.append(f)

        logits_rollouts = torch.stack(logits_list, dim=0)    # [R,N]
        yhat_rollouts   = torch.stack(yhat_list, dim=0)      # [R,N]
        rollout_feats   = torch.stack(feat_list, dim=0)      # [R,K]

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
            beta=0,
        )
        S = self.apply_S_temperature(S, beta=ot_temperature)

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
        out = solve_stochastic_subdom_coupling(
            S,
            solver=solver,
            weight_method="primal",
            normalize_subdom=normalize_s_matrix,
            row_constraints=row_constraints,
        )

        import ipdb;ipdb.set_trace()
        gamma = torch.tensor(out["gamma_np"], device=device, dtype=torch.float32)  # [R,D]
        #  weights = torch.tensor(gamma.sum(axis=1), device=device, dtype=torch.float32)  # [R]

        # reverse subdom (demo -> rollout), aligned to [R,D]
        S_demo_roll = compute_subdominance_matrix(
            demo_feats,
            rollout_feats.detach(),
            mode=self.subdom_mode,
            alpha=self.alpha if self.alpha is not None else 1.0,
            beta=0,
        )  # [D,R]
        S_rev_ji = S_demo_roll.T  # [R,D]
        S_rev_ji = self.apply_S_temperature(S_rev_ji, beta=ot_temperature)

        indicator = (torch.as_tensor(S, device=device).float() <= torch.as_tensor(S_rev_ji, device=device).float()).float()
        indicator = (torch.as_tensor(S, device=device).float() <= torch.as_tensor(S_rev_ji, device=device).float()).float()

        #  import ipdb;ipdb.set_trace()
        # ----------------------------------------------------
        # 5) Loss + GD step (uses your new loss)
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

        #  import ipdb;ipdb.set_trace()
        return {
            "train/loss": float(loss_out.loss.detach().cpu()),
            "train/mean_subdom": float(S.mean()),
            "train/std_subdom": float(S.std()),
            "train/fairness": rollout_feats.mean(dim=0).detach().cpu().tolist(),
            "train/directional_cost": dir_cost,
            "train/ot_temperature": float(ot_temperature),
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
