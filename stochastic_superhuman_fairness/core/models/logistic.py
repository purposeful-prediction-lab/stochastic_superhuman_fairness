import torch
import torch.nn as nn
import numpy as np

from stochastic_superhuman_fairness.core.models.base_model import BaseModel
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
class LogisticRegressionModel(BaseModel):
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
        meta = demonstrator.get_metadata()
        n_features = meta["n_features"]

        self.policy = nn.Linear(n_features, 1, bias=True)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = cfg.get("lr", 1e-3)
        #  import ipdb;ipdb.set_trace()
        self.device = cfg.get("device", "cpu")
        self.policy.to(self.device)

    # ----------------------------------------------------------
    def parameters(self):
        return self.policy.parameters()
    # ----------------------------------------------------------
    def forward(self, X):
        return self.policy(X)
    # ----------------------------------------------------------
    def forward_for_loss(self, X):
        logits = self.policy(X).squeeze(-1)
        probs = torch.sigmoid(logits)
        return logits, probs
    # ----------------------------------------------------------
    def compute_phi(self, X, y, add_bias=False, y_domain="01"):
        return phi_features(X, y, add_bias=add_bias, y_domain=y_domain)
    # ----------------------------------------------------------
    def phi_mean(self, X, y, add_bias=False, y_domain="01"):
        return phi_mean(X, y, add_bias=add_bias, y_domain=y_domain)
    # ----------------------------------------------------------
    def exp_phi(self, demos, add_bias=False, y_domain="01"):
        return super().exp_phi(demos, add_bias=add_bias, y_domain=y_domain)
    # ----------------------------------------------------------

    def train_gradient_descent_step(self, X, y):
        """Standard BCE + optional subdominance gradient signal."""
        self.opt.zero_grad(set_to_none=True)
        logits = self.policy(X)
        loss = self.loss_fn(logits.squeeze(), y)
        loss.backward()
        self.opt.step()
        return {"loss": float(loss.item())}
    # ----------------------------------------------------------
    def train_step(self, X=None, y=None):
        """Compatibility placeholder; logistic model trains per epoch."""
        return {"status": "LogisticRegressionModel uses train_one_epoch() instead."}
    # ----------------------------------------------------------
    def compute_weighted_closed_form_grad(self, demos, weights, target_key = 'y', y_domain="01"):
        """
        Torch-native closed-form gradient for standard + stochastic modes:

            g = Σ_r  w_r * (φ_r - φ_demo_mean)
            target_key: str = y or y_demo for logistic_regression produced y labels.
        All inputs may be on CPU or CUDA. Output is a torch tensor [F+1].
        """
        device = self.device

        # ---- 1) φ_r for each rollout (torch) ----
        phi_r_list = []
        for d in demos:
            Xd = d["X"].to(device)
            yd = d[target_key].to(device)
            phi_r = self.phi_mean(Xd, yd, add_bias=False, y_domain=y_domain)   # [F+1], torch
            phi_r_list.append(phi_r)

        phi_r = torch.stack(phi_r_list, dim=0)   # [R, F+1]

        # ---- 2) φ_demo_mean (torch) ----
        # phi_mean_per_demo still returns numpy → convert once
        #  phi_demo_np = self.phi_mean_per_demo(demos, add_bias=False, y_domain=y_domain)
        #  phi_demo_mean = torch.as_tensor(phi_demo_np, device=device, dtype=torch.float32)
        phi_demo_mean = self.exp_phi(demos, add_bias = False)

        # ---- 3) convert weights to torch ----
        if not torch.is_tensor(weights):
            weights = torch.as_tensor(weights, device=device, dtype=torch.float32)
        weights = weights.view(-1, 1)   # [R,1]

        # ---- 4) closed-form formula ----
        g = (weights * (phi_r - phi_demo_mean)).sum(dim=0)   # [F+1]
        #  import ipdb;ipdb.set_trace()
        return g

    # ----------------------------------------------------------
    @torch.no_grad()
    def sample_actions(self, X, threshold: float | None = None):
        """
        Returns (y_hat_prob, y_sampled).
        If threshold is not None -> deterministic 0/1 via threshold.
        Else -> stochastic Bernoulli sample per row.
        """
        import ipdb;ipdb.set_trace()
        logits = self.policy(X).squeeze(-1)
        p = torch.sigmoid(logits)
        if threshold is not None:
            y_samp = (p >= threshold).to(p.dtype)
        else:
            y_samp = torch.bernoulli(p)  # samples 0/1 with prob p

        return p, y_samp

    # ----------------------------------------------------------
    def train_one_epoch_standard(self, demonstrator, batch_size: int = 1, decision_threshold: float = None):
        device = self.device
        demonstrator.to_torch(device)

        demos = demonstrator.train_demos
        # demo fairness (torch) for all demos once
        f_d = torch.as_tensor(
            np.stack([d["fairness_feats"] for d in demos]),
            device=device, dtype=torch.float32
        )

        total_subdom = total_std = total_zero_one = 0.0
        batch_count = 0

        for demo_batch in demonstrator.iter_batches(batch_size=batch_size, shuffle=True):
            batch_count += 1

            rollout_feats = []
            zero_one_list = []

            # ---------- 1) collect rollouts ----------
            for d in demo_batch:
                X = d["X"].to(device)
                y = d["y"].to(device)
                y_demo = demonstrator.get_targets(d).to(device)
                A = d["A"].to(device)

                P, y_hat = self.sample_actions(X, threshold = decision_threshold)

                zero_one_list.append(self.zero_one_loss(y_hat, y).item())

                f_r = compute_fairness_features(X, y, y_hat, A, self.metrics_list)
                rollout_feats.append(f_r)
                #  import ipdb;ipdb.set_trace()

            rollout_feats = torch.stack(rollout_feats, dim=0)  # [R,K]

            # ---------- 2) subdominance via dispatcher ----------
            subdom_out = subdominance_loss_from_features(
                rollout_feats=rollout_feats,
                demo_feats=f_d,
                mode=self.subdom_mode,
                agg=self.subdom_agg,
                alpha=self.compute_alpha(),
                beta=self.compute_beta(),
                reduction="none",
            )
            S = subdom_out["S"]      # [R,D]
            v = subdom_out["per_rollout"]      # [R] per-rollout aggregated subdom
            #  import ipdb;ipdb.set_trace()
            weights_t = v            # standard: per-rollout weights = v
            total_subdom += weights_t.mean().item()
            total_std    += S.std().item()
            total_zero_one += float(np.mean(zero_one_list))

            # ---------- 3) closed-form update (uses numpy helper) ----------
            #  weights_np = weights_t.detach().cpu().numpy()
            g = self.compute_weighted_closed_form_grad(
                demos=demo_batch,
                weights=weights_t,
                target_key = demonstrator.target_key(),
                y_domain="01",
            )  # np array [F+1]

            #  g_w, g_b = g[:-1], g[-1]
            g_w = g
            self.policy.weight.data -= self.lr * torch.as_tensor(g_w, dtype=torch.float32, device=device)
            #  self.policy.bias.data   -= self.lr * torch.as_tensor([g_b], dtype=torch.float32, device=device)
        return {
            "train/zero_one_loss": total_zero_one / batch_count,
            "train/mean_subdom":   total_subdom / batch_count,
            "train/std_subdom":    total_std / batch_count,
            "train/fairness":      rollout_feats.mean(dim=0).detach().cpu().tolist(),
        }
    # ----------------------------------------------------------
    def train_one_epoch_stochastic(self, demonstrator, batch_size=1, n_dir=20, decision_threshold: float = None):
        """
        Stochastic subdominance training for Logistic Regression:
          1) One rollout per demo → compute fairness_feats
          2) Build full subdominance matrix S[R,D]
          3) Solve OT → get gamma + duals + weights
          4) UPDATE model (closed-form stub for now)
          5) Return metrics
        """
        demos = demonstrator.train_demos
        R = len(demos)
        #  import ipdb;ipdb.set_trace()
        # ----------------------------------------------------
        # 1) Collect rollouts (compute fairness features only)
        # ----------------------------------------------------
        rollout_feats = []
        for d in demos:
            Xd, y, yd, Ad = d["X"], d['y'], demonstrator.get_targets(d), d["A"]
            P, y_hat = self.sample_actions(Xd, threshold = decision_threshold)
            #  logits = Xd.cpu().numpy() @ self.policy.weight.detach().cpu().numpy() + self.policy.bias.detach().cpu().numpy()
            f_r = compute_fairness_features(Xd, yd, y_hat, Ad, self.metrics_list)
            rollout_feats.append(f_r)

        rollout_feats = torch.stack(rollout_feats, dim=0)  # [R,K]
        #  rollout_feats = np.stack(rollout_feats)         # [R, Kf]

        # demo fairness matrix
        demo_feats = np.stack([d["fairness_feats"] for d in demos])   # [D,Kf]

        # ----------------------------------------------------
        # 2) Compute subdominance matrix S[R,D]
        # ----------------------------------------------------
        S = compute_subdominance_matrix(
            rollout_feats,
            demo_feats,
            mode=self.subdom_mode,
            alpha=self.compute_alpha(),       # placeholder
            beta=self.compute_beta(),         # placeholder
        )

        # ----------------------------------------------------
        # 3) Compute directional costs (pre-OT)
        # ----------------------------------------------------
        import ipdb;ipdb.set_trace()
        dir_cost = compute_directional_cost(rollout_feats.cpu().numpy(), demo_feats, n_dir=n_dir)

        # ----------------------------------------------------
        # 4) Solve OT / QP → get weights
        # ----------------------------------------------------
        out = solve_stochastic_subdom_coupling(
            S,
            solver="mosek",
            weight_method="primal",           # or "dual" / "row_dual"
            debias_rowcol=True,
            normalize=True,
        )

        weights = out["weights_np"]          # shape [R]

        # ----------------------------------------------------
        # 5) CLOSED-FORM UPDATE (placeholder)
        # ----------------------------------------------------
        self.update_stochastic_closed_form(
            rollout_feats, demo_feats, weights
        )
        # ----------------------------------------------------
        # 6) Return metrics
        # ----------------------------------------------------
        return {
            "train/zero_one_loss": float(np.mean([d["zero_one"] for d in demos])) if "zero_one" in demos[0] else None,
            "train/mean_subdom": float(S.mean()),
            "train/std_subdom": float(S.std()),
            "train/fairness": rollout_feats.mean(axis=0).tolist(),
            "train/directional_cost": dir_cost,
        }

    @torch.no_grad()
    def _sample_policy_params_normal(self, sigma: float):
        """
        Sample (w, b) ~ Normal(current_params, sigma).
        Assumes self.policy is a torch.nn.Linear with shape [1, d] (binary logit).
        """
        if sigma is None or sigma <= 0:
            # deterministic fallback
            w = self.policy.weight.detach().clone()
            b = self.policy.bias.detach().clone() if self.policy.bias is not None else None
            return w, b

        w_mu = self.policy.weight.detach()
        b_mu = self.policy.bias.detach() if self.policy.bias is not None else None

        w = w_mu + sigma * torch.randn_like(w_mu)
        b = None
        if b_mu is not None:
            b = b_mu + sigma * torch.randn_like(b_mu)
        return w, b

    @torch.no_grad()
    def _forward_with_params(self, X: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None):
        """
        Compute probs = sigmoid(X @ w^T + b).
        X: [n, d], w: [1, d], b: [1] or None
        Returns probs: [n]
        """
        logits = X.matmul(w.t()).squeeze(-1)  # [n]
        if b is not None:
            logits = logits + b.squeeze()
        probs = torch.sigmoid(logits)
        return probs

    @torch.no_grad()
    def _sample_actions_from_probs(
        self,
        probs: torch.Tensor,
        threshold: float | None = None,
    ):
        """
        If threshold is None -> sample Bernoulli(probs).
        If threshold is provided:
          - default: deterministic decision (probs >= threshold)
        Returns y_hat in {0,1} float tensor, shape [n]
        """
        if threshold is None:
            y_hat = torch.bernoulli(probs)
        else:
            y_hat = (probs >= threshold).float()
        return y_hat
    
    def train_one_epoch_stochastic_bayesian(
        self,
        demonstrator,
        batch_size: int = 1,
        n_dir: int = 20,
        decision_threshold: float | None = None,
        sigma: float = 0.01,
        stochastic_if_threshold: bool = False,
    ):
        """
        Stochastic subdominance training via *parameter noise*:
          1) For each rollout (one per demo here), sample policy params ~ N(m, sigma)
          2) Forward X with sampled params -> probs -> y_hat decisions
          3) Compute fairness features
          4) Build S and solve OT -> weights
          5) UPDATE (left empty)
          6) Return metrics
        """
        demos = demonstrator.train_demos
        R = len(demos)

        # ----------------------------------------------------
        # 1) Collect rollouts: sample params per rollout/demo
        # ----------------------------------------------------
        rollout_feats = []
        for d in demos:
            Xd = d["X"]
            # Use demonstrator targets as in your original code
            yd = demonstrator.get_targets(d)
            Ad = d["A"]

            # Ensure torch tensors live on same device as policy
            device = self.policy.weight.device
            Xd = Xd.to(device)
            yd = yd.to(device)
            Ad = Ad.to(device) if torch.is_tensor(Ad) else Ad  # depends on your data

            # sample params and get decisions
            w_s, b_s = self._sample_policy_params_normal(sigma=sigma)
            probs = self._forward_with_params(Xd, w_s, b_s)
            y_hat = self._sample_actions_from_probs(probs, threshold=decision_threshold)
            # Compute fairness features for this demo (sample group)
            f_r = compute_fairness_features(Xd, yd, y_hat, Ad, self.metrics_list)
            rollout_feats.append(f_r)

        rollout_feats = torch.stack(rollout_feats, dim=0)  # [R, K]

        # demo fairness matrix
        demo_feats = np.stack([d["fairness_feats"] for d in demos])  # [D, K]

        # ----------------------------------------------------
        # 2) Compute subdominance matrix S[R,D]
        # ----------------------------------------------------
        S = compute_subdominance_matrix(
            rollout_feats,
            demo_feats,
            mode=self.subdom_mode,
            alpha=self.compute_alpha(),
            beta=self.compute_beta(),
        )

        # ----------------------------------------------------
        # 3) Compute directional costs (pre-OT)
        # ----------------------------------------------------
        dir_cost = compute_directional_cost(rollout_feats.detach().cpu().numpy(),demo_feats,n_dir=n_dir)

        # ----------------------------------------------------
        # 4) Solve OT / QP → get weights
        # ----------------------------------------------------
        out = solve_stochastic_subdom_coupling(
            S,
            solver="mosek",
            weight_method="primal",
            debias_rowcol=True,
            normalize=True,
        )
        weights = out["weights_np"]  # shape [R]

        # ----------------------------------------------------
        # 5) UPDATE (intentionally empty for now)
        # ----------------------------------------------------
        # e.g. later: self.update_stochastic_closed_form(rollout_feats, demo_feats, weights)
        pass

        # ----------------------------------------------------
        # 6) Return metrics
        # ----------------------------------------------------
        return {
            "train/zero_one_loss": float(np.mean([d["zero_one"] for d in demos])) if "zero_one" in demos[0] else None,
            "train/mean_subdom": float(S.mean()),
            "train/std_subdom": float(S.std()),
            "train/fairness": rollout_feats.mean(dim=0).detach().cpu().tolist(),
            "train/directional_cost": dir_cost,
            "train/sigma": float(sigma),
        }
    # ==============================================================
    # EVALUATION METHOD
    # ==============================================================
    def evaluate(self, demonstrator, y_domain="01"):
        """
        Vectorized evaluation over all evaluation demos.
        Returns metrics prefixed with 'eval/' for consistency.
        """
        demonstrator.to_torch(self.device)

        #  import ipdb;ipdb.set_trace()
        if not demonstrator.eval_demos:
            return {
                "eval/zero_one_loss": None,
                "eval/mean_subdom": None,
                "eval/std_subdom": None,
                "eval/fairness": None,
            }

        # ---- Precompute reference demo fairness ----
        f_train = torch.as_tensor(
            np.stack([d["fairness_feats"] for d in demonstrator.train_demos]),
            dtype=torch.float32, device=self.device
        )

        # ---- Concatenate all eval demos ----
        Xe = torch.cat([d["X"] for d in demonstrator.eval_demos], dim=0)
        ye = torch.cat([d["y"] for d in demonstrator.eval_demos], dim=0)
        Ae = torch.cat([d["A"] for d in demonstrator.eval_demos], dim=0)

        with torch.no_grad():
            logits = self.policy(Xe).squeeze(-1)
            y_hat = torch.sigmoid(logits)
            ye = ye.view_as(y_hat)
            zero_one_loss = self.zero_one_loss(y_hat, ye)

            # Compute fairness features for eval data
            f_eval = compute_fairness_features(Xe, ye, y_hat, Ae, metrics=self.metrics_list)

            # Subdominance relative to training demos
            subdom_out = subdominance_loss_from_features(
                rollout_feats=f_eval[None, :],
                demo_feats=f_train,
                mode=self.subdom_mode,
                agg=self.subdom_agg,
                alpha=self.alpha,
                beta=self.beta,
                reduction="none",
            )
            S = subdom_out["S"].squeeze(0)
            mean_subdom = float(S.mean().item())
            std_subdom = float(S.std().item())

        zero_one_val = float(zero_one_loss.item()) if zero_one_loss is not None else np.nan
        return {
            "eval/zero_one_loss": zero_one_val,
            "eval/mean_subdom": float(mean_subdom) if mean_subdom is not None else np.nan,
            "eval/std_subdom": float(std_subdom) if std_subdom is not None else np.nan,
            "eval/fairness": f_eval.detach().cpu().numpy().tolist() if f_eval is not None else [],
        }
