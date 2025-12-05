import torch
import torch.nn as nn
import numpy as np

from stochastic_superhuman_fairness.core.models.base_model import BaseModel
from stochastic_superhuman_fairness.core.models.utils import (
    phi_features,
    phi_mean,
    phi_mean_per_demo,
)
from stochastic_superhuman_fairness.core.fairness.subdominance import (
    subdominance_loss_from_features,
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
        meta = demonstrator.dataset["metadata"]
        n_features = meta["n_features"]

        self.policy = nn.Linear(n_features, 1, bias=True)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = cfg.get("lr", 1e-3)
        #  import ipdb;ipdb.set_trace()
        self.device = cfg.get("device", "cpu")
        self.policy.to(self.device)

        # --- subdominance configuration ---
        self.subdom_mode = cfg.get("subdom_mode", "absolute")
        self.subdom_agg = cfg.get("subdom_agg", "mean")
        self.subdom_weight_mode = cfg.get("subdom_weight_mode", "softmax")  # or "linear"
        self.alpha = cfg.get("alpha")
        self.beta = cfg.get("beta")

        self.metrics_list = self._resolve_metrics(cfg, demonstrator)
    # ----------------------------------------------------------
    def parameters(self):
        return self.policy.parameters()
    # ----------------------------------------------------------
    def forward(self, X):
        return self.policy(X)
    # ----------------------------------------------------------
    def compute_phi(self, X, y, add_bias=True, y_domain="01"):
        return phi_features(X, y, add_bias=add_bias, y_domain=y_domain)
    # ----------------------------------------------------------
    def phi_mean(self, X, y, add_bias=True, y_domain="01"):
        return phi_mean(X, y, add_bias=add_bias, y_domain=y_domain)
    # ----------------------------------------------------------
    def phi_mean_per_demo(self, demos, add_bias=True, y_domain="01"):
        return super().phi_mean_per_demo(demos, add_bias=add_bias, y_domain=y_domain)
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
    def train_one_epoch_old(self, demonstrator, y_domain="01"):
        """
        Train for one epoch using stored demo fairness features.
        Each batch is a list of demo samples from Demonstrator.iter_batches().
        """
        batch_size = self.cfg.get("batch_size", 1)
        total_loss, total_subdom = 0.0, 0.0
        batch_count = 0

        # ---- Use precomputed demo fairness features ----
        f_d = torch.as_tensor(
            np.stack([d["fairness_feats"] for d in demonstrator.train_demos]),
            dtype=torch.float32,
            device=self.device,
        )  # [D, K]
        f_d_mean = f_d.mean(dim=0, keepdim=True)  # [1, K]

        # ---- iterate through demo batches ----
        for demo_batch in demonstrator.iter_batches(batch_size=batch_size, as_torch=True, device=self.device):
            batch_count += 1
            for d in demo_batch:
                Xb, yb, Ab = d["X"], d["y"], d["A"]

                logits = self.policy(Xb).squeeze(-1)
                y_hat = torch.sigmoid(logits)
                yb = yb.view_as(y_hat)
                zero_one_loss = self.zero_one_loss(y_hat, yb)

                # fairness for rollout
                f_r = (
                    torch.as_tensor(d["fairness_feats"], dtype=torch.float32, device=self.device)
                    if "fairness_feats" in d
                    else compute_fairness_features(Xb, yb, y_hat, Ab, metrics=self.metrics_list)
                )

                # subdominance vs demo fairness
                subdom_out = subdominance_loss_from_features(
                    rollout_feats=f_r[None, :],
                    demo_feats=f_d,
                    mode=self.subdom_mode,
                    agg=self.subdom_agg,
                    alpha=self.alpha,
                    beta=self.beta,
                    reduction="none",
                )
                S = subdom_out["S"].squeeze(0)
                w = S.mean()
                total_subdom += float(w.item())

                # φ difference weighted by subdominance
                phi_r_mean = self.compute_phi(Xb, yhat, add_bias=True, y_domain=y_domain).mean(dim=0)
                phi_d_means = self.phi_mean_per_demo(demonstrator.train_demos, add_bias=True, y_domain=y_domain)
                phi_d_means = torch.as_tensor(phi_d_means, dtype=torch.float32, device=self.device)
                phi_d_mean = phi_d_means.mean(dim=0)

                g = w * (phi_r_mean - phi_d_mean)
                assert g.numel() == self.policy.in_features + 1

                g_w, g_b = g[:-1], g[-1].unsqueeze(0)
                self.policy.weight.data -= self.lr * g_w
                self.policy.bias.data -= self.lr * g_b

                total_loss += float(zero_one_loss.item())

        # ---- training averages ----
        avg_loss = total_loss / max(batch_count, 1)
        avg_subdom = total_subdom / max(batch_count, 1)

        # ---- optional evaluation ----
        eval_demo = demonstrator.eval_demos[0] if demonstrator.eval_demos else None
        if eval_demo is not None:
            Xe = torch.as_tensor(eval_demo["X"], dtype=torch.float32, device=self.device)
            ye = torch.as_tensor(eval_demo["y"], dtype=torch.float32, device=self.device)
            Ae = torch.as_tensor(eval_demo["A"], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                y_hat_eval = torch.sigmoid(self.policy(Xe).squeeze(-1))
                fair_eval = compute_fairness_features(Xe, ye, y_hat_eval, Ae, metrics=self.metrics_list)
                subdom_eval = subdominance_loss_from_features(
                    rollout_feats=fair_eval[None, :],
                    demo_feats=f_d,
                    mode=self.subdom_mode,
                    agg=self.subdom_agg,
                    alpha=self.alpha,
                    beta=self.beta,
                    reduction="none",
                )
        else:
            fair_eval, subdom_eval = None, None

        return {
            "zero_one_loss/train": avg_loss,
            "mean_subdom/train": avg_subdom,
            "fairness/train": f_r.detach().cpu().numpy().tolist(),
            "zero_one_loss/eval": float(subdom_eval["S"].mean().item()) if eval_demo else None,
            "fairness/eval": fair_eval.detach().cpu().numpy().tolist() if fair_eval is not None else None,
        }
    # ----------------------------------------------------------
    def train_one_epoch(self, demonstrator, y_domain="01", shuffle_batch_each_epoch: bool = True):
        """
        Vectorized epoch: process demo batches together efficiently.
        Uses precomputed fairness features from the demonstrator.
        Returns training metrics with 'train/' prefix.
        """
        demonstrator.to_torch(self.device)
        batch_size = self.cfg.get("batch_size", 1)
        total_loss, total_subdom, total_subdom_std = 0.0, 0.0, 0.0
        batch_count = 0

        # ---- Precompute demo fairness once ----
        f_d = torch.as_tensor(
            np.stack([d["fairness_feats"] for d in demonstrator.train_demos]),
            dtype=torch.float32,
            device=self.device,
        )

        # ---- Training loop ----
        for demo_batch in demonstrator.iter_batches(batch_size=batch_size, shuffle = shuffle_batch_each_epoch):
            batch_count += 1

            # Concatenate all demos in this batch
            #  import ipdb;ipdb.set_trace()
            Xb = torch.cat([d["X"] for d in demo_batch], dim=0)
            yb = torch.cat([d["y"] for d in demo_batch], dim=0)
            Ab = torch.cat([d["A"] for d in demo_batch], dim=0)

            # Forward pass
            #  import ipdb;ipdb.set_trace()
            logits = self.policy(Xb).squeeze(-1)
            y_hat = torch.sigmoid(logits)
            yb = yb.view_as(y_hat)
            zero_one_loss = self.zero_one_loss(y_hat, yb)

            # Fairness + subdominance
            f_r = compute_fairness_features(Xb, yb, y_hat, Ab, metrics=self.metrics_list)
            subdom_out = subdominance_loss_from_features(
                rollout_feats=f_r[None, :],
                demo_feats=f_d,
                mode=self.subdom_mode,
                agg=self.subdom_agg,
                alpha=self.alpha,
                beta=self.beta,
                reduction="none",
            )
            S = subdom_out["S"].squeeze(0)
            w = S.mean()
            total_subdom += float(w.item())
            total_subdom_std += float(S.std().item())

            # φ feature difference weighted by subdominance
            #  import ipdb;ipdb.set_trace()
            phi_r_mean = self.compute_phi(Xb, yb, add_bias=True, y_domain=y_domain).mean(dim=0)
            phi_d_means = self.phi_mean_per_demo(demonstrator.train_demos, add_bias=True, y_domain=y_domain)
            phi_d_means = torch.as_tensor(phi_d_means, dtype=torch.float32, device=self.device)
            phi_d_mean = phi_d_means.mean(dim=0)
            g = w * (phi_r_mean - phi_d_mean)

            # Gradient update
            assert g.numel() == self.policy.in_features + 1
            g_w, g_b = g[:-1], g[-1].unsqueeze(0)
            #  print(f"[DEBUG] w={w.item():.4f},  |phi_r_mean|={phi_r_mean.norm():.4f},  "
            #    f"|phi_d_mean|={phi_d_mean.norm():.4f},  |g|={g.norm():.4f}")

            self.policy.weight.data -= self.lr * g_w
            self.policy.bias.data  -= self.lr * g_b

            total_loss += float(zero_one_loss.item())

        # ---- Averages ----
        avg_loss = total_loss / max(batch_count, 1)
        avg_subdom = total_subdom / max(batch_count, 1)
        std_subdom = total_subdom_std / max(batch_count, 1)

        return {
            "train/zero_one_loss": avg_loss,
            "train/mean_subdom": avg_subdom,
            "train/std_subdom": std_subdom,
            "train/fairness": f_r.detach().cpu().numpy().tolist(),
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
