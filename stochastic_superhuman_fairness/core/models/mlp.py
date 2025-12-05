import torch
import torch.nn as nn
import torch.optim as optim
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
from stochastic_superhuman_fairness.core.fairness.fairness_metrics import compute_fairness_features
#  from stochastic_superhuman_fairness.core.fairness.compute_fairness_utils import compute_fairness_features


class MLPModel(BaseModel):
    """
    Simple MLP classifier + optional value network for subdominance.
    """

    def __init__(self, cfg, demonstrator):
        super().__init__(cfg, demonstrator)
        meta = demonstrator.dataset["metadata"]
        input_dim = meta["n_features"]
        hidden = cfg.get("hidden_size", 128)
        dropout = cfg.get("dropout", 0.1)
        lr = cfg.get("lr", 1e-3)

        # --- policy network ---
        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # --- optional value network ---
        self.value = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.opt = optim.Adam(self.parameters(), lr=lr)
    # ----------------------------------------------------------
    def forward(self, X):
        return self.policy(X)

    # ----------------------------------------------------------

    def train_step(self, X, y):
        """Standard BCE + optional subdominance gradient signal."""
        self.opt.zero_grad(set_to_none=True)
        logits = self.policy(X)
        loss = self.loss_fn(logits.squeeze(), y)
        loss.backward()
        self.opt.step()
        return {"loss": float(loss.item())}

    # ----------------------------------------------------------

    def train_one_epoch(self, demonstrator, shuffle_batch_each_epoch=True):
        device = self.device
        demonstrator.to_torch(device)

        batch_size = self.cfg.get("batch_size", 1)
        demos = demonstrator.train_demos

        total_zero_one = total_subdom = total_std = 0.0
        batch_count = 0
        subdoms = []

        # Precompute demo fairness (for all demos) once
        f_d = torch.as_tensor(
            np.stack([d["fairness_feats"] for d in demos]),
            device=device, dtype=torch.float32
        )

        # ---- Batch iterator (outer loop) ----
        for demo_batch in demonstrator.iter_batches(batch_size=batch_size,
                                                    shuffle=shuffle_batch_each_epoch):
            batch_count += 1
            batch_loss = 0.0

            # ---- Inner loop: one rollout per demo ----
            for d in demo_batch:

                X = d["X"]
                y = d["y"]
                A = d["A"]

                logits = self.policy(X).squeeze(-1)
                y_hat = torch.sigmoid(logits)

                # zero-one loss (tracking only)
                total_zero_one += float(self.zero_one_loss(y_hat, y))

                # fairness per-demo
                f_r = compute_fairness_features(y, y_hat, A, metrics = self.metrics_list)
                f_r = f_r.to(device)

                # subdominance per-demo vs all demos
                out = subdominance_loss_from_features(
                    rollout_feats=f_r.unsqueeze(0),
                    demo_feats=f_d,
                    mode=self.subdom_mode,
                    agg=self.subdom_agg,
                    alpha=self.alpha,
                    beta=self.beta,
                    reduction="mean",
                )

                #  import ipdb;ipdb.set_trace()
                #  S_row = out["S"][0]           # [D]
                w = out['loss']
                #  w = aggregate_subdominance(S_row, self.subdom_agg)   # scalar
                total_subdom += float(w)
                subdoms.append(w.item())
                #  total_std    += float(S_row.std())

                batch_loss = batch_loss + w   # accumulate rollouts inside batch

            # ---- END inner loop → update ONCE per batch ----
            self.opt.zero_grad()
            batch_loss.backward()
            self.opt.step()

        # ---- Averages ----
        R = len(demos)
        return {
            "train/zero_one_loss": total_zero_one / R,
            "train/mean_subdom":   total_subdom / R,
            "train/std_subdom":    np.array(subdoms).std()/ R,
            "train/fairness":      f_r.detach().cpu().tolist(),
        }
    # ----------------------------------------------------------
    def update_value(self, X, y, A):
        """Train value network to approximate subdominance signal."""
        if self.value is None:
            return 0.0
        self.opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            logits = torch.sigmoid(self.policy(X)).detach()
        pred_val = self.value(X).squeeze()
        target_val = (logits - y).abs()  # simple placeholder target
        v_loss = torch.mean((pred_val - target_val) ** 2)
        v_loss.backward()
        self.opt.step()
        return v_loss.item()

    @torch.no_grad()
    def evaluate(self, demonstrator):
        device = self.device
        demonstrator.to_torch(device)
        batch_size = self.cfg.get("batch_size", 1)

        demos = demonstrator.eval_demos
        if demos is None or len(demos) == 0:
            return {
                "eval/zero_one_loss": None,
                "eval/mean_subdom":   None,
                "eval/std_subdom":    None,
                "eval/fairness":      None,
            }

        # Precompute demo fairness once (same as train)
        f_d = torch.as_tensor(
            np.stack([d["fairness_feats"] for d in demos]),
            device=device, dtype=torch.float32
        )

        total_zero_one = 0.0
        total_subdom = 0.0
        total_std = 0.0
        subdoms = []
        R = len(demos)

        # ---- process demo batches ----
        for demo_batch in demonstrator.iter_batches(batch_size = batch_size, source = 'eval'):

            # per-rollout loop
            for d in demo_batch:

                X = d["X"]
                y = d["y"]
                A = d["A"]

                # forward
                logits = self.policy(X).squeeze(-1)
                y_hat  = torch.sigmoid(logits)

                # zero-one
                total_zero_one += float(self.zero_one_loss(y_hat, y))

                # fairness per rollout
                f_r = compute_fairness_features(y, y_hat, A, metrics = self.metrics_list)
                f_r = f_r.to(device)

                # subdominance per rollout vs all demos
                out = subdominance_loss_from_features(
                    rollout_feats=f_r.unsqueeze(0),
                    demo_feats=f_d,
                    mode=self.subdom_mode,
                    agg=self.subdom_agg,
                    alpha=self.alpha,
                    beta=self.beta,
                    reduction="mean",
                )

                w = out['loss']

                total_subdom += float(w)
                subdoms.append(w.item())

        return {
            "eval/zero_one_loss": total_zero_one / R,
            "eval/mean_subdom":   total_subdom / R,
            "eval/std_subdom":    np.array(subdoms).std() / R,
            "eval/fairness":      f_r.detach().cpu().tolist(),
        }
