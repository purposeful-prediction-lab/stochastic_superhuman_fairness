import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from stochastic_superhuman_fairness.core.rollout_utils import collect_rollouts, RolloutBatch
from stochastic_superhuman_fairness.core.fairness.subdominance import compute_alpha
from stochastic_superhuman_fairness.core.models.utils import (
    phi_features,
)
from core.utils import flatten_dict

class BaseModel(ABC, nn.Module):
    """
    Abstract base class for all fairness/subdominance learners.
    Every subclass must expose:
      - self.policy: classifier network
      - self.value (optional): subdominance/value estimator
      - self.opt: optimizer for trainable params
    """

    def __init__(self, cfg, demonstrator):
        super().__init__()
        self.cfg = cfg
        self.demo = demonstrator
        self.device = getattr(cfg, "device", "cpu")
        self.policy = None
        self.value = None
        self.opt = None
        # --- subdominance configuration ---
        scfg = cfg.get('subdominance', {})
        self.subdom_mode = scfg.get("mode", "absolute")
        self.subdom_type = scfg.get("type", "standard")
        self.subdom_agg = scfg.get("rollout_aggregate", "mean")
        self.subdom_weight_mode = scfg.get("weight_mode", "softmax")  # or "linear"
        self.alpha = scfg.get("alpha")
        self.beta = scfg.get("beta")
        # --- Get required fairness metrics ---
        self.metrics_list = self._resolve_metrics(cfg, demonstrator)

    # ----------------------------------------------------------

    @abstractmethod
    def forward(self, X):
        """Return raw model output (logits)."""
        pass

    @abstractmethod
    def train_step(self, X, y):
        """Perform one gradient step and return loss dict."""
        pass

    # ----------------------------------------------------------
    def train_one_epoch(self, demonstrator, batch_size: int =1, subdom_type: str = 'standard', **kwargs):
        """
        Selects standard vs stochastic subdominance training.
        Each subclass must implement:
            - train_one_epoch_standard(...)
            - train_one_epoch_stochastic(...)
        
        kwargs propagate to the underlying method (batch size, shuffle, etc.)
        """

        tkwargs = {'batch_size':batch_size,  **kwargs}
        if subdom_type == "stochastic":
            return self.train_one_epoch_stochastic(demonstrator, **tkwargs)

        elif subdom_type == "standard":
            return self.train_one_epoch_standard(demonstrator, **tkwargs)
        elif subdom_type == "bayesian":
            flattened_tkwargs = flatten_dict(kwargs, keep_path = False)
            return self.train_one_epoch_stochastic_bayesian(demonstrator, **flattened_tkwargs)
        else:
            raise ValueError(f"Unknown training mode '{self.cfg.subdom_mode}'")


    # ----------------------------------------------------------
    @torch.no_grad()
    def collect_rollouts(self, demonstrator, demos=None, decision_threshold=None):
        if demos is None:
            demos = demonstrator.train_demos
        return collect_rollouts(
            policy=self.policy,
            demonstrator=demonstrator,
            demos=demos,
            metrics_list=self.metrics_list,
            sample_actions_fn=self._sample_actions_from_probs,
            decision_threshold=decision_threshold,
            before_demo=None,
        )

    @torch.no_grad()
    def collect_eval_rollouts(self, demonstrator, demos=None, decision_threshold=0.5):
        if demos is None:
            demos = demonstrator.test_demos
        return collect_rollouts(
            policy=self.policy,
            demonstrator=demonstrator,
            demos=demos,
            metrics_list=self.metrics_list,
            sample_actions_fn=self._sample_actions_from_probs,
            decision_threshold=decision_threshold,
            before_demo=None,
            require_grad=False,
            detach_outputs=True,
        )
    # ----------------------------------------------------------
    def get_state_dict(self):
        """Return dict of policy/value weights for cross-phase transfer."""
        return {
            "policy": self.policy.state_dict() if self.policy is not None else None,
            "value": self.value.state_dict() if self.value is not None else None,
        }

    # ----------------------------------------------------------
    def update_value(self, X, y, A):
        """
        Optional: update value function to estimate subdominance.
        Default: no-op (for models that don’t have a value net).
        """
        return 0.0

    def _resolve_metrics(self, cfg, demonstrator):
        """
        Resolve the list of fairness metrics for this model.
        Priority:
          1. cfg.metrics.use if defined
          2. demonstrator.cfg.demonstrator.metrics if available
          3. default fallback
        Warns if mismatch between model and demonstrator.
        """
        default_metrics = ["D.DP", "D.EqOdds", "D.PRP", "D.Err"]

        # --- Get from model cfg if defined ---
        cfg_metrics = None
        if hasattr(cfg, "metrics"):
            if isinstance(cfg.metrics, dict) and "use" in cfg.metrics:
                cfg_metrics = cfg.metrics["use"]
            elif isinstance(cfg.metrics, (list, tuple)):
                cfg_metrics = list(cfg.metrics)

        # --- Fallback: demonstrator metrics ---
        demo_metrics = getattr(
            getattr(demonstrator.cfg, "demonstrator", {}),
            "metrics",
            None,
        )

        # --- Resolve final list ---
        if cfg_metrics:
            metrics_list = cfg_metrics
            if demo_metrics and set(metrics_list) != set(demo_metrics):
                print(
                    f"⚠️ Metric mismatch: model uses {metrics_list}, "
                    f"but demonstrator built with {demo_metrics}"
                )
        elif demo_metrics:
            metrics_list = demo_metrics
            print(f"ℹ️ Using metrics from demonstrator: {metrics_list}")
        else:
            metrics_list = default_metrics
            print(f"ℹ️ No metrics found; using default {metrics_list}")

        return metrics_list

    def compute_alpha(self, rollouts, demos, beta = None, 
                      mode: str = 'absolute', update_self_alpha: bool = True, alpha_max: float = 10.0, reduce: str = 'mean',
        ):
        """Placeholder – later: learn alpha per fairness dimension."""
        beta = self.beta if beta is None else beta
        alpha =  compute_alpha(rollouts, demos, beta, mode = mode, alpha_max = alpha_max, reduce = reduce)
        if update_self_alpha:
            self.alpha =  alpha
        return alpha

    def compute_beta(self):
        """Placeholder – later: learn beta per fairness dimension."""
        return None  # triggers default β = ones(K)

    def zero_one_loss(self, y_hat, y_true):
        preds = (y_hat > 0.5).float()
        return (preds != y_true).float().mean()

    def phi_mean_per_demo(
        self,
        demos,
        add_bias: bool = False,
        y_domain: str = "01",
        stochastic: bool = True,
    ):
        """
        Compute mean φ(x, ŷ) for each demo under the current model.

        Args:
            demos: list of demo dicts with 'observations' arrays.
            add_bias: whether to include bias term in φ(x, ŷ).
            y_domain: kept for compatibility, assumed '01'.
            stochastic: if True, sample ŷ ~ Bernoulli(p); otherwise use deterministic rounding.

        Returns:
            Tensor [N_demos, F+1] of mean φ(x, ŷ) per demo.
        """
        phi_list = []
        for d in demos:
            Xd = d["X"]
            Xd_t = torch.as_tensor(Xd, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                y_hat = self.policy(Xd_t).squeeze()
                y_hat = torch.clamp(y_hat, 0.0, 1.0)  # <-- critical safety clamp
                if torch.isnan(y_hat).any():
                    raise ValueError("NaN in model predictions during phi_mean_per_demo.")
                if stochastic:
                    y_sample = torch.bernoulli(y_hat)
                else:
                    y_sample = (y_hat > 0.5).float()
            phi_d = phi_features(Xd_t, y_sample, add_bias=add_bias, y_domain=y_domain)
            phi_list.append(phi_d.mean(dim=0))
        return torch.stack(phi_list, dim=0)

    def exp_phi(
            self,
            demos,
            add_bias: bool = False,
            y_domain: str = "01",
            stochastic: bool = True,
        ):
            """
            Compute mean φ(x, ŷ) for all demos under the current model.

            Args:
                demos: list of demo dicts with 'observations' arrays.
                add_bias: whether to include bias term in φ(x, ŷ).
                y_domain: kept for compatibility, assumed '01'.
                stochastic: if True, sample ŷ ~ Bernoulli(p); otherwise use deterministic rounding.

            Returns:
                Tensor [F] of mean φ(x, ŷ) for the batch.
            """
            return self.phi_mean_per_demo(demos, add_bias = add_bias, stochastic = stochastic).mean(axis=0)
