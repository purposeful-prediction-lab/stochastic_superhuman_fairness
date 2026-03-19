import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import math
from abc import ABC, abstractmethod
from stochastic_superhuman_fairness.core.rollout_utils import collect_rollouts, RolloutBatch
from stochastic_superhuman_fairness.core.fairness.subdominance import compute_alpha
from stochastic_superhuman_fairness.core.models.utils import (
    phi_features,
)
from stochastic_superhuman_fairness.core.fairness.subdominance import (
    subdominance_loss_from_features,
    compute_subdominance_matrix,
)
from stochastic_superhuman_fairness.core.fairness.fairness_metrics import compute_fairness_features, zero_one_loss
from core.utils import flatten_dict, sample_actions_from_policy

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
        self.device = cfg.get('device')
        self.policy = None
        self.value = None
        self.opt = None
        self.modeltype = str(cfg['algo'])

        self.model_cfg = cfg.get('model_cfg', {})
        # --- subdominance configuration ---
        scfg = cfg.get('subdominance', {})
        self.subdom_mode = scfg.get("mode", "absolute")
        self.subdom_type = scfg.get("type", "standard")
        self.subdom_agg = scfg.get("rollout_aggregate", "mean")
        self.subdom_weight_mode = scfg.get("weight_mode", "softmax")  # or "linear"
        self.alpha_updates = scfg.get("alpha_updates", "analytical")
        self.alpha = scfg.get("alpha", 1)
        self.beta = scfg.get("beta", 0)
        # --- Get required fairness metrics ---
        self.metrics_list = self._resolve_metrics(cfg, demonstrator)
        #  import ipdb;ipdb.set_trace()

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
    def get_policy(self):
        if hasattr(self, "policies") and self.policies is not None:
            policy = list(self.policies)
        else:
            policy = [self.policy]
        return policy

    # ----------------------------------------------------------

    def train_one_epoch(self, demonstrator, batch_size: int =1, subdom_type: str = 'standard', **kwargs):
        """
        Selects standard vs stochastic subdominance training.
        Each subclass must implement:
            - train_one_epoch_standard(...)
            - train_one_epoch_stochastic(...)
        
        kwargs propagate to the underlying method (batch size, shuffle, etc.)
        """

        t_function= kwargs['t_function']
        tkwargs = {'batch_size':batch_size, 'alpha_updates': self.alpha_updates, **kwargs}
        #  import ipdb;ipdb.set_trace()
        tkwargs = flatten_dict(tkwargs, keep_path = False, no_flatten_terms = ['loss_fn_kwargs'])
        return getattr(self, f'_train_one_epoch_{t_function}')(demonstrator, no_update = False, **tkwargs)

    # ----------------------------------------------------------
    def collect_training_rollouts(self, demonstrator, demos=None, decision_threshold: float=0.5, stochastic = True,
                                  use_demos_as_gtruth: bool = False):
        if demos is None:
            demos = demonstrator.train_demos
        return collect_rollouts(
            policy=self.get_policy(),
            demonstrator=demonstrator,
            demos=demos,
            metrics_list=self.metrics_list,
            sample_actions_fn=self.sample_actions,
            decision_threshold=decision_threshold,
            stochatic = stochastic,
            require_grad = True,
            detach_outputs = False,
            use_demos_as_gtruth = use_demos_as_gtruth,
        )
    @torch.no_grad()
    def collect_rollouts(self, demonstrator, demos=None, decision_threshold: float=0.5, stochastic = True,
                        use_demos_as_gtruth: bool = False
                         ):
        detach_outputs = require_grad
        if demos is None:
            demos = demonstrator.train_demos
        return collect_rollouts(
            policy=self.get_policy(),
            demonstrator=demonstrator,
            demos=demos,
            metrics_list=self.metrics_list,
            sample_actions_fn=self.sample_actions,
            decision_threshold=decision_threshold,
            stochatic = stochastic,
            require_grad = False,
            detach_outputs = True,
            use_demos_as_gtruth = use_demos_as_gtruth,
        )
    @torch.no_grad()
    def collect_eval_rollouts(self, demonstrator, demos=None,
                    shared_x: bool = False,
                    decision_threshold=0.5, n_rollouts: int = 10,
                    use_demos_as_gtruth: bool = False,
                    stochastic : bool = False):
        if demos is None:
            demos = demonstrator.test_demos
        return collect_rollouts(
            policy=self.get_policy(),
            demonstrator=demonstrator,
            demos=demos,
            shared_x = shared_x,
            metrics_list=self.metrics_list,
            n_rollouts = n_rollouts,
            decision_threshold=decision_threshold,
            require_grad=False,
            detach_outputs=True,
            stochastic = stochastic,
            #  sample_actions_fn=self.sample_actions, # <-- Use models sample fn
            use_demos_as_gtruth = use_demos_as_gtruth,
            return_logits = True,
        )
    # ----------------------------------------------------------
    def get_state_dict(self):
        """Return dict of policy/value weights for cross-phase transfer."""
        return {
            "policy": self.policy.state_dict() if self.get_policy() is not None else None,
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
        try:
            alpha =  compute_alpha(rollouts, demos, beta, mode = mode, alpha_max = alpha_max, reduce = reduce)
        except:
            import ipdb;ipdb.set_trace()
        if update_self_alpha:
            self.alpha =  alpha
        return alpha

    def compute_beta(self):
        """Placeholder – later: learn beta per fairness dimension."""
        return None  # triggers default β = ones(K)

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

    @torch.no_grad()
    def apply_S_temperature(self, S, beta: float = 1.0):
        """Use before OT: S_beta = beta * S."""
        return beta * S

    def sample_actions(
        self,
        X: torch.Tensor,
        decision_threshold: float = None,
        return_logits: bool = True,
        return_probs: bool = False,
        require_grad: bool = False
    ):
       # Genreal function from utils.py
       return sample_actions_from_policy(self.policy, X, decision_threshold = decision_threshold,
                             return_logits=return_logits, return_probs=return_probs,require_grad=require_grad)

    def evaluate(self, demonstrator, decision_threshold: float = 0.5, per_policy_rollouts : int = 20,
                    stochastic: bool = True):
        """
        Evaluate either a single policy (self.policy) or multiple policies (self.policies).
        Returns aggregate metrics + per-policy metrics under key "per_policy".
        """
        demonstrator.to_torch(self.device)

        if not demonstrator.eval_demos:
            return {
                "eval/zero_one_loss": None,
                "eval/mean_subdom": None,
                "eval/std_subdom": None,
                "eval/fairness": None,
                "per_policy": [],
            }

        # ---- Precompute reference demo fairness (train demos) ----
        demo_feats = demonstrator.eval_demo_feats
        f_train = torch.as_tensor(
            np.stack([d["fairness_feats"] for d in demonstrator.eval_demos]),
            dtype=torch.float32,
            device=self.device,
        )

        # ---- Concatenate all eval demos ----
        Xe = torch.cat([d["X"] for d in demonstrator.eval_demos], dim=0)
        ye = torch.cat([d["y"] for d in demonstrator.eval_demos], dim=0)
        Ae = torch.cat([d["A"] for d in demonstrator.eval_demos], dim=0)

        # ---- collect policies uniformly ----
        policies = self.get_policy()
        #  import ipdb;ipdb.set_trace()
        with torch.no_grad():
            #  rb = self.collect_eval_rollouts(
            #                  demonstrator,
            #                  demos=demonstrator.eval_demos,
            #                  shared_x = demonstrator.shared_x,
            #                  decision_threshold = decision_threshold,
            #                  n_rollouts = per_policy_rollouts,
            #                  stochastic=stochastic,
            #                  use_demos_as_gtruth = False,
            #                  )
            #  rollout_feats = rb.feats.detach().cpu().numpy()
            #  import ipdb;ipdb.set_trace()
            #  probs = torch.stack(rb.y_hat) # (P x Per_rollout, N)
             
            #  feats = rb.feats
            # logits: (P, N)
            logits = torch.stack([p(Xe).squeeze(-1) for p in policies], dim=0)
            probs = torch.sigmoid(logits)                      # (P, N)
            ye_b = ye.view(1, -1).expand_as(probs)             # (P, N)
            Ae_b = Ae.view(-1).squeeze(-1)                     # (N,)

            # zero-one per policy (thresholded)
            y_pred = (probs >= decision_threshold).float()
            z1 = (y_pred != ye_b).float().mean(dim=1)          # (P,)

            # fairness per policy (loop over P; metrics need y_true,y_pred,a)
            f_list = []
            for i in range(probs.shape[0]):
                f_list.append(compute_fairness_features(ye_b[i], probs[i], Ae_b, metrics=self.metrics_list))
            f_eval = torch.stack(f_list, dim=0)                # (P, Kfeat)

            # subdominance per policy vs train demos
            subdom_out = subdominance_loss_from_features(
                rollout_feats=f_eval,
                #  demo_feats=f_train,
                demo_feats=demo_feats,
                mode=self.subdom_mode,
                agg=self.subdom_agg,
                alpha=self.alpha,
                beta=self.beta,
                reduction="none",
            )
            S = subdom_out["S"]                                # (P, num_train_demos)
            mean_subdom_per = S.mean(dim=1)                    # (P,)
            std_subdom_per = S.std(dim=1)                      # (P,)

            # aggregate over policies
            mean_fair = f_eval.mean(dim=0)
            agg_S = S.mean(dim=0)
            agg_mean_subdom = float(agg_S.mean().item())
            agg_std_subdom = float(agg_S.std().item())
            agg_zero_one = float(z1.mean().item())

        per_policy = []
        for i in range(len(policies)):
            per_policy.append({
                "eval/zero_one_loss": float(z1[i].item()),
                "eval/mean_subdom": float(mean_subdom_per[i].item()),
                "eval/std_subdom": float(std_subdom_per[i].item()),
                "eval/fairness": f_eval[i].detach().cpu().numpy().tolist(),
            })

        return {
            "eval/zero_one_loss": agg_zero_one,
            "eval/mean_subdom": agg_mean_subdom,
            "eval/std_subdom": agg_std_subdom,
            "eval/fairness": mean_fair.detach().cpu().numpy().tolist(),
            "per_policy": per_policy,
        }
    # --------------------------------------------------------------------------------------------
    def _clone_policy(self) -> nn.Module:
        # Create a new instance with the same architecture and copy weights.
        # This is the most robust approach if your policy isn't a simple Linear.
        pol = copy.deepcopy(self.policy)
        return pol

    def init_policy_mixture(
        self,
        source_policies,
        mixtures,
        n_total_policies: int,
    ):
        """
        Build a ModuleList containing a mixture of cloned source policies.

        Args:
            source_policies: list of nn.Module
            mixtures:
                list/array of either:
                  - probabilities
                  - integer occurrences
            n_total_policies: total number of output policies

        Returns:
            nn.ModuleList
        """
        if mixtures is None:
            raise ValueError("mixtures must be provided.")

        if not isinstance(source_policies, (list, tuple)) or len(source_policies) == 0:
            raise ValueError("source_policies must be a non-empty list or tuple.")

        m = len(source_policies)
        if n_total_policies <= 0:
            raise ValueError("n_total_policies must be > 0.")

        mix = np.asarray(mixtures).reshape(-1)
        if len(mix) > m:
            raise ValueError("mixtures cannot be longer than source_policies.")

        # pad if needed
        if len(mix) < m:
            pad_len = m - len(mix)

            # counts case: fill with 1
            if np.all(np.equal(np.mod(mix, 1), 0)):
                mix = np.concatenate([mix, np.ones(pad_len, dtype=mix.dtype)])
            else:
                # probs case: fill later from leftover mass
                mix = np.concatenate([mix, np.full(pad_len, np.nan)])

        # detect counts vs probs
        provided = ~np.isnan(mix.astype(float))
        provided_vals = mix[provided].astype(float)

        is_counts = np.all(provided_vals >= 0) and np.all(np.isclose(provided_vals, np.round(provided_vals)))

        if is_counts:
            counts = mix.astype(int)

        else:
            if np.any(provided_vals < 0):
                raise ValueError("Probability mixtures must be non-negative.")

            probs = np.zeros(m, dtype=float)
            probs[provided] = provided_vals

            p_sum = probs.sum()
            if p_sum > 1.0 + 1e-12:
                raise ValueError(f"Probability mixtures sum to {p_sum:.6f} > 1.")

            missing = np.isnan(mix.astype(float))
            leftover = 1.0 - p_sum

            if missing.any():
                probs[missing] = leftover / missing.sum()
            elif leftover > 1e-12:
                print(f"\nProbability mixture of start policies is {p_sum} <= 1.0. Dividing remaning {leftover:.4f} uniformly to source policies.\n")
                probs += leftover / m

            probs = probs / probs.sum()

            raw = probs * n_total_policies
            counts = np.floor(raw).astype(int)

            remainder = n_total_policies - counts.sum()
            if remainder > 0:
                frac = raw - counts
                order = np.argsort(-frac)
                counts[order[:remainder]] += 1

        # adjust counts to exact total
        total = counts.sum()

        if total > n_total_policies:
            extra = total - n_total_policies
            order = np.argsort(-counts)
            for idx in order:
                take = min(extra, counts[idx])
                counts[idx] -= take
                extra -= take
                if extra == 0:
                    break

        elif total < n_total_policies:
            deficit = n_total_policies - total
            order = np.argsort(-counts)
            print(f'\nSum of all source policies is {total} <= {n_total_policies} required. Uniformly spreading {deficit} deficit to all source policies\n')
            for i in range(deficit):
                counts[order[i % m]] += 1

        # build ModuleList
        out = []
        for pol, c in zip(source_policies, counts):
            for _ in range(int(c)):
                out.append(copy.deepcopy(pol))

        if len(out) != n_total_policies:
            raise RuntimeError(f"Internal error: created {len(out)} policies, expected {n_total_policies}.")

        return nn.ModuleList(out)
