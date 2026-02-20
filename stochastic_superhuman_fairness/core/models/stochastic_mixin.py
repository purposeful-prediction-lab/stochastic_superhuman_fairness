import warnings
from typing import Dict, List, Optional, Tuple, Literal

import torch


DistMode = Literal["per_param_diag", "full_param_mvn"]


def _cov_bytes(d: int, dtype: torch.dtype) -> int:
    return d * d * torch.tensor([], dtype=dtype).element_size()


def _format_bytes(n: int) -> str:
    x = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024.0:
            return f"{x:.1f}{unit}" if unit != "B" else f"{int(x)}{unit}"
        x /= 1024.0
    return f"{x:.1f}PB"


class StochasticParamDistMixin:
    """
    Distribution over policy parameters with a uniform interface:
      - init_dist(mode, **cfg)
      - sample_dist(mode, **cfg) -> (theta_dict, aux)
      - update_dist(mode, **cfg)

    Implemented here:
      1) per_param_diag  (factorized Gaussian over parameters)
      2) full_param_mvn  (full covariance MVN over all parameters; feasible only for small d)

    Ready to extend with:
      3) layerwise neuron MVN
      4) global neuron MVN
    """

    # -------------------------
    # Common helpers
    # -------------------------
    @torch.no_grad()
    def apply_ot_temperature(self, S, beta: float = 1.0):
        """Use before OT: S_beta = beta * S."""
        return beta * S

    @torch.no_grad()
    def _param_names(self) -> List[str]:
        return [n for n, _ in self.policy.named_parameters()]

    @torch.no_grad()
    def _theta_dict_from_current_policy(self) -> Dict[str, torch.Tensor]:
        return {n: p.detach().clone() for n, p in self.policy.named_parameters()}

    @torch.no_grad()
    def load_params_into_policy(self, theta: Dict[str, torch.Tensor]):
        for n, p in self.policy.named_parameters():
            p.copy_(theta[n])

    @torch.no_grad()
    def _flatten_theta_dict(self, theta: Dict[str, torch.Tensor], names: List[str]) -> torch.Tensor:
        return torch.cat([theta[n].reshape(-1) for n in names], dim=0)

    @torch.no_grad()
    def _unflatten_to_theta_dict(self, vec: torch.Tensor, template: Dict[str, torch.Tensor], names: List[str]) -> Dict[str, torch.Tensor]:
        out = {}
        i = 0
        for n in names:
            t = template[n]
            num = t.numel()
            out[n] = vec[i:i+num].view_as(t)
            i += num
        return out

    # ============================================================
    # Public interface (dispatch)
    # ============================================================
    @torch.no_grad()
    def init_dist(self, mode: DistMode = 'per_param_diag', **cfg):
        self.dist_initialized = 1
        if mode == "per_param_diag":
            return self._init_per_param_diag(**cfg)
        if mode == "full_param_mvn":
            return self._init_full_param_mvn(**cfg)
        raise ValueError(f"Unknown mode: {mode}")

    @torch.no_grad()
    def sample_dist(self, mode: DistMode, **cfg) -> Tuple[Dict[str, torch.Tensor], dict]:
        #  import ipdb;ipdb.set_trace()
        if mode == "per_param_diag":
            return self._sample_per_param_diag(**cfg)
        if mode == "full_param_mvn":
            return self._sample_full_param_mvn(**cfg)
        raise ValueError(f"Unknown mode: {mode}")

    @torch.no_grad()
    def update_dist(self, mode: DistMode, **cfg):
        if mode == "per_param_diag":
            return self._update_per_param_diag(**cfg)
        if mode == "full_param_mvn":
            return self._update_full_param_mvn(**cfg)
        raise ValueError(f"Unknown mode: {mode}")
    @torch.no_grad()
    def get_dist_state(self) -> dict:
        state = {}
        # per-param diag
        if hasattr(self, "per_param_mean") and hasattr(self, "per_param_var"):
            state["mode"] = "per_param_diag"
            state["per_param_mean"] = {k: v.detach().cpu() for k, v in self.per_param_mean.items()}
            state["per_param_var"]  = {k: v.detach().cpu() for k, v in self.per_param_var.items()}

        # full mvn
        if hasattr(self, "full_mu") and hasattr(self, "full_Sigma"):
            state["mode"] = "full_param_mvn"
            state["full_names"] = list(getattr(self, "full_names", []))
            state["full_mu"] = self.full_mu.detach().cpu()
            state["full_Sigma"] = self.full_Sigma.detach().cpu()

        #  import ipdb;ipdb.set_trace()
        state["full_template"] = {k: v.detach().cpu() for k, v in self.full_template.items()}
        return state

    @torch.no_grad()
    def load_dist_state(self, state: dict, device: str | torch.device = "cpu"):
        if not state:
            return
        mode = state.get("mode", None)

        self.full_template = {k: v.to(device) for k, v in state["full_template"].items()}
        if mode == "per_param_diag":
            self.per_param_mean = {k: v.to(device) for k, v in state["per_param_mean"].items()}
            self.per_param_var  = {k: v.to(device) for k, v in state["per_param_var"].items()}
            self.dist_initialized = 1
            return

        if mode == "full_param_mvn":
            self.full_names = list(state.get("full_names", []))
            self.full_mu = state["full_mu"].to(device)
            self.full_Sigma = state["full_Sigma"].to(device)
            self.dist_initialized = 1
            return
    # ============================================================
    # 1) Naive per-parameter diagonal Gaussian
    #    theta_i ~ N(mu_i, var_i) independently
    # ============================================================
    @torch.no_grad()
    def _init_per_param_diag(
        self,
        init_var: float = 1e-4,
        relative: bool = False,   # if True, scale init_var by param magnitude
        eps: float = 1e-12,
    ):
        self.per_param_mean: Dict[str, torch.Tensor] = {}
        self.per_param_var: Dict[str, torch.Tensor] = {}
        for n, p in self.policy.named_parameters():
            mu = p.detach().clone()
            if relative:
                scale = mu.abs().mean().clamp_min(eps)
                v = float((scale * init_var) ** 2)
            else:
                v = float(init_var)
            self.per_param_mean[n] = mu
            self.per_param_var[n] = torch.full_like(mu, v)

    @torch.no_grad()
    def _sample_per_param_diag(self, store_eps: bool = True):
        theta: Dict[str, torch.Tensor] = {}
        eps_dict: Optional[Dict[str, torch.Tensor]] = {} if store_eps else None
        for n, p in self.policy.named_parameters():
            e = torch.randn_like(p)
            std = torch.sqrt(self.per_param_var[n])
            theta[n] = self.per_param_mean[n] + std * e
            if store_eps:
                eps_dict[n] = e
        return theta, {"eps": eps_dict} if store_eps else {}

    @torch.no_grad()
    def _update_per_param_diag(
        self,
        eps_list: List[Dict[str, torch.Tensor]],  # per rollout eps dict
        weights: torch.Tensor,                    # [R] from OT (row sums)
        ema: float = 0.2,
        var_floor: float = 1e-8,
        max_mean_delta: Optional[float] = None,
        var_ratio_clip: Optional[Tuple[float, float]] = (0.5, 2.0),
    ):
        device = next(self.policy.parameters()).device
        w = weights.to(device)
        w = w / (w.sum() + 1e-12)

        for n, _p in self.policy.named_parameters():
            # E: [R, *shape]
            E = torch.stack([e[n].to(device) for e in eps_list], dim=0)
            wv = w.view(-1, *([1] * (E.dim() - 1)))
            eps_bar = (wv * E).sum(dim=0)

            mu_old = self.per_param_mean[n]
            var_old = self.per_param_var[n]
            std_old = torch.sqrt(var_old)

            # target moments (moment matching)
            mu_tgt = mu_old + std_old * eps_bar
            centered = E - eps_bar
            var_scale = (wv * centered.pow(2)).sum(dim=0)
            var_tgt = (var_old * var_scale).clamp_min(var_floor)

            # EMA
            mu_upd = (1 - ema) * mu_old + ema * mu_tgt
            var_upd = (1 - ema) * var_old + ema * var_tgt

            # trust region on mean
            if max_mean_delta is not None:
                delta = (mu_upd - mu_old).clamp(-max_mean_delta, max_mean_delta)
                mu_upd = mu_old + delta

            # trust region on var (multiplicative)
            if var_ratio_clip is not None:
                lo, hi = var_ratio_clip
                ratio = (var_upd / (var_old + 1e-12)).clamp(lo, hi)
                var_upd = (var_old * ratio).clamp_min(var_floor)
            else:
                var_upd = var_upd.clamp_min(var_floor)

            self.per_param_mean[n] = mu_upd
            self.per_param_var[n] = var_upd

    # ============================================================
    # 2) Full covariance MVN over ALL parameters (small d only)
    #    theta ~ N(mu, Sigma)
    # ============================================================
    @torch.no_grad()
    def _init_full_param_mvn(
        self,
        init_var: float = 1e-4,
        cov_dtype: torch.dtype = torch.float32,
        max_cov_bytes: int = 1_073_741_824,   # ~1GiB
        allow_large_cov: bool = False,
    ):
        self.full_names = self._param_names()
        template = self._theta_dict_from_current_policy()
        theta0 = self._flatten_theta_dict(template, self.full_names).to(dtype=cov_dtype)

        d = theta0.numel()
        need = _cov_bytes(d, cov_dtype)
        if need > max_cov_bytes and not allow_large_cov:
            msg = (
                f"Full covariance would allocate ~{_format_bytes(need)} "
                f"for a {d}x{d} matrix (dtype={cov_dtype}). "
                f"Set allow_large_cov=True (and/or increase max_cov_bytes) to proceed."
            )
            warnings.warn(msg)
            raise RuntimeError(msg)

        device = theta0.device
        self.full_template = template  # shapes for unflattening
        self.full_mu = theta0.detach().clone()
        self.full_Sigma = torch.eye(d, device=device, dtype=cov_dtype) * float(init_var)
        #  import ipdb;ipdb.set_trace()

    @torch.no_grad()
    def _sample_full_param_mvn(
        self,
        store_z: bool = True,
        jitter: float = 1e-6,     # for numerical stability in cholesky
    ):
        # Cholesky sample
        d = self.full_mu.numel()
        S = self.full_Sigma
        if jitter > 0:
            S = S + torch.eye(d, device=S.device, dtype=S.dtype) * float(jitter)
        try:
            L = torch.linalg.cholesky(S)
        except:
            import ipdb;ipdb.set_trace()
        z = torch.randn_like(self.full_mu)
        theta_vec = self.full_mu + L @ z

        #  import ipdb;ipdb.set_trace()
        theta = self._unflatten_to_theta_dict(theta_vec, self.full_template, self.full_names)
        return theta, {"theta_vec": theta_vec, "z": z} if store_z else {"theta_vec": theta_vec}

    @torch.no_grad()
    def scale_mean_to_1e3(self, M, target=1e-3, eps=1e-12):
        M = M.clone()
        mean_val = M.abs().mean().clamp_min(eps)
        scale = target / mean_val
        return M * scale

    @torch.no_grad()
    def _update_full_param_mvn(
        self,
        theta_vec_list: List[torch.Tensor],  # list of [d] sampled parameter vectors
        weights: torch.Tensor,               # [R]
        ema: float = 0.2,
        cov_floor: float = 1e-8,
        max_mean_delta: Optional[float] = None,
        diag_ratio_clip: Optional[Tuple[float, float]] = (0.5, 2.0),
        shrinkage: float = 0.2,              # 0..1, pulls cov toward diagonal
    ):
        device = self.full_mu.device
        X = torch.stack([t.to(device, dtype=self.full_mu.dtype) for t in theta_vec_list], dim=0)  # [R,d]
        w = weights.to(device) 
        #  import ipdb;ipdb.set_trace()
        #  w = self.scale_mean_to_1e3(w)
        w = w / (w.sum() + 1e-12)

        # target moments
        mu_tgt = (w[:, None] * X).sum(dim=0)  # [d]
        Xc = X - mu_tgt[None, :]
        Sigma_tgt = (w[:, None, None] * (Xc[:, :, None] * Xc[:, None, :])).sum(dim=0) # [d,d]

        # diag floor
        d = Sigma_tgt.shape[0]
        diag = torch.diag(Sigma_tgt).clamp_min(cov_floor)
        Sigma_tgt = Sigma_tgt.clone()
        Sigma_tgt[range(d), range(d)] = diag

        # optional shrinkage toward diagonal
        if shrinkage > 0:
            Sigma_tgt = (1 - shrinkage) * Sigma_tgt + shrinkage * torch.diag(diag)

        # EMA
        mu_old, S_old = self.full_mu, self.full_Sigma
        mu_upd = (1 - ema) * mu_old + ema * mu_tgt
        S_upd  = (1 - ema) * S_old  + ema * Sigma_tgt
        #  mu_upd *= 2
        #  print(f"\nMu diffs: {abs(mu_upd-mu_old)}\n")
        # trust region on mean (abs clamp)
        if max_mean_delta is not None:
            delta = (mu_upd - mu_old).clamp(-max_mean_delta, max_mean_delta)
            mu_upd = mu_old + delta

        # trust region on covariance (diag-only multiplicative clamp)
        diag_old = torch.diag(S_old)
        diag_upd = torch.diag(S_upd).clamp_min(cov_floor)
        if diag_ratio_clip is not None:
            lo, hi = diag_ratio_clip
            ratio = (diag_upd / (diag_old + 1e-12)).clamp(lo, hi)
            diag_upd = (diag_old * ratio).clamp_min(cov_floor)

        S_upd = S_upd.clone()
        S_upd[range(d), range(d)] = diag_upd

        if shrinkage > 0:
            S_upd = (1 - shrinkage) * S_upd + shrinkage * torch.diag(diag_upd)

        self.full_mu = mu_upd
        self.full_Sigma = S_upd

        #  import ipdb;ipdb.set_trace()



# ========================================================================================================================
# 
# ========================================================================================================================
