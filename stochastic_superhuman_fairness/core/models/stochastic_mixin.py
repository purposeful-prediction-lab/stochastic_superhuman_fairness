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

    @torch.no_grad()
    def _sample_full_param_mvn(
        self,
        store_z: bool = True,
        jitter: float = 1e-8,     # for numerical stability in cholesky
    ):
        # Cholesky sample
        d = self.full_mu.numel()
        S = self.full_Sigma
        if jitter > 0:
            S = S + torch.eye(d, device=S.device, dtype=S.dtype) * float(jitter)
        L = torch.linalg.cholesky(S)
        z = torch.randn_like(self.full_mu)
        theta_vec = self.full_mu + L @ z

        #  import ipdb;ipdb.set_trace()
        theta = self._unflatten_to_theta_dict(theta_vec, self.full_template, self.full_names)
        return theta, {"theta_vec": theta_vec, "z": z} if store_z else {"theta_vec": theta_vec}

    @torch.no_grad()
    def _update_full_param_mvn(
        self,
        theta_vec_list: List[torch.Tensor],  # list of [d] sampled parameter vectors
        weights: torch.Tensor,               # [R]
        ema: float = 0.2,
        cov_floor: float = 1e-8,
        max_mean_delta: Optional[float] = None,
        diag_ratio_clip: Optional[Tuple[float, float]] = (0.5, 2.0),
        shrinkage: float = 0.0,              # 0..1, pulls cov toward diagonal
    ):
        device = self.full_mu.device
        X = torch.stack([t.to(device, dtype=self.full_mu.dtype) for t in theta_vec_list], dim=0)  # [R,d]
        w = weights.to(device)
        w = w / (w.sum() + 1e-12)

        # target moments
        mu_tgt = (w[:, None] * X).sum(dim=0)  # [d]
        Xc = X - mu_tgt[None, :]
        Sigma_tgt = (w[:, None, None] * (Xc[:, :, None] * Xc[:, None, :])).sum(dim=0)  # [d,d]

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



# ========================================================================================================================
# 
# ========================================================================================================================




#  import torch
#  from typing import Dict, List, Tuple, Optional, Literal
#
#  DistMode = Literal["layer_diag", "neuron_mvn"]
#
#  class StochasticParamDistMixin:
#      """
#      PyTorch mixin: maintains a distribution to sample policy parameters from, with:
#        - layer-wise diagonal Gaussian over all params (mean + var per tensor)
#        - neuron-level MVN latent z (size=h) with covariance (h x h), injected into params
#        - beta (cost temperature) for OT inputs
#        - covariance floors
#        - trust-region style updates (mean step clamp + variance/cov diag ratio clamp)
#      """
#
#      # ============================================================
#      # 0) OT temperature
#      # ============================================================
#      @torch.no_grad()
#      def apply_ot_temperature(self, S, beta: float = 1.0):
#          """Use before OT: S_beta = beta * S."""
#          return beta * S
#
#      # ============================================================
#      # A) Layer-wise diagonal Gaussian over parameters
#      # ============================================================
#      @torch.no_grad()
#      def init_layer_diag_dist(self, init_var: float = 1e-4):
#          """Initialize mean/var dicts from current policy parameters. Usually mean = starting val and covar = set"""
#          self.dist_mode: DistMode = "layer_diag"
#          self.param_mean: Dict[str, torch.Tensor] = {}
#          self.param_var: Dict[str, torch.Tensor] = {}
#          for n, p in self.policy.named_parameters():
#              self.param_mean[n] = p.detach().clone()
#              self.param_var[n] = torch.full_like(p, float(init_var))
#
#      @torch.no_grad()
#      def sample_layer_diag_params(
#          self,
#          store_eps: bool = True,
#      ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
#          """
#          Sample theta[name] = mean[name] + sqrt(var[name]) * eps[name].
#          Returns sampled params dict + eps dict (optional).
#          """
#          theta: Dict[str, torch.Tensor] = {}
#          eps: Optional[Dict[str, torch.Tensor]] = {} if store_eps else None
#          for n, p in self.policy.named_parameters():
#              e = torch.randn_like(p)
#              std = torch.sqrt(self.param_var[n])
#              theta[n] = self.param_mean[n] + std * e
#              if store_eps:
#                  eps[n] = e
#          return theta, eps
#
#      @torch.no_grad()
#      def load_params_into_policy(self, theta: Dict[str, torch.Tensor]):
#          """In-place overwrite live policy params with theta dict."""
#          for n, p in self.policy.named_parameters():
#              p.copy_(theta[n])
#
#      @torch.no_grad()
#      def compute_new_layer_diag_from_eps(
#          self,
#          eps_list: List[Dict[str, torch.Tensor]],  # length R
#          weights: torch.Tensor,                    # [R]
#          var_floor: float = 1e-8,
#      ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
#          """
#          Moment-match new mean/var using eps only (no need to store theta snapshots).
#            theta_r = mu + std * eps_r
#            mu_new = sum w_r theta_r
#            var_new = sum w_r (theta_r - mu_new)^2   (diagonal)
#          """
#          device = next(self.policy.parameters()).device
#          w = weights.to(device)
#          w = w / (w.sum() + 1e-12)
#
#          new_mean: Dict[str, torch.Tensor] = {}
#          new_var: Dict[str, torch.Tensor] = {}
#
#          for n, _p in self.policy.named_parameters():
#              E = torch.stack([e[n].to(device) for e in eps_list], dim=0)  # [R, *shape]
#              wv = w.view(-1, *([1] * (E.dim() - 1)))
#              eps_bar = (wv * E).sum(dim=0)
#
#              mu = self.param_mean[n]
#              var = self.param_var[n]
#              std = torch.sqrt(var)
#
#              mu_new = mu + std * eps_bar
#              centered = E - eps_bar
#              var_scale = (wv * centered.pow(2)).sum(dim=0)
#              var_new = (var * var_scale).clamp_min(var_floor)
#
#              new_mean[n] = mu_new
#              new_var[n] = var_new
#
#          return new_mean, new_var
#
#      @torch.no_grad()
#      def update_layer_diag_dist(
#          self,
#          new_mean: Dict[str, torch.Tensor],
#          new_var: Dict[str, torch.Tensor],
#          ema: float = 0.2,
#          var_floor: float = 1e-8,
#          max_mean_delta: Optional[float] = None,                 # abs clamp per-element
#          var_ratio_clip: Optional[Tuple[float, float]] = (0.5, 2.0),  # multiplicative clamp
#      ):
#          """EMA + trust region + floor update for layer-diagonal distribution."""
#          for n, _p in self.policy.named_parameters():
#              mu_old = self.param_mean[n]
#              v_old  = self.param_var[n]
#
#              mu_tgt = new_mean[n]
#              v_tgt  = new_var[n].clamp_min(var_floor)
#
#              mu_upd = (1 - ema) * mu_old + ema * mu_tgt
#              v_upd  = (1 - ema) * v_old  + ema * v_tgt
#
#              if max_mean_delta is not None:
#                  delta = (mu_upd - mu_old).clamp(-max_mean_delta, max_mean_delta)
#                  mu_upd = mu_old + delta
#
#              if var_ratio_clip is not None:
#                  lo, hi = var_ratio_clip
#                  ratio = (v_upd / (v_old + 1e-12)).clamp(lo, hi)
#                  v_upd = (v_old * ratio).clamp_min(var_floor)
#
#              self.param_mean[n] = mu_upd
#              self.param_var[n]  = v_upd
#
#      # ============================================================
#      # B) Neuron-level MVN latent z (h x h covariance)
#      # ============================================================
#      @torch.no_grad()
#      def init_neuron_mvn_dist(
#          self,
#          hidden_size: int,
#          init_var: float = 1e-2,
#          init_mu: float = 0.0,
#      ):
#          """
#          Maintain z ~ N(z_mu, z_Sigma), z in R^h, with feasible (h x h) covariance.
#          """
#          self.dist_mode = "neuron_mvn"
#          self.h = int(hidden_size)
#          device = next(self.policy.parameters()).device
#          self.z_mu = torch.full((self.h,), float(init_mu), device=device)
#          self.z_Sigma = torch.eye(self.h, device=device) * float(init_var)
#
#      @torch.no_grad()
#      def sample_neuron_latent(self) -> Tuple[torch.Tensor, torch.Tensor]:
#          """Sample z, returning (z, eps) where z = mu + chol(Sigma) @ eps."""
#          L = torch.linalg.cholesky(self.z_Sigma)  # [h,h]
#          eps = torch.randn_like(self.z_mu)        # [h]
#          z = self.z_mu + L @ eps
#          return z, eps
#
#      @torch.no_grad()
#      def inject_z_into_params(
#          self,
#          theta: Dict[str, torch.Tensor],
#          z: torch.Tensor,
#          *,
#          fc1_w: str = "fc1.weight",
#          fc1_b: str = "fc1.bias",
#          fc2_w: str = "fc2.weight",
#          # scales:
#          z_to_b1: float = 1.0,
#          z_to_W1_rows: float = 0.0,
#          z_to_W2_cols: float = 0.0,
#      ) -> Dict[str, torch.Tensor]:
#          """
#          Inject neuron latent z (size h) into a sampled parameter dict theta.
#          Assumes a 2-layer MLP naming by default: fc1.*, fc2.*. Override names if needed.
#
#          - bias injection: b1 += z_to_b1 * z
#          - incoming rows:  W1 += z_to_W1_rows * z[:,None]
#          - outgoing cols:  W2 += z_to_W2_cols * z[None,:]
#          """
#          if fc1_b in theta:
#              theta[fc1_b] = theta[fc1_b] + float(z_to_b1) * z
#
#          if float(z_to_W1_rows) != 0.0 and fc1_w in theta:
#              theta[fc1_w] = theta[fc1_w] + float(z_to_W1_rows) * z[:, None]
#
#          if float(z_to_W2_cols) != 0.0 and fc2_w in theta:
#              theta[fc2_w] = theta[fc2_w] + float(z_to_W2_cols) * z[None, :]
#
#          return theta
#
#      @torch.no_grad()
#      def compute_new_neuron_mvn(
#          self,
#          z_list: List[torch.Tensor],  # length R, each [h]
#          weights: torch.Tensor,       # [R]
#          cov_floor: float = 1e-6,
#          shrinkage: float = 0.0,      # 0..1
#      ) -> Tuple[torch.Tensor, torch.Tensor]:
#          """Weighted mean/cov of z's (h-dim) with diag floor and optional shrinkage."""
#          device = self.z_mu.device
#          Z = torch.stack([z.to(device) for z in z_list], dim=0)  # [R,h]
#          w = weights.to(device)
#          w = w / (w.sum() + 1e-12)
#
#          mu = (w[:, None] * Z).sum(dim=0)  # [h]
#          Zc = Z - mu[None, :]
#          Sigma = (w[:, None, None] * (Zc[:, :, None] * Zc[:, None, :])).sum(dim=0)  # [h,h]
#
#          h = Sigma.shape[0]
#          diag = torch.diag(Sigma).clamp_min(cov_floor)
#          Sigma = Sigma.clone()
#          Sigma[range(h), range(h)] = diag
#
#          if shrinkage > 0:
#              Sigma = (1 - shrinkage) * Sigma + shrinkage * torch.diag(diag)
#
#          return mu, Sigma
#
#      @torch.no_grad()
#      def update_neuron_mvn_dist(
#          self,
#          new_mu: torch.Tensor,
#          new_Sigma: torch.Tensor,
#          ema: float = 0.2,
#          cov_floor: float = 1e-6,
#          max_mean_delta: Optional[float] = None,                 # abs clamp per-dim
#          diag_ratio_clip: Optional[Tuple[float, float]] = (0.5, 2.0),  # trust region on diag
#          shrinkage: float = 0.0,
#      ):
#          """EMA + trust region + floor update for neuron MVN."""
#          mu_old, S_old = self.z_mu, self.z_Sigma
#          mu_tgt = new_mu.to(mu_old.device)
#          S_tgt  = new_Sigma.to(mu_old.device)
#
#          mu_upd = (1 - ema) * mu_old + ema * mu_tgt
#          S_upd  = (1 - ema) * S_old  + ema * S_tgt
#
#          if max_mean_delta is not None:
#              delta = (mu_upd - mu_old).clamp(-max_mean_delta, max_mean_delta)
#              mu_upd = mu_old + delta
#
#          h = S_upd.shape[0]
#          diag_old = torch.diag(S_old)
#          diag_upd = torch.diag(S_upd).clamp_min(cov_floor)
#
#          if diag_ratio_clip is not None:
#              lo, hi = diag_ratio_clip
#              ratio = (diag_upd / (diag_old + 1e-12)).clamp(lo, hi)
#              diag_upd = (diag_old * ratio).clamp_min(cov_floor)
#
#          S_upd = S_upd.clone()
#          S_upd[range(h), range(h)] = diag_upd
#
#          if shrinkage > 0:
#              S_upd = (1 - shrinkage) * S_upd + shrinkage * torch.diag(diag_upd)
#
#          self.z_mu = mu_upd
#          self.z_Sigma = S_upd
#
#      # ============================================================
#      # C) High-level “sample policy params” that supports both modes
#      # ============================================================
#      @torch.no_grad()
#      def sample_policy_params(
#          self,
#          mode: DistMode,
#          *,
#          # neuron injection config:
#          neuron_inject_cfg: Optional[dict] = None,
#      ):
#          """
#          Returns:
#            theta: dict[name->tensor] sampled params to load into policy
#            aux:   dict with stored objects for later updates
#                   - for layer_diag: {"eps": eps_dict}
#                   - for neuron_mvn: {"eps": eps_dict, "z": z}
#          """
#          if mode == "layer_diag":
#              theta, eps = self.sample_layer_diag_params(store_eps=True)
#              return theta, {"eps": eps}
#
#          if mode == "neuron_mvn":
#              if neuron_inject_cfg is None:
#                  neuron_inject_cfg = {}
#
#              # base sample from layer-diag (still recommended) then inject correlated z
#              theta, eps = self.sample_layer_diag_params(store_eps=True)
#              z, _ = self.sample_neuron_latent()
#
#              theta = self.inject_z_into_params(
#                  theta,
#                  z,
#                  fc1_w=neuron_inject_cfg.get("fc1_w", "fc1.weight"),
#                  fc1_b=neuron_inject_cfg.get("fc1_b", "fc1.bias"),
#                  fc2_w=neuron_inject_cfg.get("fc2_w", "fc2.weight"),
#                  z_to_b1=neuron_inject_cfg.get("z_to_b1", 1.0),
#                  z_to_W1_rows=neuron_inject_cfg.get("z_to_W1_rows", 0.0),
#                  z_to_W2_cols=neuron_inject_cfg.get("z_to_W2_cols", 0.0),
#              )
#              return theta, {"eps": eps, "z": z}
#
#          raise ValueError(f"Unknown mode: {mode}")
#
#      # ============================================================
#      # D) High-level “update distributions” given OT weights
#      # ============================================================
#      @torch.no_grad()
#      def update_distributions_from_rollouts(
#          self,
#          mode: DistMode,
#          *,
#          # collected during rollout generation:
#          eps_list: List[Dict[str, torch.Tensor]],
#          weights: torch.Tensor,   # [R] typically row-sums of gamma
#          # layer-diag knobs:
#          var_floor: float = 1e-8,
#          ema: float = 0.2,
#          max_mean_delta: Optional[float] = None,
#          var_ratio_clip: Optional[Tuple[float, float]] = (0.5, 2.0),
#          # neuron mvn knobs (only used if mode == "neuron_mvn"):
#          z_list: Optional[List[torch.Tensor]] = None,
#          cov_floor: float = 1e-6,
#          diag_ratio_clip: Optional[Tuple[float, float]] = (0.5, 2.0),
#          shrinkage: float = 0.0,
#      ):
#          """
#          Updates:
#            - always updates layer-diag mean/var (needed for sampling theta dict)
#            - if mode == neuron_mvn, also updates z_mu/z_Sigma using z_list
#          """
#          # Update layer-diag in all cases (it underpins theta sampling)
#          new_mean, new_var = self.compute_new_layer_diag_from_eps(
#              eps_list=eps_list,
#              weights=weights,
#              var_floor=var_floor,
#          )
#          self.update_layer_diag_dist(
#              new_mean=new_mean,
#              new_var=new_var,
#              ema=ema,
#              var_floor=var_floor,
#              max_mean_delta=max_mean_delta,
#              var_ratio_clip=var_ratio_clip,
#          )
#
#          if mode == "neuron_mvn":
#              if z_list is None:
#                  raise ValueError("mode='neuron_mvn' requires z_list.")
#              new_z_mu, new_z_Sigma = self.compute_new_neuron_mvn(
#                  z_list=z_list,
#                  weights=weights,
#                  cov_floor=cov_floor,
#                  shrinkage=shrinkage,
#              )
#              self.update_neuron_mvn_dist(
#                  new_mu=new_z_mu,
#                  new_Sigma=new_z_Sigma,
#                  ema=ema,
#                  cov_floor=cov_floor,
#                  max_mean_delta=max_mean_delta,
#                  diag_ratio_clip=diag_ratio_clip,
#                  shrinkage=shrinkage,
#              )
