
import torch
import numpy as np
import mosek
import matplotlib.pyplot as plt

def prepare_subdom_cost(
    S: np.ndarray,
    debias_rowcol: bool = True,
    normalize: bool = True,
    tau: float = 1.0,
    adaptive_tau: bool = True,
    target_std: float = 5.0,
    max_tau: float = 1000.0,
    tol_gamma_std: float = 0.03,
    solver_fn=None,                 # optional: callable returning γ for diagnostics
    solver_kwargs=None,
    verbose: bool = True,
):
    """
    Prepare the subdominance cost matrix before solving for γ,
    with optional adaptive temperature (τ) tuning.

    Parameters
    ----------
    S : np.ndarray
        Raw subdominance matrix (num_samples × num_demos).
    debias_rowcol : bool
        Remove additive A_i + B_j structure.
    normalize : bool
        Center and scale to 0 mean, unit std.
    tau : float
        Initial temperature (contrast multiplier).
    adaptive_tau : bool
        If True, increase τ until γ becomes sufficiently non-uniform.
    target_std : float
        Target standard deviation for the cost after τ scaling.
    max_tau : float
        Upper bound for adaptive τ.
    tol_gamma_std : float
        Minimum acceptable std(γ) to consider coupling “non-uniform.”
    solver_fn : callable or None
        Optional function handle: gamma = solver_fn(S, **solver_kwargs)
        Used only for diagnostic γ.std measurement during τ tuning.
    solver_kwargs : dict
        Passed to solver_fn if provided.
    verbose : bool
        Print summary statistics.

    Returns
    -------
    S_prepared : np.ndarray
        Preprocessed cost matrix (same shape as input).
    final_tau : float
        Final temperature used (after adaptive tuning).
    """

    S = np.asarray(S, dtype=float)
    if verbose:
        print(f"[prepare_subdom_cost] input mean/std = {S.mean():.3f}/{S.std():.3f}")

    # ------------------------------------------------------------
    # 1️⃣  Remove additive row/column bias
    # ------------------------------------------------------------
    if debias_rowcol:
        row_mean = S.mean(axis=1, keepdims=True)
        col_mean = S.mean(axis=0, keepdims=True)
        grand_mean = S.mean()
        S = S - row_mean - col_mean + grand_mean
        if verbose:
            print(f"[prepare_subdom_cost] debiased mean/std = {S.mean():.3f}/{S.std():.3f}")

    # ------------------------------------------------------------
    # 2️⃣  Normalize
    # ------------------------------------------------------------
    if normalize:
        std = S.std()
        if std < 1e-8:
            std = 1.0
        S = (S - S.mean()) / std
        if verbose:
            print(f"[prepare_subdom_cost] normalized mean/std = {S.mean():.3f}/{S.std():.3f}")

    # ------------------------------------------------------------
    # 3️⃣  Apply temperature scaling (initial τ)
    # ------------------------------------------------------------
    S_scaled = S * tau
    if verbose:
        print(f"[prepare_subdom_cost] applied initial τ={tau:.2f} → mean/std = {S_scaled.mean():.3f}/{S_scaled.std():.3f}")

    # ------------------------------------------------------------
    # 4️⃣  Optional adaptive τ tuning
    # ------------------------------------------------------------
    final_tau = tau
    if adaptive_tau and solver_fn is not None:
        # clone solver kwargs
        solver_kwargs = solver_kwargs or {}
        gamma_std = 0.0
        while gamma_std < tol_gamma_std and final_tau < max_tau:
            S_try = S * final_tau
            gamma_try, *_ = solver_fn(S_try, **solver_kwargs)
            gamma_std = float(np.std(gamma_try))
            if verbose:
                print(f"[τ-tune] τ={final_tau:.2f} → γ.std={gamma_std:.4f}")
            if gamma_std < tol_gamma_std:
                final_tau *= 1.5  # increase contrast
            else:
                break

        if final_tau > max_tau:
            final_tau = max_tau
            if verbose:
                print(f"[τ-tune] reached max τ={max_tau:.2f} (γ.std={gamma_std:.4f})")
        else:
            if verbose:
                print(f"[τ-tune] final τ={final_tau:.2f} achieved γ.std={gamma_std:.4f}")

        S_scaled = S * final_tau

    # Optional: rescale to target_std (e.g. 5.0)
    if target_std is not None and target_std > 0:
        S_scaled = (S_scaled - S_scaled.mean()) / (S_scaled.std() + 1e-8) * target_std

    if verbose:
        print(f"[prepare_subdom_cost] final mean/std = {S_scaled.mean():.3f}/{S_scaled.std():.3f}")

    return S_scaled, final_tau


"""
Superhuman Fairness — Coupling Solvers  (samples × demos version)
-----------------------------------------------------------------

Unified interface for computing γ couplings used in subdominance
optimization. Supports both:
   - MOSEK QP (classic constrained OT)
   - Sinkhorn entropic OT (strictly convex, smooth coupling)

✅ Orientation
    subdom_matrix.shape == (num_samples, num_demos)
    rows → model rollouts / samples
    cols → demonstrations

Usage
-----
gamma, dual_r, dual_c = solve_qp_superhuman_samples_demos(
    subdom_matrix=S_np,
    rollout_marginals=None,
    solver="mosek",     # or "sinkhorn"
    lambda_reg=None,
    normalize_subdom=True,
    tau=5.0,
    epsilon=0.05,
    verbose=True,
)
"""

# ============================================================
# 🔹 MOSEK QP solver  (samples × demos)
# ============================================================
def _solve_qp_mosek_core(
    subdom_matrix: np.ndarray,
    rollout_marginals: np.ndarray | list = None,
    lambda_reg: float | None = None,
    normalize_subdom: bool = True,
    tau: float = 1.0,
    verbose: bool = True,
    row_constraints: bool = True,
    col_constraints: bool = True,
):
    """
    Solve QP:
        minimize_γ   <γ, S> + (λ/2)||γ||²
        s.t.          γ ≥ 0,
                      (optional) γ 1 = p_samples
                      (optional) γᵀ 1 = q_demos
    """

    # --- Input preparation ---
    S = np.asarray(subdom_matrix, dtype=float)
    num_samples, num_demos = S.shape
    n = num_samples * num_demos

    # --- Scaling ---
    if normalize_subdom:
        std = S.std()
        if std < 1e-8:
            std = 1.0
        S = (S - S.mean()) / std
    if tau != 1.0:
        S = S * tau
    if lambda_reg is None:
        lambda_reg = 1e-6
    # Reduce lampbda in unconstrained cases
    if not (row_constraints or col_constraints):
        lambda_reg = 1e-9

    if verbose:
        print(f"[MOSEK] Solving QP with λ={lambda_reg:.1e}, τ={tau}, "
              f"rows={row_constraints}, cols={col_constraints}")

    # --- Marginals ---
    if rollout_marginals is None:
        p_samples = np.ones(num_samples) / num_samples
    else:
        p_samples = np.asarray(rollout_marginals, dtype=float)
        p_samples /= p_samples.sum()

    q_demos = np.ones(num_demos) / num_demos

    # --- MOSEK setup ---
    with mosek.Env() as env, env.Task(0, 0) as task:

        task.putobjsense(mosek.objsense.minimize)
        task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, 1e-8)

        # Variables γ_ij ≥ 0
        task.appendvars(n)
        task.putvarboundslice(0, n,
                              [mosek.boundkey.lo]*n,
                              [0.0]*n,
                              [float("inf")]*n)

        # --- Constraints ---
        constraint_idx = 0
        task.appendcons(1)
        idx = list(range(n))
        task.putarow(constraint_idx, idx, [1.0]*n)
        task.putconbound(constraint_idx, mosek.boundkey.fx, 1.0, 1.0)
        constraint_idx += 1
        # Row constraints (samples)
        if row_constraints:
            task.appendcons(num_samples)
            for i in range(num_samples):
                idx = [i*num_demos + j for j in range(num_demos)]
                task.putarow(constraint_idx + i, idx, [1.0]*num_demos)
                task.putconbound(constraint_idx + i, mosek.boundkey.fx,
                                 p_samples[i], p_samples[i])
            constraint_idx += num_samples

        # Column constraints (demos)
        if col_constraints:
            task.appendcons(num_demos)
            for j in range(num_demos):
                idx = [i*num_demos + j for i in range(num_samples)]
                task.putarow(constraint_idx + j, idx, [1.0]*num_samples)
                task.putconbound(constraint_idx + j, mosek.boundkey.fx,
                                 q_demos[j], q_demos[j])
            constraint_idx += num_demos

        # Linear objective
        task.putcslice(0, n, S.flatten())

        # Optional quadratic regularization
        if lambda_reg > 0:
            qsubi = qsubj = list(range(n))
            qval = [lambda_reg] * n
            task.putqobj(qsubi, qsubj, qval)

        # --- Solve ---
        # print("Constraints added:", task.getnumcon())
        # print("Expect:", (num_samples if row_constraints else 0) + (num_demos if col_constraints else 0))
        task.optimize()
        solsta = task.getsolsta(mosek.soltype.itr)
        if solsta != mosek.solsta.optimal:
            raise RuntimeError(f"[MOSEK] optimization failed: {solsta}")

        # --- Extract primal solution ---
        gamma = np.zeros(n)
        task.getxx(mosek.soltype.itr, gamma)
        gamma_matrix = gamma.reshape((num_samples, num_demos))
        # print("γ column sums:", gamma_matrix.sum(axis=0))
        # print("Target q:", q_demos)

        # --- Extract duals safely ---
        duals = np.zeros(constraint_idx)
        try:
            task.gety(mosek.soltype.itr, duals)
        except Exception:
            duals[:] = 0.0

        # Partition duals based on which constraints are active
        dual_rows = np.zeros(num_samples)
        dual_cols = np.zeros(num_demos)
        offset = 0
        if row_constraints:
            dual_rows = duals[offset:offset + num_samples]
            offset += num_samples
        if col_constraints:
            dual_cols = duals[offset:offset + num_demos]

    # --- Diagnostics ---
    row_sum = gamma_matrix.sum(axis=1)
    col_sum = gamma_matrix.sum(axis=0)
    if verbose:
        print(f"γ mean/std: {gamma_matrix.mean():.6f} / {gamma_matrix.std():.6f}")
        print(f"γ row-sum std: {row_sum.std():.3e}, col-sum std: {col_sum.std():.3e}")

    return gamma_matrix, dual_rows, dual_cols
# ============================================================
# 🔹 Sinkhorn entropic solver  (samples × demos)
# ============================================================
def _solve_sinkhorn_core(
    subdom_matrix: np.ndarray,
    rollout_marginals: np.ndarray | list = None,  # rows p
    normalize_subdom: bool = True,
    tau: float = 1.0,
    epsilon: float = 0.05,
    max_iters: int = 1_000,
    tol: float = 1e-8,
    verbose: bool = True,
    row_constraints: bool = True,
    col_constraints: bool = True,
    renormalize_gamma: bool = True,   # optional: scale Γ to sum 1 for interpretability
):
    """
    Entropic OT (Sinkhorn) with selectable marginal constraints.

    If row_constraints=False, rows are not fitted (only columns if enabled).
    If col_constraints=False, columns are not fitted (only rows if enabled).
    If both are False, returns unprojected kernel mass Γ = K (optionally renormalized).

    Returns
    -------
    gamma  : (num_samples, num_demos)
    dual_r : zeros if row_constraints=False, else log u (up to constants)
    dual_c : zeros if col_constraints=False, else log v (up to constants)
    """
    S = np.asarray(subdom_matrix, dtype=float)
    num_samples, num_demos = S.shape

    # ---- scale costs (match your MOSEK path) ----
    if normalize_subdom:
        std = S.std()
        if std < 1e-8: std = 1.0
        S = (S - S.mean()) / std
    if tau != 1.0:
        S = S * tau

    # ---- marginals ----
    if rollout_marginals is None:
        p = np.ones(num_samples) / num_samples
    else:
        p = np.asarray(rollout_marginals, dtype=float)
        p = p / (p.sum() + 1e-12)

    q = np.ones(num_demos) / num_demos

    # ---- kernel (log-stable shift) ----
    S_shift = S - S.min()
    K = np.exp(-S_shift / max(epsilon, 1e-12)) + 1e-300  # keep strictly positive

    # ---- init scalings ----
    u = np.ones(num_samples) / num_samples
    v = np.ones(num_demos)  / num_demos

    # If a side is unconstrained, we keep its scale at 1 (no projection)
    if not row_constraints:
        u[:] = 1.0
    if not col_constraints:
        v[:] = 1.0

    def row_err(g):
        return np.linalg.norm(g.sum(axis=1) - p, 1)
    def col_err(g):
        return np.linalg.norm(g.sum(axis=0) - q, 1)

    # ---- iterations ----
    for it in range(max_iters):
        u_prev, v_prev = u.copy(), v.copy()

        # project rows if requested
        if row_constraints:
            Ku = K @ v
            Ku[Ku == 0] = 1e-300
            u = p / Ku

        # project columns if requested
        if col_constraints:
            KTu = K.T @ u
            KTu[KTu == 0] = 1e-300
            v = q / KTu

        # diagnostics
        if (it % 50 == 0) or (it == max_iters - 1):
            gamma = (u[:, None] * K) * v[None, :]
            err_r = row_err(gamma) if row_constraints else 0.0
            err_c = col_err(gamma) if col_constraints else 0.0
            err = max(err_r, err_c)
            if verbose:
                print(f"[Sinkhorn] it={it:04d} | "
                      f"marg_err(L1) rows={err_r:.2e} cols={err_c:.2e}")
            # stopping rule: only check enabled sides
            if err < tol:
                break

        # early stationary check (helps when both sides disabled)
        if np.allclose(u, u_prev, atol=1e-14) and np.allclose(v, v_prev, atol=1e-14):
            break

    gamma = (u[:, None] * K) * v[None, :]

    # Optional global renormalization: makes Γ a probability table even if unconstrained
    if renormalize_gamma:
        s = gamma.sum()
        if s > 0:
            gamma = gamma / s

    # Duals (log scalings). If side disabled, return zeros.
    dual_rows = np.log(u + 1e-300) if row_constraints else np.zeros(num_samples)
    dual_cols = np.log(v + 1e-300) if col_constraints else np.zeros(num_demos)

    if verbose:
        rs, cs = gamma.sum(axis=1), gamma.sum(axis=0)
        print(f"γ mean/std: {gamma.mean():.6f} / {gamma.std():.6f}")
        print(f"γ row-sum std: {rs.std():.3e}, col-sum std: {cs.std():.3e}")

    return gamma, dual_rows, dual_cols
# ============================================================
# 🔹 Unified entry point  (samples × demos)
# ============================================================
def solve_qp_superhuman(
    subdom_matrix: np.ndarray,
    rollout_marginals: np.ndarray | list = None,
    solver: str = "mosek",         # "mosek" or "sinkhorn"
    lambda_reg: float | None = None,
    normalize_subdom: bool = True,
    tau: float = 1.0,
    epsilon: float = 0.05,
    row_constraints: bool = True,
    col_constraints: bool = True,
    verbose: bool = True,
):
    """
    Unified interface for γ coupling solvers.
    Expects subdom_matrix with shape (num_samples, num_demos)
    """
    solver = solver.lower()
    if solver == "mosek":
        return _solve_qp_mosek_core(
            subdom_matrix=subdom_matrix,
            rollout_marginals=rollout_marginals,
            lambda_reg=lambda_reg,
            normalize_subdom=normalize_subdom,
            tau=tau,
            col_constraints = col_constraints,
            row_constraints = row_constraints,
            verbose=verbose,
        )
    elif solver == "sinkhorn":
        return _solve_sinkhorn_core(
            subdom_matrix=subdom_matrix,
            rollout_marginals=rollout_marginals,
            normalize_subdom=normalize_subdom,
            tau=tau,
            epsilon=epsilon,
            col_constraints = col_constraints,
            row_constraints = row_constraints,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown solver '{solver}'. Must be 'mosek' or 'sinkhorn'.")


# ============================================================
# 🔹 Visualization helper
# ============================================================
def plot_gamma_heatmap(gamma_matrix, title="γ Coupling (samples × demos)", cmap="viridis"):
    plt.figure(figsize=(6, 4))
    plt.imshow(gamma_matrix, cmap=cmap, aspect="auto")
    plt.colorbar(label="γ value")
    plt.title(title)
    plt.xlabel("Demos")
    plt.ylabel("Samples")
    plt.tight_layout()
    plt.show()

def compute_subdom_weights(
    gamma, dual_rows, dual_cols,
    method: str = "primal",
    backend: str = "numpy"
):
    """
    method:
      - 'primal'    → weights = gamma.sum(axis=1)
      - 'dual'      → weights = dual_cols
      - 'row_dual'  → weights = dual_rows

    backend:
      - 'numpy'
      - 'torch'
    """

    method = method.lower()
    if method == "primal":
        w = gamma.sum(axis=1 if backend=="numpy" else 1)

    elif method == "dual":
        if dual_cols is None:
            raise ValueError("dual_cols unavailable for method='dual'")
        w = dual_cols

    elif method == "row_dual":
        if dual_rows is None:
            raise ValueError("dual_rows unavailable for method='row_dual'")
        w = dual_rows

    else:
        raise ValueError(f"Unknown weight method '{method}'")

    return w
# ==================================================================================================
# All in one call
# ==================================================================================================
def solve_stochastic_subdom_coupling(
    S,
    solver="mosek",
    rollout_marginals=None,
    debias_rowcol=True,
    normalize=True,
    tau=1.0,
    adaptive_tau=False,
    target_std=5.0,
    max_tau=1000.0,
    tol_gamma_std=0.03,
    row_constraints=True,
    col_constraints=True,
    lambda_reg=None,
    epsilon=0.05,
    verbose=False,
    weight_method="primal",   # NEW
):
    # ---- Convert input to numpy ----
    if torch.is_tensor(S):
        device = S.device
        S_np = S.detach().cpu().numpy().astype(float)
        backend = "torch"
    else:
        S_np = np.asarray(S, dtype=float)
        device = None
        backend = "numpy"

    # ---- Preprocess S ----
    solver_kwargs_for_tau = {
        "rollout_marginals": rollout_marginals,
        "solver": solver,
        "lambda_reg": lambda_reg,
        "normalize_subdom": False,
        "tau": 1.0,
        "epsilon": epsilon,
        "row_constraints": row_constraints,
        "col_constraints": col_constraints,
        "verbose": False,
    }

    S_pre, final_tau = prepare_subdom_cost(
        S_np,
        debias_rowcol=debias_rowcol,
        normalize=normalize,
        tau=tau,
        adaptive_tau=adaptive_tau,
        target_std=target_std,
        max_tau=max_tau,
        tol_gamma_std=tol_gamma_std,
        solver_fn=solve_qp_superhuman,
        solver_kwargs=solver_kwargs_for_tau,
        verbose=verbose,
    )

    # ---- Solve QP/OT ----
    gamma_np, dual_rows_np, dual_cols_np = solve_qp_superhuman(
        subdom_matrix=S_pre,
        rollout_marginals=rollout_marginals,
        solver=solver,
        lambda_reg=lambda_reg,
        normalize_subdom=False,
        tau=1.0,
        epsilon=epsilon,
        row_constraints=row_constraints,
        col_constraints=col_constraints,
        verbose=verbose,
    )

    # ---- Compute weights (numpy) ----
    weights_np = compute_subdom_weights(
        gamma_np, dual_rows_np, dual_cols_np,
        method=weight_method,
        backend="numpy"
    )

    # ---- Convert to torch if needed ----
    if backend == "torch":
        gamma_t  = torch.tensor(gamma_np, device=device, dtype=torch.float32)
        dual_r_t = torch.tensor(dual_rows_np, device=device, dtype=torch.float32) if dual_rows_np is not None else None
        dual_c_t = torch.tensor(dual_cols_np, device=device, dtype=torch.float32) if dual_cols_np is not None else None
        weights_t = torch.tensor(weights_np, device=device, dtype=torch.float32)
    else:
        gamma_t = dual_r_t = dual_c_t = weights_t = None

    return {
        # original outputs
        "S_prepared_np": S_pre,
        "tau": final_tau,
        "gamma_np": gamma_np,
        "dual_rows_np": dual_rows_np,
        "dual_cols_np": dual_cols_np,

        # new: torch versions
        "gamma_torch": gamma_t,
        "dual_rows_torch": dual_r_t,
        "dual_cols_torch": dual_c_t,

        # new: weights
        "weights_np": weights_np,
        "weights_torch": weights_t,
    }
