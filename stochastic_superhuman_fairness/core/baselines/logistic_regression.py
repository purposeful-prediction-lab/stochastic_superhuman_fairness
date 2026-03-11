from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
import joblib
from pathlib import Path
import copy
import torch.nn as nn



def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def logistic_training_settings(level="medium"):
    """
    Return sklearn LogisticRegression training settings.

    Levels
    ------
    poor        : intentionally weak / underfit model
    medium      : reasonable default
    strong      : near full convergence

    Returns
    -------
    dict suitable for LogisticRegression(**settings)
    """

    level = level.lower()

    if level == "poor" or level == 'low':
        return dict(
            solver="liblinear",
            C=0.000001,
            max_iter=1,
            class_weight = {0: 0.5, 1: 1.8},
            tol=1e-2,
            n_jobs=1,
        )

    elif level == "medium" or level == 'intermediate':
        return dict(
            solver="lbfgs",
            C=1.0,
            max_iter=2,
            class_weight = {0: 0.5, 1: 1.8},
            tol=1e-4,
            n_jobs=1,
        )

    elif level in {"strong", "converged", "high"}:
        return dict(
            solver="lbfgs",
            C=5.0,
            max_iter=1500,
            tol=1e-6,
            n_jobs=1,
        )

    else:
        raise ValueError(f"Unknown level: {level}")

def train_logistic_from_demos(
    demos,
    *,
    solver="lbfgs",
    C=1.0,
    max_iter=200,
    tol = 1e-6,
    n_jobs=None,
    fit_intercept=True,
    class_weight=None,
    random_state=None,
    save_path=None,
):
    X_list = []
    y_list = []

    for d in demos:
        X = _to_numpy(d["X"])
        y = _to_numpy(d["y"]).reshape(-1)

        X_list.append(X)
        y_list.append(y)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    model = LogisticRegression(
        solver=solver,
        C=C,
        max_iter=max_iter,
        n_jobs=n_jobs,
        tol = tol,
        fit_intercept=fit_intercept,
        class_weight=class_weight,
        random_state=random_state,
    )
    model.fit(X, y)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)

    return model


def load_logistic_model(load_path):
    load_path = Path(load_path)
    return joblib.load(load_path)

def sample_logistic_model(cfg, seed=None):
    '''Sample a logistic model to serve as a demonstrator. Vary Regulirization type, level, class weight.'''
    rng = np.random.default_rng(seed)

    solvers = ["lbfgs", "liblinear", "saga"]
    solver = rng.choice(solvers)

    # Valid penalties per solver
    if solver == "liblinear":
        penalty = rng.choice(["l1", "l2"])
    elif solver == "saga":
        penalty = rng.choice(["l1", "l2", "elasticnet"])
    else:  # lbfgs
        penalty = "l2"

    #  C = float(rng.lognormal(mean=0.0, sigma=1.0))  # wide variability
    # vary regularization + class weights (clean)
    C = 10 ** rng.uniform(-4, 4)  # 1e-4..1e4
    cw = {0: float(10 ** rng.uniform(-0.5, 0.5)),
          1: float(10 ** rng.uniform(-0.5, 0.5))}

    max_iter = int(rng.integers(0,10))

    l1_ratio = None
    if penalty == "elasticnet":
        l1_ratio = rng.uniform(0.0, 1.0)

    lr = LogisticRegression(
        solver=solver,
        penalty=penalty,
        C=C,
        class_weight = cw,
        max_iter=max_iter,
        l1_ratio=l1_ratio,
        n_jobs=int(getattr(cfg, "lr_n_jobs", 1)),
        random_state=rng.integers(0, 10_000),
    )

    return lr



# ============================================================================================================
# SKLEARN TO Torch Helpers
# ============================================================================================================
def load_saved_sklearn_policy_into_torch(
    load_path,
    *,
    device="cpu",
    bias=True,
    dtype=torch.float32,
):
    """
    Load a saved sklearn LogisticRegression model and return a torch policy.

    Returns:
        nn.Linear(in_features, 1) with copied sklearn weights/bias, moved to device.
    """
    sk_model = load_logistic_model(load_path)
    return sklearn_model_to_torch_module(sk_model, device = device, dtype = dtype)

# ------------------------------------------------------------------------------------------

def sklearn_model_to_torch_module(sk_model, *, device="cpu", dtype=torch.float32):
    """
    Convert a fitted sklearn LogisticRegression model into a PyTorch module.

    Supports:
      - binary logistic regression
      - multiclass logistic regression

    Returns:
      nn.Module on the requested device
    """
    if not hasattr(sk_model, "coef_"):
        raise ValueError("Model does not appear to be a fitted sklearn linear/logistic model.")

    coef = torch.as_tensor(sk_model.coef_, dtype=dtype)
    has_bias = hasattr(sk_model, "intercept_")
    bias = torch.as_tensor(sk_model.intercept_, dtype=dtype) if has_bias else None

    out_features, in_features = coef.shape

    # sklearn binary logistic often stores coef_ as (1, D)
    layer = nn.Linear(in_features, out_features, bias=has_bias)

    with torch.no_grad():
        layer.weight.copy_(coef)
        if has_bias:
            layer.bias.copy_(bias)

    return layer.to(device)

# ------------------------------------------------------------------------------------------

def load_sklearn_logistic_into_torch_policy(
    sklearn_model,
    torch_policy: nn.Module,
    *,
    weight_attr: str = "linear",
    device=None,
):
    """
    Copy a fitted sklearn LogisticRegression into a torch policy.

    Assumes torch_policy has a submodule like:
        torch_policy.linear = nn.Linear(in_dim, 1)

    Args:
        sklearn_model: fitted sklearn LogisticRegression
        torch_policy: torch nn.Module to copy into
        weight_attr: name of the final linear layer
        device: optional device to move policy to

    Returns:
        torch_policy with copied weights
    """
    layer = getattr(torch_policy, weight_attr, None)
    if layer is None:
        raise ValueError(f"torch_policy has no attribute '{weight_attr}'")
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"torch_policy.{weight_attr} must be nn.Linear")

    coef = torch.as_tensor(sklearn_model.coef_, dtype=layer.weight.dtype)
    intercept = torch.as_tensor(sklearn_model.intercept_, dtype=layer.bias.dtype)

    if coef.shape != layer.weight.shape:
        raise ValueError(
            f"Weight shape mismatch: sklearn {tuple(coef.shape)} vs torch {tuple(layer.weight.shape)}"
        )
    if intercept.shape != layer.bias.shape:
        raise ValueError(
            f"Bias shape mismatch: sklearn {tuple(intercept.shape)} vs torch {tuple(layer.bias.shape)}"
        )

    with torch.no_grad():
        layer.weight.copy_(coef)
        layer.bias.copy_(intercept)

    if device is not None:
        torch_policy = torch_policy.to(device)

    return torch_policy


def sklearn_logistic_to_torch_clone(
    sklearn_model,
    base_policy: nn.Module,
    *,
    weight_attr: str = "linear",
    device=None,
):
    """
    Clone a torch policy template and load sklearn logistic weights into it.
    """
    pol = copy.deepcopy(base_policy)
    return load_sklearn_logistic_into_torch_policy(
        sklearn_model,
        pol,
        weight_attr=weight_attr,
        device=device,
    )
