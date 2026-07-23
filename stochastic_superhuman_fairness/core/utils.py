from omegaconf import OmegaConf
from types import SimpleNamespace
import os, random, numpy as np, torch
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from dataclasses import is_dataclass

@torch.no_grad()
def sample_binary_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """
    probs: tensor in [0,1], any shape
    returns: {0,1} float tensor of same shape
    """
    return torch.bernoulli(probs).float()

def sample_rollouts_from_probs(probs: torch.Tensor, n_rollouts: int) -> torch.Tensor:
    """
    probs: [P, N]
    returns: [P, M, N]
    """
    if probs.ndim != 2:
        raise ValueError(f"Expected probs to have shape [P, N], got {tuple(probs.shape)}")

    P, N = probs.shape
    probs_expanded = probs.unsqueeze(1).expand(P, n_rollouts, N)  # [P, M, N]
    return torch.bernoulli(probs_expanded)

class NamespaceDict(SimpleNamespace):
    """A SimpleNamespace with dict-like get() method and repr that hides internals."""

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        if not hasattr(self, key):
            raise KeyError(key)
        delattr(self, key)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

    def __iter__(self):
        return iter(vars(self))

    def items(self):
        return vars(self).items()

    def keys(self):
        return vars(self).keys()

    def values(self):
        return vars(self).values()

    def as_dict(self):
        """Recursively convert to a dict."""
        def convert(obj):
            if isinstance(obj, NamespaceDict):
                return {k: convert(v) for k, v in vars(obj).items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        return convert(self)

    def update_from(self, other):
            """
            Update this NamespaceDict with values from `other`.
            - Keys missing in `other` are left untouched; keys with None values get assigned self values.
            - If both values are NamespaceDict, update recursively.
            """
            if not isinstance(other, NamespaceDict):
                raise TypeError("update_from expects a NamespaceDict")

            #  import ipdb;ipdb.set_trace()
            for key, value in vars(other).items():
                if hasattr(self, key):
                    current = getattr(self, key)
                    if value is not None:
                        if isinstance(current, NamespaceDict) and isinstance(value, NamespaceDict):
                            current.update_from(value)
                        else:
                            setattr(self, key, value)
                else:
                    setattr(self, key, value)
    def update(self, other):
        """
        Update from either a plain dict or a NamespaceDict.
        Converts dict to NamespaceDict recursively before delegating to update_from.
        """
        if isinstance(other, NamespaceDict):
            self.update_from(other)
        elif isinstance(other, dict):
            self.update_from(to_namespace(other))
        else:
            raise TypeError(f"update expects a dict or NamespaceDict, got {type(other).__name__}")

    def copy(self):
        """Return a deep copy of this NamespaceDict."""
        def clone(obj):
            if isinstance(obj, NamespaceDict):
                new = NamespaceDict()
                for k, v in vars(obj).items():
                    setattr(new, k, clone(v))
                return new
            elif isinstance(obj, list):
                return [clone(v) for v in obj]
            else:
                return obj
        return clone(self)

    def pretty_print(self, indent=0):
        """Recursively pretty-print the NamespaceDict."""
        indent_str = "    " * indent  # 4 spaces per level

        for key, value in vars(self).items():
            if isinstance(value, (NamespaceDict, dict)):
                print(f"{indent_str}{key}:")
                if isinstance(value, dict):
                    # Convert dict to NamespaceDict-like behavior
                    for sub_key, sub_val in value.items():
                        if isinstance(sub_val, (NamespaceDict, dict)):
                            print(f"{indent_str}    {sub_key}:")
                            if isinstance(sub_val, dict):
                                NamespaceDict(**sub_val).pretty_print(indent + 2)
                            else:
                                sub_val.pretty_print(indent + 2)
                        else:
                            print(f"{indent_str}    {sub_key}: {sub_val}")
                else:
                    value.pretty_print(indent + 1)

            elif isinstance(value, list):
                print(f"{indent_str}{key}:")
                for i, item in enumerate(value):
                    if isinstance(item, (NamespaceDict, dict)):
                        print(f"{indent_str}    -")
                        if isinstance(item, dict):
                            NamespaceDict(**item).pretty_print(indent + 2)
                        else:
                            item.pretty_print(indent + 2)
                    else:
                        print(f"{indent_str}    - {item}")
            else:
                print(f"{indent_str}{key}: {value}")

def to_namespace(obj):
    if isinstance(obj, dict):
        ns = NamespaceDict()
        for k, v in obj.items():
            setattr(ns, k, to_namespace(v))
        return ns
    if isinstance(obj, list):
        return [to_namespace(v) for v in obj]
    return obj
def dict_to_ns(d):
    if isinstance(d, dict):
        ns = NamespaceDict(**{k: dict_to_ns(v) for k, v in d.items()})
        return ns
    elif isinstance(d, list):
        return [dict_to_ns(v) for v in d]
    else:
        return d

def ns_to_dict(obj):
    if isinstance(obj, dict):
        return {k: ns_to_dict(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [ns_to_dict(v) for v in obj]

    # Don't unwrap dataclasses
    if is_dataclass(obj):
        return obj

    if hasattr(obj, "__dict__"):
        return {
            k: ns_to_dict(v)
            for k, v in vars(obj).items()
        }

    return obj
def flatten_dict(d, parent_key="", sep=".", keep_path=True, no_flatten_terms: list = []):
    """
    Flatten a nested dictionary.

    keep_path=True:
      {"a": {"b": 1}} -> {"a.b": 1}

    keep_path=False:
      {"a": {"b": 1}} -> {"b": 1}
      (last key wins if collisions occur)
    """
    items = {}
    for k, v in d.items():
        key = str(k)
        if key in no_flatten_terms:
            items[key] = v
            continue
        if keep_path and parent_key:
            key = f"{parent_key}{sep}{k}"

        if isinstance(v, dict) or isinstance(v, SimpleNamespace):
            items.update(flatten_dict(v, key if keep_path else "", sep=sep, keep_path=keep_path, no_flatten_terms = no_flatten_terms))
        else:
            items[key] = v
    return items

def normalize_cfg(cfg):
    """
    Convert an OmegaConf/Hydra config into a SimpleNamespace hierarchy
    with plain Python types (lists, dicts, floats, etc.).
    """
    #  cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_dict(cfg)  else cfg
    return dict_to_ns(cfg_dict)

def sanitize_vector_param(x, K: int, like=None):
    """
    Normalize x to length K and match backend of `like`.

    - scalar/list/np/torch → vector of length K
    - crop if longer
    - pad with last value if shorter
    - output matches backend/dtype/device of `like`
    """
    if x is None:
        return None

    # First: convert to numpy for shape logic
    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    # scalar → vector
    if arr.ndim == 0:
        arr = np.full(K, float(arr))
    else:
        arr = arr.flatten()

    # adjust length
    if len(arr) > K:
        arr = arr[:K]
    elif len(arr) < K:
        pad = np.full(K - len(arr), arr[-1])
        arr = np.concatenate([arr, pad])

    # Now convert back to desired backend
    if like is None:
        return arr.astype(np.float32)

    return to_backend(arr, like)

def minmax_normalize(S):
    S_min = np.min(S)
    S_max = np.max(S)

    if S_max == S_min:
        return np.zeros_like(S)  # avoid division by zero

    return (S - S_min) / (S_max - S_min)

def to_backend(x, like):
    """Convert x to backend/dtype/device of like (torch or numpy)."""

    if torch.is_tensor(like):                         # Torch backend
        if x is None: return None
        if torch.is_tensor(x): return x.to(like.device, like.dtype)
        x = np.asarray(x)
        return torch.as_tensor(x, device=like.device, dtype=like.dtype)

    # NumPy backend
    if x is None: return None
    if torch.is_tensor(x): x = x.detach().cpu().numpy()
    x = np.asarray(x)
    tgt = like.dtype if hasattr(like, "dtype") else np.float32
    return x.astype(tgt) if x.dtype != tgt else x


def set_all_seeds(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

def sample_actions_from_policy(
    policy,
    X: torch.Tensor,
    decision_threshold: float = None,
    return_logits: bool = True,
    return_probs: bool = False,
    require_grad: bool = False,
    **kwargs,
):
    """
    If threshold is not None -> deterministic 0/1 via threshold. 
    Else: Sample actions (labels) given the current parameters, using logits-> probs as the distribtuion.
    Returns y_hat in {0,1} float tensor, shape [n], optional logits in [-inf, inf] and optional probs in [0,1]
    """

    ctx = torch.enable_grad() if require_grad else torch.no_grad()
    with ctx:
        logits = policy(X).squeeze(-1)
        probs = torch.sigmoid(logits)

        #  import ipdb;ipdb.set_trace()
        if decision_threshold is None:
            y_hat = torch.bernoulli(probs)
        else:
            y_hat = (probs >= decision_threshold).float()
        return (y_hat,) + (logits,)*return_logits + (probs,)*return_probs
# =============================================================================================================
# Sample/Population Probabiliy Computation Functions
# =============================================================================================================

def compute_sample_logprob(logits, labels, reduce="sum"):
    """
    Compute the probablity of a sample R given logits and label decisions over populations (D, N)
    This assumes  logits are shared over sample for each R or  logits are of shape (R, D, N).
    logits: (R, N) or (R, 1, N)
    labels: (D, N), (N,), or (1, D, N)
    returns:  (R,) or (R, D)
    """
    logits = torch.as_tensor(logits)
    labels = torch.as_tensor(labels, device=logits.device, dtype=logits.dtype)

    if logits.ndim == 2:
        logits = logits[:, None, :]   # (R, 1, N)

    if labels.ndim == 1:
        labels = labels[None, None, :] # (1, 1, N)
    elif labels.ndim == 2:
        labels = labels[None, :, :]    # (1, D, N)

    logits = F.binary_cross_entropy_with_logits(
        logits.expand(-1, labels.size(1), -1), # broadcasts to (R, D, N)
        labels.expand(logits.size(0), -1, -1), # broadcasts to (R, D, N)
        reduction="none"
    ).sum(dim=-1)  # (R,D), sum along the sample dim N.

    if reduce == "sum":
        return logits.sum(dim=-1)        # (R)
    elif reduce == "mean":
        return logits.mean(dim=-1)       # (R)
    else:
        return logits                    # (R, D)

#--------------------------------------------------------------

def compute_sample_distribution(logits, labels, reduce="sum"):
    """
    Returns (R,) the total probablity of each sample, normalized to a distribution over the samples
    For example, R, N sample logits over  D, N population decisions, will yield a distribution
    D_R = [p_0, p_1, ... p_R]
    This assumes that each population d in D shares features; otherwise logits must be (R, D, N).
    """
    logprobs = compute_sample_logprob(logits, labels, reduce=reduce)
    return logprobs / logprobs.sum(axis=-1)

#--------------------------------------------------------------

def trajectory_logprob(logits, labels, return_numpy=False, require_grad: bool = False ):
    """
    Compute log-probability of trajectories under Bernoulli logits.

    Args
    ----
    logits : (R,S) tensor/array or list of tensors
    labels : (R,S) or (S,)
    return_numpy : return numpy instead of torch

    Returns
    -------
    log_probs : (R,) log p(y | logits) for each rollout
    """

    ctx = torch.enable_grad() if require_grad else torch.no_grad()
    with ctx:
        # ---- convert logits ----
        if isinstance(logits, list):
            logits = torch.stack([
                l if torch.is_tensor(l) else torch.tensor(l)
                for l in logits
            ])
        elif not torch.is_tensor(logits):
            logits = torch.tensor(logits)

        R, S = logits.shape

        # ---- convert labels ----
        if isinstance(labels, list):
            labels = torch.stack([
                l if torch.is_tensor(l) else torch.tensor(l)
                for l in labels
            ])
        elif not torch.is_tensor(labels):
            labels = torch.tensor(labels)


        #  import ipdb;ipdb.set_trace()
        if labels.ndim == 1:  # (S,) -> repeat
            labels = labels.unsqueeze(0).expand(R, -1)

        if labels.shape != (R, S):
            raise ValueError(f"labels must be (R,S) or (S,), got {labels.shape}")
        labels = labels.to(logits.device)
        labels = labels.float()

        # ---- BCE with logits gives -log p(y|logits) elementwise ----
        neg_logprob = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )  # (R,S)

        logprob = -neg_logprob.sum(dim=1)  # trajectory log prob

        if return_numpy:
            return logprob.detach().cpu().numpy()

        return logprob

#--------------------------------------------------------------

def rollout_scores(logits, labels, mode="logprob", reduce="sum", require_grad=True):
    """
    logits: (R, N)
    labels: (R, N)
    returns: (R,)
    """
    ctx = torch.enable_grad() if require_grad else torch.no_grad()

    with ctx:
        logits = torch.as_tensor(logits)
        labels = torch.as_tensor(labels, device=logits.device, dtype=logits.dtype)

        if mode in ["logprob", "prob"]:
            logp = -F.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            )  # (R,N)

            logp = logp.sum(dim=-1) if reduce == "sum" else logp.mean(dim=-1)
            return logp if mode == "logprob" else logp.exp()

        elif mode == "01":
            preds = (logits > 0).to(labels.dtype)
            acc = (preds == labels).float()
            return acc.mean(dim=-1) if reduce == "mean" else acc.sum(dim=-1)

        else:
            raise ValueError(mode)

#--------------------------------------------------------------

def ensemble_scores(
    logits, labels,
    mode="logprob",      # "logprob" | "prob" | "01"
    reduce="sum",
    require_grad=True,
    temperature = 1.0,
    normalize=True,
):
    ctx = torch.enable_grad() if require_grad else torch.no_grad()

    if temperature == 0.:
        print("Temperature 0.0 will lead to div by 0. Setting temperature to 1.0")
        temperature = 1.

    with ctx:
        logits = torch.as_tensor(logits)
        labels = torch.as_tensor(labels, device=logits.device, dtype=logits.dtype)

        logits = logits[:, None, :]      # (M,1,N)
        labels = labels[None, :, :]      # (1,D,N)

        if mode in ["logprob", "prob"]:
            logp = -F.binary_cross_entropy_with_logits(
                logits.expand(-1, labels.size(1), -1),
                labels.expand(logits.size(0), -1, -1),
                reduction="none",
            )  # (M,D,N)

            logp = logp.sum(dim=-1) if reduce == "sum" else logp.mean(dim=-1)  # (M,D)

            #  if normalize:
            #      # subtract logsumexp over D → stable
            #      logp = logp - torch.logsumexp(logp, dim=1, keepdim=True)

            return logp if mode == "logprob" else torch.softmax((logp / temperature), dim=1)

        elif mode == "01":
            preds = (logits > 0).to(labels.dtype)
            acc = (preds == labels).float()
            return acc.mean(dim=-1) if reduce == "mean" else acc.sum(dim=-1)

        else:
            raise ValueError(mode)
