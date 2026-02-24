from omegaconf import OmegaConf
from types import SimpleNamespace
import os, random, numpy as np, torch
import torch

from sklearn.linear_model import LogisticRegression

@torch.no_grad()
def sample_binary_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """
    probs: tensor in [0,1], any shape
    returns: {0,1} float tensor of same shape
    """
    return torch.bernoulli(probs).float()

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
    """
    Recursively convert NamespaceDict / SimpleNamespace / objects
    with __dict__ into plain Python dicts.
    """
    if isinstance(obj, dict):
        return {k: ns_to_dict(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [ns_to_dict(v) for v in obj]

    # NamespaceDict / SimpleNamespace / similar
    if hasattr(obj, "__dict__"):
        return {
            k: ns_to_dict(v)
            for k, v in vars(obj).items()
        }

    return obj

def flatten_dict(d, parent_key="", sep=".", keep_path=True):
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
        if keep_path and parent_key:
            key = f"{parent_key}{sep}{k}"

        if isinstance(v, dict) or isinstance(v, SimpleNamespace):
            items.update(flatten_dict(v, key if keep_path else "", sep=sep, keep_path=keep_path))
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

def minmax_normalize(S):
    S_min = np.min(S)
    S_max = np.max(S)

    if S_max == S_min:
        return np.zeros_like(S)  # avoid division by zero

    return (S - S_min) / (S_max - S_min)

def set_all_seeds(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_logistic_model(cfg, seed=None):
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

    C = float(rng.lognormal(mean=0.0, sigma=1.0))  # wide variability
    max_iter = int(rng.integers(0,10))

    l1_ratio = None
    if penalty == "elasticnet":
        l1_ratio = rng.uniform(0.0, 1.0)

    lr = LogisticRegression(
        solver=solver,
        penalty=penalty,
        C=C,
        max_iter=max_iter,
        l1_ratio=l1_ratio,
        n_jobs=int(getattr(cfg, "lr_n_jobs", 1)),
        random_state=rng.integers(0, 10_000),
    )

    return lr
