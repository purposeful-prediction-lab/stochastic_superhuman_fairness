from omegaconf import OmegaConf
from types import SimpleNamespace
import os, random, numpy as np, torch

class NamespaceDict(SimpleNamespace):
    """A SimpleNamespace with dict-like get() method and repr that hides internals."""

    def get(self, key, default=None):
        return getattr(self, key, default)

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

def dict_to_ns(d):
    if isinstance(d, dict):
        ns = NamespaceDict(**{k: dict_to_ns(v) for k, v in d.items()})
        return ns
    elif isinstance(d, list):
        return [dict_to_ns(v) for v in d]
    else:
        return d

def normalize_cfg(cfg):
    """
    Convert an OmegaConf/Hydra config into a SimpleNamespace hierarchy
    with plain Python types (lists, dicts, floats, etc.).
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    return dict_to_ns(cfg_dict)


def set_all_seeds(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
