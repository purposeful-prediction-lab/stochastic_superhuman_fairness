import json
from pathlib import Path
from omegaconf import ListConfig, DictConfig, OmegaConf
from types import SimpleNamespace
import numpy as np
import torch
from dataclasses import is_dataclass, asdict
from collections.abc import Mapping
from omegaconf import OmegaConf, DictConfig, ListConfig


# Pretty Print and Formatting Functions
# =====================================================================================================

def _to_dict(obj):
    """Convert namespaces / OmegaConf to plain dict recursively."""
    try:
        # OmegaConf support (optional)
        from omegaconf import OmegaConf
        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
    except Exception:
        pass

    if isinstance(obj, SimpleNamespace):
        return vars(obj)

    return obj

def filter_dict_exclude(obj, exclude_keys):
    """
    Recursively remove keys in exclude_keys from nested dict / namespace.
    """
    obj = _to_dict(obj)

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in exclude_keys:
                continue
            out[k] = filter_dict_exclude(v, exclude_keys)
        return out

    if isinstance(obj, list):
        return [filter_dict_exclude(v, exclude_keys) for v in obj]

    if isinstance(obj, tuple):
        return tuple(filter_dict_exclude(v, exclude_keys) for v in obj)

    return obj


def format_nested(obj, indent=0):
    obj = _to_dict(obj)
    sp = "  " * indent

    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            v = _to_dict(v)
            if isinstance(v, (dict, list, tuple)):
                lines.append(f"{sp}{k}:")
                lines.append(format_nested(v, indent + 1))
            else:
                lines.append(f"{sp}{k}: {v}")
        return "\n".join(lines)

    if isinstance(obj, (list, tuple)):
        lines = []
        for v in obj:
            v = _to_dict(v)
            if isinstance(v, (dict, list, tuple)):
                lines.append(f"{sp}-")
                lines.append(format_nested(v, indent + 1))
            else:
                lines.append(f"{sp}- {v}")
        return "\n".join(lines)

    return f"{sp}{obj}"


def pretty_dict_print(obj):
    print(format_nested(obj))

def format_dict_text(obj, indent=0):
    sp = "  " * indent

    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            if isinstance(v, (dict, list, tuple)):
                lines.append(f"{sp}{k}:")
                lines.append(format_dict_text(v, indent + 1))
            else:
                lines.append(f"{sp}{k}: {v}")
        return "\n".join(lines)

    if isinstance(obj, (list, tuple)):
        lines = []
        for v in obj:
            if isinstance(v, (dict, list, tuple)):
                lines.append(f"{sp}-")
                lines.append(format_cfg_text(v, indent + 1))
            else:
                lines.append(f"{sp}- {v}")
        return "\n".join(lines)

    return f"{sp}{obj}"
#=========================================================================================================

def save_dict_text(cfg, path="cfg_pretty.txt"):
    txt = format_dict_text(cfg)
    path = Path(path)
    path.write_text(txt, encoding="utf-8")
    return path

def classify_arg(arg: str):
    p = Path(arg)

    if p.is_absolute():
        return "absolute_path"
    elif p.parent != Path("."):
        return "relative_path"
    else:
        return "name"

def get_base_path(path_str: str) -> str:
    return str(Path(path_str).parent)

def split_path(path_str: str):
    p = Path(path_str)
    return str(p.parent), p.name

def to_pure(obj):
    """Recursively convert OmegaConf containers (DictConfig, ListConfig) into native Python types."""
    if isinstance(obj, DictConfig):
        return {k: to_pure(v) for k, v in obj.items()}
    elif isinstance(obj, ListConfig):
        return [to_pure(v) for v in obj]
    elif isinstance(obj, (list, tuple)):
        return [to_pure(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: to_pure(v) for k, v in obj.items()}
    else:
        return obj

#
# JSON IO
#


def to_json_serializable(obj):
    """
    Recursively convert object into JSON-serializable form.

    Handles:
    - dict / Mapping / NamespaceDict
    - OmegaConf DictConfig / ListConfig
    - dataclasses
    - numpy arrays/scalars
    - torch tensors
    - lists / tuples / sets
    """

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # OmegaConf
    if OmegaConf is not None and isinstance(obj, (DictConfig, ListConfig)):
        obj = OmegaConf.to_container(
            obj,
            resolve=True,
            throw_on_missing=False,
        )
        return to_json_serializable(obj)

    # Dataclass
    if is_dataclass(obj):
        return to_json_serializable(asdict(obj))

    # Dict-like
    if isinstance(obj, Mapping):
        return {
            str(k): to_json_serializable(v)
            for k, v in obj.items()
        }

    # Sequences
    if isinstance(obj, (list, tuple, set)):
        return [to_json_serializable(v) for v in obj]

    # NumPy
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    # Torch
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    # Namespace / simple object
    if hasattr(obj, "__dict__"):
        return to_json_serializable(vars(obj))

    return str(obj)
def safe_json_dump(data, path, indent=2):
    """
    Dump to JSON, automatically converting OmegaConf containers
    into native Python types.
    """
    pure = to_pure(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(pure, f, indent=indent)


def safe_json_load(path, as_omegaconf=False):
    """
    Load JSON files saved with `safe_json_dump`.

    Args:
        path (str | Path): Path to the JSON file.
        as_omegaconf (bool): If True, wrap the result into an OmegaConf DictConfig.

    Returns:
        dict | OmegaConf.DictConfig: Loaded data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if as_omegaconf:
        return OmegaConf.create(data)
    return data


def load_metrics_jsonl(path, return_df=False):
    """
    Load a metrics.jsonl file.

    Args:
        path (str or Path): path to metrics.jsonl
        return_df (bool): if True, also return pandas DataFrame

    Returns:
        logs (list[dict]) 
        optionally: (logs, df)
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No metrics file found at {path}")

    logs = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                # skip corrupted/partial lines
                continue

    if return_df:
        import pandas as pd
        df = pd.DataFrame(logs)
        return logs, df

    return logs

# ========================================================================
# Dataclass Utils
def dataclass_from_dict(dc_cls, d):
    fields = dc_cls.__annotations__.keys()
    return dc_cls(**{k: d[k] for k in fields if k in d})
