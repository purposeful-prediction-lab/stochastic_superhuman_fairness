import json
from pathlib import Path
from omegaconf import ListConfig, DictConfig, OmegaConf


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
