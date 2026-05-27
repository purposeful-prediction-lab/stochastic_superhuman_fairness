import importlib
import inspect
from dataclasses import is_dataclass, fields
from pathlib import Path
from stochastic_superhuman_fairness.core.utils import NamespaceDict, dict_to_ns, ns_to_dict
from collections.abc import Mapping





def snake_to_camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_") if part)

def import_module_from_file(py_file: Path):
    module_name = f"_cfg_dc_{py_file.stem}"
    spec = importlib.util.spec_from_file_location(module_name, py_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {py_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dataclass_registry(dataclass_dir="core/dataclasses"):
    """
    Imports all dataclass files once and returns:
        {"CouplingConfig": CouplingConfig, ...}
    """
    registry = {}
    path = Path(dataclass_dir)

    if not path.exists():
        return registry

    for py_file in path.glob("*.py"):
        if py_file.name.startswith("_") or "utils" in py_file.name:
            continue

        module = import_module_from_file(py_file)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if is_dataclass(obj):
                registry[obj.__name__] = obj

    return registry

def normalize_yaml_key_base(yaml_key: str) -> str:
    """
    Converts:
        coupling_cfg     -> coupling
        coupling_config  -> coupling
        something_else   -> something_else
    """
    key = yaml_key.lower()

    if key.endswith("_cfg"):
        key = key[:-4]
    elif key.endswith("_config"):
        key = key[:-7]

    return key


def dataclass_for_yaml_key(
    yaml_key,
    registry,
    *,
    suffixes=("Config", "Cfg"),
):
    base_key = normalize_yaml_key_base(yaml_key)
    base_camel = snake_to_camel(base_key)

    # Try: SomethingConfig, SomethingCfg
    for suffix in suffixes:
        cls = registry.get(f"{base_camel}{suffix}")
        if cls is not None:
            return cls

    # Optional fallback: exact match without suffix
    cls = registry.get(base_camel)
    if cls is not None:
        return cls

    return None

def maybe_build_dataclass(yaml_key, value, registry, *, strict=False, suffixes=('Config', 'Cfg')):
    #  if 'coupl' in yaml_key:
    #      import ipdb;ipdb.set_trace()
    cls = dataclass_for_yaml_key(yaml_key, registry, suffixes=suffixes)

    if cls is None:
        return value

    value = ns_to_dict(value)

    field_names = {f.name for f in fields(cls)}
    extra = set(value.keys()) - field_names

    if strict and extra:
        raise ValueError(f"Extra keys for {cls.__name__}: {sorted(extra)}")

    kwargs = {k: v for k, v in value.items() if k in field_names}
    return cls(**kwargs)

def convert_dataclass_sections_recursive(
    obj,
    registry,
    *,
    suffixes=("Config", "Cfg"),
    strict=False,
):
    obj = ns_to_dict(obj)

    if isinstance(obj, Mapping):
        out = {}

        for key, value in obj.items():
            # First recursively convert children
            converted_value = convert_dataclass_sections_recursive(
                value,
                registry,
                strict=strict,
            )

            # Then try converting this key itself
            if isinstance(converted_value, Mapping):
                converted_value = maybe_build_dataclass(
                    key,
                    converted_value,
                    registry,
                    strict=strict,
                    suffixes=suffixes,
                )

            out[key] = converted_value

        return out

    if isinstance(obj, list):
        return [
            convert_dataclass_sections_recursive(v, registry, strict=strict, suffixes=suffixes)
            for v in obj
        ]

    return obj

def convert_matching_sections_to_dataclasses(
    cfg_entry,
    *,
    dataclass_dir="core/dataclasses",
    strict=False,
    suffixes=("Config", "Cfg"),
):
    registry = load_dataclass_registry(dataclass_dir)

    out = convert_dataclass_sections_recursive(
        cfg_entry,
        registry,
        strict=strict,
        suffixes=suffixes,
    )

    return dict_to_ns(out)
