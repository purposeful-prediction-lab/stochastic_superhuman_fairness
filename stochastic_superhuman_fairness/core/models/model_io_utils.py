import io
import json
import os
import zipfile
import torch
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple
from stochastic_superhuman_fairness.core.utils import (
        NamespaceDict,
        to_namespace,
        )
from stochastic_superhuman_fairness.core.models.registry import MODEL_REGISTRY
from stochastic_superhuman_fairness.core.demonstrator import Demonstrator
from stochastic_superhuman_fairness.core.utils import NamespaceDict, dict_to_ns

try:
    import yaml
except Exception:
    yaml = None

try:
    import tomllib  # py3.11+
except Exception:
    tomllib = None

#  # ---------------------------------------------------------------------
#  # Namespace helpers (match training-time cfg behavior)
#  # ---------------------------------------------------------------------
#
#  class NamespaceDict(SimpleNamespace):
#      def get(self, key, default=None):
#          return getattr(self, key, default)
#
#
#  def to_namespace(obj):
#      if isinstance(obj, dict):
#          ns = NamespaceDict()
#          for k, v in obj.items():
#              setattr(ns, k, to_namespace(v))
#          return ns
#      if isinstance(obj, list):
#          return [to_namespace(v) for v in obj]
#      return obj
#

# ---------------------------------------------------------------------
# Safe partial loading (formerly Learner._safe_load)
# ---------------------------------------------------------------------

def safe_load_state_dict(module: torch.nn.Module,
                         state_dict: Dict[str, torch.Tensor],
                         name: str = "model") -> Dict[str, torch.Tensor]:
    """
    Load only overlapping parameters with matching shapes.
    Returns the matched sub-dict.
    """
    own_state = module.state_dict()
    matched = {
        k: v for k, v in state_dict.items()
        if k in own_state and v.shape == own_state[k].shape
    }
    own_state.update(matched)
    module.load_state_dict(own_state)
    print(f"   → {len(matched)} {name} layers transferred")
    return matched


# ---------------------------------------------------------------------
# Zip / config helpers
# ---------------------------------------------------------------------

CONFIG_NAMES = ("config.json", "config.yaml", "config.yml", "config.toml")
WEIGHT_NAMES = ("checkpoint.pt", "model_state.pt", "state_dict.pt", "model.pt", "weights.pt")
DIST_NAMES   = ("dist_state.pt", "distribution.pt", "posterior.pt")


def _detect_one(names, candidates):
    for c in candidates:
        if c in names:
            return c
    for n in names:
        if os.path.basename(n) in candidates:
            return n
    return None


def _parse_config(text: str, filename: str) -> Dict[str, Any]:
    fn = filename.lower()
    if fn.endswith(".json"):
        return json.loads(text)
    if fn.endswith((".yml", ".yaml")):
        if yaml is None:
            raise RuntimeError("YAML config but pyyaml not installed.")
        return yaml.safe_load(text)
    if fn.endswith(".toml"):
        if tomllib is None:
            raise RuntimeError("TOML config but tomllib not available.")
        return tomllib.loads(text)
    raise RuntimeError(f"Unsupported config format: {filename}")


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    """
    Handles:
      - bare state_dict
      - Logger checkpoint dicts
    """
    if isinstance(obj, dict):
        for k in (
            "model_state",
            "state_dict",
            "model",
            "policy",
            "policy_state_dict",
            "model_state_dict",
        ):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        if any(torch.is_tensor(v) for v in obj.values()):
            return obj
    raise RuntimeError("Unrecognized weights format.")

# ---------------------------------------------------------------------
# Schedule formation and selection from CFG
# ---------------------------------------------------------------------


def validate_and_select_phase_cfg(cfg_ns, *, phase_idx: int, device: str):
    """
    Returns (phase_cfg_ns, algo_str).

    Merges cfg.learner.default into cfg.learner.schedule[phase_idx],
    then injects device + seed, and lowercases algo.
    """
    # cfg_ns is NamespaceDict (attribute access) produced from config.json/yaml
    learner = getattr(cfg_ns, "learner", None)
    if learner is None:
        raise ValueError("Config missing 'learner' section.")

    schedule = getattr(learner, "schedule", None)
    if not schedule:
        raise ValueError("Training schedule missing: cfg.learner.schedule is empty.")

    if not (0 <= phase_idx < len(schedule)):
        raise IndexError(f"phase_idx={phase_idx} out of range; schedule has {len(schedule)} entries.")

    default_cfg = getattr(learner, "default", {}) or {}

    # Merge: default first, then phase overrides
    entry = dict_to_ns(default_cfg.copy())
    dict_to_ns(entry).update_from(schedule[phase_idx])

    if "algo" not in entry:
        raise ValueError(f"Missing 'algo' in schedule entry {phase_idx}")

    entry["algo"] = str(entry["algo"]).lower()
    entry["device"] = device

    # seed lives at top-level in your config
    if hasattr(cfg_ns, "seed"):
        entry["seed"] = cfg_ns.seed

    return dict_to_ns(entry), entry["algo"]
def validate_schedule(schedule, *, cfg, device, default_learner_cfg=None):
    """
    Returns a list of validated phase configs (NamespaceDict),
    where each phase is default_learner_cfg merged with the phase override.
    """

    if default_learner_cfg is None:
        default_learner_cfg = cfg.get("learner", {}).get("default", {})

    if not schedule:
        raise ValueError("Training schedule missing in config.")

    validated = []
    for i, phase in enumerate(schedule):
        entry = dict_to_ns(default_learner_cfg.copy())
        dict_to_ns(entry).update_from(phase)

        if "algo" not in entry:
            raise ValueError(f"Missing 'algo' in schedule entry {i}")

        entry["algo"] = entry["algo"].lower()
        entry["device"] = device
        entry["seed"] = cfg.seed if hasattr(cfg, "seed") else cfg.get("seed", None)

        validated.append(dict_to_ns(entry))

    return validated
# ---------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------
def load_model_from_archive(
    archive_path: str,
    device: str = "cpu",
    phase_idx: int = 0,
    strict: bool = False,
    use_safe_load: bool = True,
):
    # -------------------------------------------------
    # Open archive (NEW FORMAT ONLY)
    # -------------------------------------------------
    with zipfile.ZipFile(archive_path, "r") as zf:
        names = zf.namelist()

        if "model_state.pt" not in names or "config.json" not in names:
            raise RuntimeError(
                f"Invalid archive format. Expected model_state.pt + config.json.\nContents: {names}"
            )

        cfg_dict = dict_to_ns(
            json.loads(zf.read("config.json").decode("utf-8"))
        )

        state_dict = torch.load(
            io.BytesIO(zf.read("model_state.pt")),
            map_location=device,
            weights_only=True,
        )

        dist_state = None
        if "dist_state.pt" in names:
            dist_state = torch.load(
                io.BytesIO(zf.read("dist_state.pt")),
                map_location=device,
                weights_only=True,
            )

    # -------------------------------------------------
    # Resolve phase + algo (shared with Learner)
    # -------------------------------------------------
    phase_cfg, algo = validate_and_select_phase_cfg(
        cfg_dict, phase_idx=phase_idx, device=device
    )

    if algo not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model type '{algo}'. Available: {list(MODEL_REGISTRY)}")

    # -------------------------------------------------
    # Build demonstrator
    # -------------------------------------------------
    demo = Demonstrator(cfg_dict)

    # Attach global fields (mirrors learner behavior)
    if hasattr(cfg_dict, "metrics"):
        phase_cfg["metrics"] = cfg_dict.metrics
    if hasattr(cfg_dict, "subdominance"):
        phase_cfg["subdominance"] = cfg_dict.subdominance
    if hasattr(cfg_dict, "global"):
        phase_cfg["global"] = getattr(cfg_dict, "global")

    # -------------------------------------------------
    # Instantiate model
    # -------------------------------------------------
    model = MODEL_REGISTRY[algo](phase_cfg, demo).to(device)
    model.eval()

    # -------------------------------------------------
    # Load parameters
    # -------------------------------------------------
    if use_safe_load:
        safe_load_state_dict(model, state_dict, name="model")
    else:
        model.load_state_dict(state_dict, strict=strict)

    if dist_state is not None and hasattr(model, "load_dist_state"):
        model.load_dist_state(dist_state, device=device)

    return model, cfg_dict, demo
