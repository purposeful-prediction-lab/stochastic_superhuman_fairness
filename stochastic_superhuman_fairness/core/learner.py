# stochastic_superhuman_fairness/core/learner.py
from __future__ import annotations
import torch
from typing import Dict, Any
from copy import deepcopy

from stochastic_superhuman_fairness.core.models.logistic import LogisticRegressionModel
from stochastic_superhuman_fairness.core.models.mlp import MLPModel
from stochastic_superhuman_fairness.core.models.ppo import PPOModel
from stochastic_superhuman_fairness.core.models import logistic, mlp, ppo
from stochastic_superhuman_fairness.core.fairness import subdominance
from stochastic_superhuman_fairness.core.utils import normalize_cfg, dict_to_ns
import torch, numpy as np

MODEL_REGISTRY = {
    "logistic": logistic.LogisticRegressionModel,
    "mlp": mlp.MLPModel,
    "ppo": ppo.PPOModel,
}

DEFAULT_PHASE_CFG = {
    "epochs": 10,
    "lr": 1e-3,
    "batch_size": 32,
    "clip_range": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.0,
}




class Learner:
    """
    Main orchestrator for fairness-aware training across multiple models.

    Responsibilities:
      - Instantiate model(s) from config
      - Execute training schedule (sequential phases)
      - Handle switching between models (policy/value transfer)
      - Maintain logs per phase and epoch
    """

    def __init__(self, cfg, demonstrator, logger=None):
        self.cfg = normalize_cfg(cfg)
        self.demo = demonstrator
        self.logger = logger
        self.model = None
        self.phase_idx = 0
        self.device = getattr(self.cfg, "device", 'cpu')
        self.global_step = 0
        # schedule: list of dicts, each with 'algo', 'epochs', and model-specific params
        # Validate + normalize training schedule
        self.schedule = self._validate_schedule(self.cfg.learner.schedule)

    # ------------------------------------------------------------------
    def _validate_schedule(self, schedule):
        """Fill missing keys in schedule entries with defaults."""
        validated = []
        if not schedule:
            raise ValueError("Training schedule missing in config.")

        for i, phase in enumerate(schedule):
            entry = {**DEFAULT_PHASE_CFG, **phase}
            if "algo" not in entry:
                raise ValueError(f"Missing 'algo' in schedule entry {i}")
            entry["algo"] = entry["algo"].lower()
            entry['device'] = self.device
            entry['seed'] = self.cfg.seed
            validated.append(dict_to_ns(entry))
        return validated
    # ----------------------------------------------------------
    def _transfer_parameters(self, old_model, new_model):
        """Transfer policy/value weights between compatible models."""
        if not hasattr(old_model, "get_state_dict") or not hasattr(new_model, "get_state_dict"):
            return
        old_sd, new_sd = old_model.get_state_dict(), new_model.get_state_dict()

        if "policy" in old_sd and "policy" in new_sd:
            self._safe_load(new_model.policy, old_sd["policy"], "policy")

        if "value" in old_sd and "value" in new_sd and old_sd["value"] and new_sd["value"]:
            self._safe_load(new_model.value, old_sd["value"], "value")

        print("🔄 Transferred compatible weights.")
        return new_model

    # ----------------------------------------------------------
    def _log(self, stats: Dict[str, Any]):
        """Pass metrics to logger (if available)."""
        if self.logger is not None:
            self.logger.log(stats)
        else:
            print(stats)

    # ----------------------------------------------------------
    def run(self):
        """Execute the full training schedule."""
        for phase_idx, phase_cfg in enumerate(self.schedule):
            scfg = getattr(self.cfg, 'subdominance', {})
            algo = phase_cfg["algo"]
            epochs = phase_cfg.get("epochs", 1)
            batch_size = phase_cfg.get("batch_size", 1)
            subdom_type = getattr(scfg, "type", "standard").lower()
            print(f"\n🚀 Phase {phase_idx+1}/{len(self.schedule)} — {algo.upper()} ({epochs} epochs)")

            # Initialize or switch model
            #  model = self._init_model(algo, phase_cfg)

            #  import ipdb;ipdb.set_trace()
            self.switch_algo(algo, phase_cfg)
            #  if self.model is not None:
                #  model = self._transfer_params(self.model, model)
            #  self.model = model

            # Log model transition
            self.logger.log_transition(phase_idx, algo)

            # Determine evaluation frequency
            eval_freq = phase_cfg.get(
                "eval_freq",
                getattr(self.cfg['global'], "eval_freq", 5)
            )

            # ----------------------
            # Phase training loop
            # ----------------------
            for ep in range(epochs):
                stats_train = self.model.train_one_epoch(self.demo, batch_size = batch_size, subdom_type = subdom_type)
                stats_train.update({
                    "epoch": ep,
                    "phase": phase_idx,
                    "algo": algo,
                    "stage": "train"
                })
                self._log(stats_train)

                # ---- Conditional evaluation ----
                if (ep + 1) % eval_freq == 0:
                    eval_stats = self.model.evaluate(self.demo)
                    eval_stats.update({
                        "epoch": ep,
                        "phase": phase_idx,
                        "algo": algo,
                        "stage": "eval"
                    })
                    self._log(eval_stats)

            # ----------------------
            # Save checkpoint per phase
            # ----------------------
            if self.logger:
                self.logger.save_checkpoint(self.model, phase_idx, algo, cfg=phase_cfg)

        print("\n✅ Training completed.")
    # ----------------------------------------------------------
    def get_model(self):
        """Return current trained model."""
        return self.model

    def _init_model(self, algo: str, phase_cfg: dict):
        """Create model for current phase with merged global config."""
        if algo not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {algo}")

        # Merge per-phase config with global fields
        merged_cfg = dict(phase_cfg)
        # attach global fairness/subdominance/metrics if present
        if hasattr(self.cfg, "metrics"):
            merged_cfg["metrics"] = self.cfg.metrics
        if hasattr(self.cfg, "subdominance"):
            merged_cfg["subdominance"] = self.cfg.subdominance
        if hasattr(self.cfg, "global"):
            merged_cfg["global"] = self.cfg['global']

        model_cls = MODEL_REGISTRY[algo]
        return model_cls(merged_cfg, self.demo)

    def switch_algo(self, algo_name, phase_cfg):
        """Initialize a new model with parameter transfer."""
        algo_name = algo_name.lower()
        new_model = self._init_model(algo_name, phase_cfg)
        if hasattr(new_model, "to"):
            new_model.to(self.device)

        if self.model is not None:
            self._transfer_parameters(self.model, new_model)

        self.model = new_model
        self.current_algo = algo_name
        print(f"🔁 Switched to algorithm: {algo_name.upper()}")

    # ------------------------------------------------------------------
    def _safe_load(self, module, state_dict, name):
        """Safely load overlapping parameters by name."""
        own_state = module.state_dict()
        matched = {k: v for k, v in state_dict.items() if k in own_state and v.shape == own_state[k].shape}
        own_state.update(matched)
        module.load_state_dict(own_state)
        print(f"   → {len(matched)} {name} layers transferred")


