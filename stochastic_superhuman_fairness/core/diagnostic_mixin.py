# stochastic_superhuman_fairness/core/diagnostic_mixin.py

from pathlib import Path
import numpy as np

from stochastic_superhuman_fairness.core.utils_io import safe_json_dump, safe_json_load, to_pure
from stochastic_superhuman_fairness.core.fairness.fairness_metrics import (
    compute_fairness_features,
    zero_one_loss,
)
from stochastic_superhuman_fairness.core.diagnostic_sets import (
    make_semi_random_diagnostic_split,
    make_pareto_clean_diagnostic_split,
)


class DiagnosticDemosMixin:
    def _diagnostic_defaults(self):
        dcfg = self.cfg.demonstrator
        return {
            "train_frac": float(getattr(dcfg, "diagnostic_train_frac", 0.5)),
            "random_mult": float(getattr(dcfg, "diagnostic_random_mult", 0.5)),
            "replace_train": bool(getattr(dcfg, "diagnostic_replace_train", False)),
            "p_one": float(getattr(dcfg, "diagnostic_p_one", 0.5)),
        }

    def _build_demo_base_name(self):
        cfg = self.cfg.demonstrator
        demotype = getattr(cfg, "demotype", "partition")
        if demotype == "fullset":
            demotype = "lrdecisions"

        extra = f"_type{demotype}"
        if demotype == "lrdecisions":
            n_models = getattr(cfg, "n_models", None)
            subset_ratio = getattr(cfg, "subset_ratio", None)
            subset_size = getattr(cfg, "subset_size", None)

            if n_models is not None:
                extra += f"_n{int(n_models)}"
            if subset_ratio is not None:
                extra += f"_sr{float(subset_ratio):.3g}"
            if subset_size is not None:
                extra += f"_ss{int(subset_size)}"

        return (
            f"{cfg.dataset}_demo{extra}"
            f"_size{cfg.demo_size}"
            f"_glob{cfg.compute_global}"
            f"_norm{cfg.normalize}"
            f"_sample{self.sample_id}"
        )

    def _build_diagnostic_name(
        self,
        split: str,
        mode: str,
        train_frac: float,
        random_mult: float,
        replace_train: bool,
        p_one: float,
    ):
        base = self._build_demo_base_name()
        return (
            f"{base}"
            f"_diag{split}"
            f"_mode{mode}"
            f"_tf{train_frac:.3g}"
            f"_rm{random_mult:.3g}"
            f"_rt{replace_train}"
            f"_p1{p_one:.3g}"
        )

    def _save_bundle(self, out: dict, np_path: Path, meta_path: Path):
        save_format = getattr(self.cfg.demonstrator, "save_format", "separate")
        meta_pure = to_pure(out["metadata"])

        if save_format == "zip":
            np.savez_compressed(np_path, **out)
        else:
            np.save(np_path, out)
            safe_json_dump(meta_pure, meta_path)

    def _load_bundle(self, np_path: Path, meta_path: Path):
        save_format = getattr(self.cfg.demonstrator, "save_format", "separate")

        if save_format == "zip":
            out = np.load(np_path, allow_pickle=True)
            return {
                "train_demos": out["train_demos"].tolist() if "train_demos" in out else None,
                "eval_demos": out["eval_demos"].tolist() if "eval_demos" in out else None,
                "fairness": out["fairness"].tolist() if "fairness" in out else None,
                "metadata": out["metadata"].tolist() if "metadata" in out else {},
            }

        out = np.load(np_path, allow_pickle=True).item()
        if "metadata" not in out and meta_path.exists():
            out["metadata"] = safe_json_load(meta_path)
        return out

    def _load_or_create_diagnostic_split(self, base_demos, split: str, mode: str):
        params = self._diagnostic_defaults()

        name = self._build_diagnostic_name(
            split=split,
            mode=mode,
            train_frac=params["train_frac"],
            random_mult=params["random_mult"],
            replace_train=params["replace_train"],
            p_one=params["p_one"],
        )

        save_format = getattr(self.cfg.demonstrator, "save_format", "separate")
        np_path = self.diagnostics_dir / (
            f"{name}.npz" if save_format == "zip" else f"{name}.npy"
        )
        meta_path = self.diagnostics_dir / f"{name}_meta.json"

        overwrite = bool(getattr(self.cfg.demonstrator, "overwrite", False))

        if np_path.exists() and not overwrite:
            print(f"📂 Loading cached diagnostic {split} demos from {np_path}")
            out = self._load_bundle(np_path, meta_path)
            demos = out["train_demos"] if split == "train" else out["eval_demos"]
            if demos is not None:
                return demos

        seed_base = int(getattr(self.cfg, "seed", 0))
        seed = seed_base if split == "train" else seed_base + 10000

        if mode == "semi_random":
            diag_demos = make_semi_random_diagnostic_split(
                demos=base_demos,
                metrics=self.cfg.demonstrator.metrics,
                compute_fairness_features=compute_fairness_features,
                zero_one_loss=zero_one_loss,
                train_frac=params["train_frac"],
                random_mult=params["random_mult"],
                replace_train=params["replace_train"],
                p_one=params["p_one"],
                seed=seed,
            )

        elif mode == "pareto_clean":
            diag_demos = make_pareto_clean_diagnostic_split(
                base_demo=base_demos[0],
                metrics=self.cfg.demonstrator.metrics,
                compute_fairness_features=compute_fairness_features,
                zero_one_loss=zero_one_loss,
                n_demos=int(getattr(self.cfg.demonstrator, "pareto_clean_n_demos", 7)),
                seed=seed,
            )

        else:
            raise ValueError(
                f"Unknown diagnostic mode '{mode}'. "
                "Expected False, 'semi_random', or 'pareto_clean'."
            )

        fairness = self._compute_fairness(diag_demos)

        meta = {
            **(self.meta if hasattr(self, "meta") and self.meta is not None else {}),
            "is_diagnostic": True,
            "diagnostic_mode": mode,
            "diagnostic_split": split,
            "n_base_demos": len(base_demos),
            "n_diagnostic_demos": len(diag_demos),
        }

        out = {
            "train_demos": diag_demos if split == "train" else None,
            "eval_demos": diag_demos if split == "eval" else None,
            "fairness": fairness,
            "metadata": meta,
        }

        self._save_bundle(out, np_path, meta_path)
        print(f"💾 Saved diagnostic {split} demos to {np_path}")
        return diag_demos

    def maybe_apply_diagnostics(self):
        dcfg = self.cfg.demonstrator

        train_mode = getattr(dcfg, "use_diagnostic_train", False)
        eval_mode = getattr(dcfg, "use_diagnostic_eval", False)

        if train_mode is True:
            train_mode = "semi_random"
        if eval_mode is True:
            eval_mode = "semi_random"

        if not train_mode and not eval_mode:
            return {
                "train_demos": self.train_demos,
                "eval_demos": self.eval_demos,
                "metadata": self.meta,
            }

        base_train = self.train_demos
        base_eval = self.eval_demos

        if train_mode:
            self.train_demos = self._load_or_create_diagnostic_split(
                base_train,
                split="train",
                mode=train_mode,
            )

        if eval_mode:
            self.eval_demos = self._load_or_create_diagnostic_split(
                base_eval,
                split="eval",
                mode=eval_mode,
            )

        if hasattr(self, "meta") and self.meta is not None:
            self.meta = {
                **self.meta,
                "use_diagnostic_train": train_mode,
                "use_diagnostic_eval": eval_mode,
                "n_demos_train": len(self.train_demos),
                "n_demos_eval": len(self.eval_demos),
            }

        return {
            "train_demos": self.train_demos,
            "eval_demos": self.eval_demos,
            "metadata": self.meta,
        }
