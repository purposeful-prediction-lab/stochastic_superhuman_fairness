import numpy as np
import json
import random
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from omegaconf import OmegaConf
from stochastic_superhuman_fairness.core.dataset_utils import load_adult, load_compas
from stochastic_superhuman_fairness.core.data_defaults import DEFAULT_DATA_CONFIGS
from stochastic_superhuman_fairness.core.fairness.fairness_metrics import METRIC_REGISTRY
from stochastic_superhuman_fairness.core.utils_io import safe_json_dump, safe_json_load, to_pure
from stochastic_superhuman_fairness.core.utils import normalize_cfg, NamespaceDict

class Demonstrator:
    """Main orchestrator for dataset loading, demo partitioning, and fairness computation."""

    def __init__(self, cfg, auto_create: bool = True):
        self.cfg = cfg
        self.cfg = normalize_cfg(cfg)
        self._resolve_defaults()
        self.cache_dir = Path(self.cfg.demonstrator.cache_dir) / self.cfg.demonstrator.dataset
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = None
        self.train_demos = None
        self.eval_demos = None
        self.sample_id = 1
        if auto_create:
            self.create_demos()
    # ------------------------------------------------------------
    # Config resolution
    # ------------------------------------------------------------
    def _resolve_defaults(self):
        dataset_name = self.cfg.demonstrator.dataset.lower()
        defaults = DEFAULT_DATA_CONFIGS.get(dataset_name, {})

        for key, value in defaults.items():
            if not hasattr(self.cfg.demonstrator, key):
                setattr(self.cfg.demonstrator, key, value)

        # --- ensure data is a NamespaceDict, not a dict ---
        if not hasattr(self.cfg, "data") or not isinstance(self.cfg.data, NamespaceDict):
            self.cfg.data = NamespaceDict()

        for k in ["label_col", "protected_attrs", "normalize", "one_hot", "train_ratio"]:
            setattr(self.cfg.data, k, getattr(self.cfg.demonstrator, k))
    # ------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------
    def create_demos(self, resample=False):
        """Create or load demos according to config."""
        # handle sample selection override
        sample_override = getattr(self.cfg.demonstrator, "demo_sample", None)
        if sample_override:
            print(f"⚡ Loading requested demo sample: {sample_override}")
            return self._load_sample(sample_override)

        if resample:
            self.sample_id += 1

        demo_file = self._build_demo_name()
        npy_path = self.cache_dir / demo_file
        meta_path = npy_path.with_name(npy_path.stem + "_meta.json")

        # Load dataset in memory
        if self.dataset is None:
            self.dataset = self._load_dataset()
        ds = self.dataset

        Xtr, Xte = ds["X_train"], ds["X_test"]
        ytr, yte = ds["y_train"], ds["y_test"]
        Atr, Ate = ds["protected_train"], ds["protected_test"]

        if self.cfg.demonstrator.normalize:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xte = scaler.transform(Xte)

        # Partition into views
        self.train_demos = self._partition(Xtr, ytr, Atr, resample=resample)
        self.eval_demos = self._partition(Xte, yte, Ate, resample=resample)

        fairness = self._compute_fairness(self.train_demos)

        meta = {
            "dataset": self.cfg.demonstrator.dataset,
            "demo_size": self.cfg.demonstrator.demo_size,
            "compute_global": self.cfg.demonstrator.compute_global,
            "normalize": self.cfg.demonstrator.normalize,
            "sample_id": self.sample_id,
            "n_demos_train": len(self.train_demos),
            "n_demos_eval": len(self.eval_demos),
            "train_ratio": self.cfg.demonstrator.train_ratio,
            "n_samples_total": len(Xtr) + len(Xte),
            "n_features": Xtr.shape[1],
            "feature_names": ds["feature_names"],
            "protected_attrs": ds["protected_attrs"],
        }

        out = {
            "train_demos": self.train_demos,
            "eval_demos": self.eval_demos,
            "fairness": fairness,
            "metadata": meta,
        }

        save_format = getattr(self.cfg.demonstrator, "save_format", "separate")
        meta = to_pure(meta)
        if save_format == "zip":
            np.savez_compressed(npy_path.with_suffix(".npz"), **out)
        else:
            np.save(npy_path, out)
            safe_json_dump(meta, meta_path)

        print(f"💾 Saved sample {self.sample_id} demos to {npy_path}")
        return out

    # ------------------------------------------------------------
    # RL Interface
    # ------------------------------------------------------------
    def reset(self, mode="train"):
        """Reset internal sampling index for train or eval demos."""
        if not hasattr(self, "_seen"):
            self._seen = {"train": set(), "eval": set()}
        self._seen[mode] = set()
        print(f"🔄 Reset demo sampler for {mode} demos.")

    def remaining(self, mode="train"):
        """Return number of remaining unseen demos."""
        demos = self.train_demos if mode == "train" else self.eval_demos
        seen = self._seen.get(mode, set())
        return len(demos) - len(seen)

    def step(self, n_samples=1, mode="train", as_torch=False, device="cpu", flatten=False):
        """
        Return a random batch of unseen demos for RL-style training.

        Args:
            n_samples (int): Number of unseen demos to sample.
            mode (str): 'train' or 'eval'.
            as_torch (bool): If True, return PyTorch tensors.
            device (str or torch.device): Device for tensors.
            flatten (bool): If True, concatenate demos into single batch.

        Returns:
            list[dict] or tuple: 
                - List of demo dicts if flatten=False.
                - Tuple (X, y, A) if flatten=True.
        """
        # Lazy init
        if not hasattr(self, "_seen"):
            self._seen = {"train": set(), "eval": set()}

        demos = self.train_demos if mode == "train" else self.eval_demos
        seen = self._seen[mode]
        n_total = len(demos)
        unseen = list(set(range(n_total)) - seen)

        if len(unseen) == 0:
            raise StopIteration(f"All {mode} demos have been sampled. Call reset().")

        n_samples = min(n_samples, len(unseen))
        selected = random.sample(unseen, n_samples)
        seen.update(selected)
        self._seen[mode] = seen

        batch_demos = [demos[i] for i in selected]

        # ---- Flatten mode (for RL learners) ----
        if flatten:
            X = np.concatenate([d["X"] for d in batch_demos], axis=0)
            y = np.concatenate([d["y"] for d in batch_demos], axis=0)
            A = np.concatenate([d["A"] for d in batch_demos], axis=0)
            if as_torch:
                if torch is None:
                    raise ImportError("torch is required when as_torch=True")
                X = torch.tensor(X, dtype=torch.float32, device=device)
                y = torch.tensor(y, dtype=torch.float32, device=device)
                A = torch.tensor(A, dtype=torch.float32, device=device)
            return X, y, A

        # ---- List-of-demos mode ----
        if as_torch:
            if torch is None:
                raise ImportError("torch is required when as_torch=True")
            for d in batch_demos:
                d["X"] = torch.tensor(d["X"], dtype=torch.float32, device=device)
                d["y"] = torch.tensor(d["y"], dtype=torch.float32, device=device)
                d["A"] = torch.tensor(d["A"], dtype=torch.float32, device=device)

        return batch_demos
    # -----------------------------------------------------------------
    # Internal Target Sample Loading Helpers
    # -----------------------------------------------------------------
    def _load_sample(self, sample_id):
        """Load a specific demo sample, automatically handling .npy+json or .npz."""
        base = (
            f"{self.cfg.demonstrator.dataset}_demo"
            f"_size{self.cfg.demonstrator.demo_size}"
            f"_glob{self.cfg.demonstrator.compute_global}"
            f"_norm{self.cfg.demonstrator.normalize}"
            f"_sample{sample_id}"
        )

        npy_path = self.cache_dir / f"{base}.npy"
        npz_path = self.cache_dir / f"{base}.npz"
        meta_path = self.cache_dir / f"{base}_meta.json"

        # --- Case 1: single zipped file (.npz) ---
        if npz_path.exists():
            print(f"📂 Loading compressed demo archive: {npz_path}")
            with np.load(npz_path, allow_pickle=True) as data:
                out = {k: data[k].item() if data[k].shape == () else data[k] for k in data.files}
            # ensure 'metadata' exists as dict
            if "metadata" not in out:
                print("⚠️  Metadata not found inside archive; continuing with minimal meta info.")
                out["metadata"] = {"dataset": self.cfg.demonstrator.dataset, "sample_id": sample_id}
            return out

        # --- Case 2: separate files (.npy + _meta.json) ---
        if npy_path.exists():
            print(f"📂 Loading legacy demo files: {npy_path}")
            demos = np.load(npy_path, allow_pickle=True).item()
            if meta_path.exists():
                demos["metadata"] = safe_json_load(meta_path)
            else:
                print("⚠️  Metadata JSON missing; using embedded metadata if present.")
            return demos

        raise FileNotFoundError(f"No demo sample found for ID {sample_id}")
    # -----------------------------------------------------------------
    def iter_batches(self, batch_size: int = 64, as_torch: bool = False, device: str = "cpu"):
        """
        Yield (X, y, A) batches from all training demos concatenated.

        Parameters
        ----------
        batch_size : int
            Number of samples per batch.
        as_torch : bool
            If True, returns torch tensors instead of numpy arrays.
        device : str
            Device for torch tensors ("cpu" or "cuda").
        """
        if not hasattr(self, "train_demos") or len(self.train_demos) == 0:
            raise ValueError("No training demos found. Did you call create_demos()?")

        # --- Concatenate all training demos ---
        X_all = np.concatenate([d["X"] for d in self.train_demos], axis=0)
        y_all = np.concatenate([d["y"] for d in self.train_demos], axis=0)
        A_all = np.concatenate([d["A"] for d in self.train_demos], axis=0)

        n = len(X_all)
        indices = np.arange(n)

        # --- Deterministic shuffle per epoch ---
        rng = np.random.default_rng(getattr(self, "seed", None))
        rng.shuffle(indices)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            Xb, yb, Ab = X_all[idx], y_all[idx], A_all[idx]

            if as_torch:
                yield (
                    torch.as_tensor(Xb, dtype=torch.float32, device=device),
                    torch.as_tensor(yb, dtype=torch.float32, device=device),
                    torch.as_tensor(Ab, dtype=torch.float32, device=device),
                )
            else:
                yield Xb, yb, Ab

    def _load_latest_sample(self):
        """Convenience: automatically load the latest sample by max(sample_id)."""
        pattern = f"{self.cfg.demonstrator.dataset}_demo_*_sample*.npy"
        files = list(self.cache_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No demo samples found in {self.cache_dir}")
        latest = max(files, key=lambda p: int(p.stem.split("_sample")[-1]))
        sample_id = int(latest.stem.split("_sample")[-1])
        return self._load_sample(sample_id)
    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _build_demo_name(self):
        cfg = self.cfg.demonstrator
        return (
            f"{cfg.dataset}_demo"
            f"_size{cfg.demo_size}"
            f"_glob{cfg.compute_global}"
            f"_norm{cfg.normalize}"
            f"_sample{self.sample_id}.npy"
        )

    def _load_dataset(self):
        dcfg = self.cfg.demonstrator
        data_dir = Path(dcfg.cache_dir) / dcfg.dataset
        if dcfg.dataset == "adult":
            return load_adult(
                data_dir,
                dcfg.label_col,
                dcfg.protected_attrs,
                dcfg.normalize,
                dcfg.one_hot,
                dcfg.train_ratio,
                self.cfg.seed,
            )
        elif dcfg.dataset == "compas":
            return load_compas(
                data_dir,
                dcfg.label_col,
                dcfg.protected_attrs,
                dcfg.normalize,
                dcfg.one_hot,
                dcfg.train_ratio,
                self.cfg.seed,
            )
        else:
            raise ValueError(f"Unsupported dataset {dcfg.dataset}")

    def _partition(self, X, y, prot, resample=False):
        """Partition into demos using index views (no copying)."""
        n = len(X)
        demo_size = self.cfg.demonstrator.demo_size
        n_demos = n // demo_size
        indices = np.arange(n)
        if resample:
            np.random.shuffle(indices)
        demos = []
        for i in range(n_demos):
            start, end = i * demo_size, (i + 1) * demo_size
            idx = indices[start:end]
            demos.append({
                "indices": idx,
                # Create views into the shared arrays
                "X": X[idx],
                "y": y[idx],
                "A": prot[idx],
            })
        return demos

    def resample(self):
        """Explicit resampling call (increments sample ID)."""
        return self.create_demos(resample=True)

    def _compute_fairness(self, demos):
        metrics = self.cfg.demonstrator.metrics
        results = []
        for d in demos:
            y_true = d["y"]
            a = d["A"][:, 0]
            y_pred = y_true
            demo_metrics = {}
            for m in metrics:
                fn = METRIC_REGISTRY[m]
                demo_metrics[m] = float(fn(y_true, y_pred, a))
            results.append(demo_metrics)
        global_metrics = {m: float(np.mean([dm[m] for dm in results])) for m in metrics}
        return {"local": results, "global": global_metrics}
