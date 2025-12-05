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
from stochastic_superhuman_fairness.core.fairness.compute_fairness_utils import compute_fairness_features
from stochastic_superhuman_fairness.core.utils_io import safe_json_dump, safe_json_load, to_pure
from stochastic_superhuman_fairness.core.utils import normalize_cfg, NamespaceDict


class Demonstrator:
    def __init__(self, cfg, auto_create: bool = True, to_torch: bool = True):
        self.cfg = normalize_cfg(cfg)
        self._resolve_defaults()
        self.protected_attrs = getattr(self.cfg.demonstrator, "protected_attrs", [])
        self.sensitive_attrs = getattr(self.cfg.demonstrator, "sensitive_attrs", [])
        self.cache_dir = Path(self.cfg.demonstrator.cache_dir) / self.cfg.demonstrator.dataset
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = None
        self.train_demos = None
        self.eval_demos = None
        self.sample_id = 1
        self.device = getattr(cfg, "device", "cpu")
        self.to_torch_flag = to_torch
        self.metrics = self._resolve_metrics(cfg)
        self.cfg.demonstrator.metrics = self.metrics  # ensure downstream consistency
        self.cfg.demonstrator.device = self.device

        if auto_create:
            self.create_demos(to_torch = self.to_torch_flag)
    #----------------------------------------------------------------------
    def _resolve_defaults(self):
        dataset_name = self.cfg.demonstrator.dataset.lower()
        defaults = DEFAULT_DATA_CONFIGS.get(dataset_name, {})
        for key, value in defaults.items():
            if not hasattr(self.cfg.demonstrator, key):
                setattr(self.cfg.demonstrator, key, value)
        if not hasattr(self.cfg, "data") or not isinstance(self.cfg.data, NamespaceDict):
            self.cfg.data = NamespaceDict()
        for k in ["label_col", "protected_attrs", "sensitive_attrs", "normalize", "one_hot", "train_ratio"]:
            setattr(self.cfg.data, k, getattr(self.cfg.demonstrator, k, None))

    def _resolve_metrics(self, cfg):
        """
        Resolve fairness metrics for the Demonstrator.
        Priority:
          1. cfg.metrics.use
          2. cfg.demonstrator.metrics
        Raises an error if neither is provided.
        """
        metrics = None

        # --- Try cfg.metrics.use ---
        if hasattr(cfg, "metrics"):
            m = getattr(cfg.metrics, "use", None)
            if m:
                metrics = list(m)

        # --- Fallback to demonstrator tab ---
        if not metrics and hasattr(cfg, "demonstrator"):
            m = getattr(cfg.demonstrator, "metrics", None)
            if m:
                metrics = list(m)

        # --- Error if none found ---
        if not metrics:
            raise ValueError(
                "❌ No fairness metrics specified in cfg.metrics.use or cfg.demonstrator.metrics."
            )

        # --- Optional consistency notice ---
        if hasattr(cfg, "metrics") and hasattr(cfg, "demonstrator"):
            d_m = getattr(cfg.demonstrator, "metrics", None)
            if d_m and set(metrics) != set(d_m):
                print(
                    f"⚠️ Metric mismatch: metrics.use={metrics} vs demonstrator.metrics={d_m}"
                )

        return metrics
    def _build_demo_name(self):
            cfg = self.cfg.demonstrator
            return (
                f"{cfg.dataset}_demo"
                f"_size{cfg.demo_size}"
                f"_glob{cfg.compute_global}"
                f"_norm{cfg.normalize}"
                f"_sample{self.sample_id}.npy"
            )


    def to_torch(self, device="cpu"):
        """One-time in-place tensor materialization for train/eval demos."""
        if getattr(self, "_torch_device", None) == device:
            return
        for bucket in (self.train_demos or [], self.eval_demos or []):
            for d in bucket:
                for k in ("X", "y", "A"):
                    v = d.get(k, None)
                    if v is not None and not isinstance(v, torch.Tensor):
                        d[k] = torch.as_tensor(v, dtype=torch.float32, device=device)
        self._torch_device = device

    def create_demos(self, resample=False, to_torch: bool = True):
        sample_override = getattr(self.cfg.demonstrator, "demo_sample", None)
        if sample_override:
            print(f"⚡ Loading requested demo sample: {sample_override}")
            return self._load_sample(sample_override)
        if resample:
            self.sample_id += 1

        demo_file = self._build_demo_name()
        npy_path = self.cache_dir / demo_file
        meta_path = npy_path.with_name(npy_path.stem + "_meta.json")

        if self.dataset is None:
            self.dataset = self._load_dataset()
        ds = self.dataset

        Xtr, Xte = ds["X_train"], ds["X_test"]
        ytr, yte = ds["y_train"], ds["y_test"]
        Atr, Ate = ds["sensitive_train"], ds["sensitive_test"]

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
            "n_features": Xtr.shape[1],
            "protected_attrs": self.protected_attrs,
            "sensitive_attrs": self.sensitive_attrs,
            "metrics": self.cfg.demonstrator.metrics,
            "normalize_mode" : getattr(self.cfg.demonstrator, "normalize_mode", "continuous"),
            'device': self.device,
        }
        self.meta = meta

        out = {"train_demos": self.train_demos, "eval_demos": self.eval_demos, "fairness": fairness, "metadata": meta}
        save_format = getattr(self.cfg.demonstrator, "save_format", "separate")
        meta = to_pure(meta)
        if save_format == "zip":
            np.savez_compressed(npy_path.with_suffix(".npz"), **out)
        else:
            np.save(npy_path, out)
            safe_json_dump(meta, meta_path)
        print(f"💾 Saved sample {self.sample_id} demos to {npy_path}")

        if to_torch:
            self.to_torch(device=self.device)
        return out

    def _load_dataset(self):
        dcfg = self.cfg.demonstrator
        data_dir = Path(dcfg.cache_dir) / dcfg.dataset
        if dcfg.dataset == "adult":
            ds = load_adult(data_dir, dcfg.label_col, dcfg.protected_attrs, dcfg.sensitive_attrs,  dcfg.normalize, dcfg.one_hot, dcfg.train_ratio, self.cfg.seed)
        elif dcfg.dataset == "compas":
            ds = load_compas(data_dir, dcfg.label_col, dcfg.protected_attrs, dcfg.sensitive_attrs, dcfg.normalize, dcfg.one_hot, dcfg.train_ratio, self.cfg.seed)
        else:
            raise ValueError(f"Unsupported dataset {dcfg.dataset}")

        df_tr, df_te = ds.get("df_train"), ds.get("df_test")
        if df_tr is not None and df_te is not None:
            A_train = df_tr[self.sensitive_attrs].to_numpy() if self.sensitive_attrs else None
            A_test = df_te[self.sensitive_attrs].to_numpy() if self.sensitive_attrs else None
            if self.protected_attrs:
                Xtr, Xte = ds["X_train"], ds["X_test"]
                drop_tr = [i for i, c in enumerate(df_tr.columns) if c in self.protected_attrs]
                drop_te = [i for i, c in enumerate(df_te.columns) if c in self.protected_attrs]
                if drop_tr:
                    Xtr = np.delete(Xtr, drop_tr, axis=1)
                    Xte = np.delete(Xte, drop_te, axis=1)
                ds.update({"X_train": Xtr, "X_test": Xte})
        else:
            A_train = ds.get("sensitive_train")
            A_test = ds.get("sensitive_test")

        ds.update({"sensitive_train": A_train, "sensitive_test": A_test})
        return ds

    def _partition(self, X, y, prot, resample=False):
        n = len(X)
        if n == 0:
            return []
        demo_size = self.cfg.demonstrator.demo_size
        n_demos = max(1, n // demo_size)
        indices = np.arange(n)
        if resample:
            np.random.shuffle(indices)
        demos = []
        for i in range(n_demos):
            start, end = i * demo_size, (i + 1) * demo_size
            idx = indices[start:end]
            X_i, y_i, A_i = X[idx], y[idx], prot[idx] if prot is not None else None
            fairness_feats = compute_fairness_features(
                torch.as_tensor(X_i, dtype=torch.float32),
                torch.as_tensor(y_i, dtype=torch.float32),
                torch.as_tensor(y_i, dtype=torch.float32),
                torch.as_tensor(A_i, dtype=torch.float32),
                metrics=self.cfg.demonstrator.metrics,
            )
            demos.append({
                "indices": idx,
                "X": X_i,
                "y": y_i,
                "A": A_i,
                "fairness_feats": fairness_feats.detach().cpu().numpy(),
            })
        return demos

    def _should_regenerate_cache(self, meta_path: Path) -> bool:
        """
        Check whether cached demos should be regenerated based on metadata.
        Returns True if regeneration is needed.
        """
        if not meta_path.exists():
            print("⚠️ No existing metadata found — creating demos fresh.")
            return True

        try:
            old_meta = safe_json_load(meta_path)
        except Exception as e:
            print(f"⚠️ Could not read old metadata ({e}), regenerating demos.")
            return True

        keys_to_check = [
            "dataset",
            "train_ratio",
            "normalize",
            "normalize_mode",
            "demo_size",
            "compute_global",
            "protected_attrs",
            "sensitive_attrs",
            "metrics",
            'device',
        ]

        for k in keys_to_check:
            old_val = old_meta.get(k, None)
            new_val = getattr(self.cfg.demonstrator, k, None)
            if old_val != new_val:
                print(f"⚠️ Metadata mismatch on '{k}': {old_val} → {new_val}")
                return True

        print(f"🟢 Using cached demos for {self.cfg.demonstrator.dataset}")
        return False
    def resample(self):
        """Explicit resampling call (increments sample ID and recreates demos)."""
        return self.create_demos(resample=True)

    def iter_batches(self, batch_size: int = 1, source: str = 'train', as_torch: bool = False, device: str = "cpu", shuffle: bool = True):
        """Yield lists of demo samples (populations), one list per batch."""
        if not hasattr(self, "train_demos") or len(self.train_demos) == 0:
            raise ValueError("No training demos found. Did you call create_demos()?")

        demos = self.train_demos if source == 'train' else self.eval_demos

        n = len(demos)
        indices = np.arange(n)
        rng = np.random.default_rng(getattr(self, "seed", None))
        if shuffle:
            rng.shuffle(indices)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = [demos[i] for i in indices[start:end]]
            if as_torch:
                for d in batch:
                    for k in ["X", "y", "A"]:
                        d[k] = torch.as_tensor(d[k], dtype=torch.float32, device=device)
            yield batch

    def _compute_fairness(self, demos):
        metrics = self.cfg.demonstrator.metrics
        results = []
        for d in demos:
            y_true = d["y"]
            a = d["A"][:, 0] if d["A"] is not None else np.zeros_like(y_true)
            y_pred = y_true
            demo_metrics = {m: float(METRIC_REGISTRY[m](y_true, y_pred, a)) for m in metrics}
            results.append(demo_metrics)
        global_metrics = {m: float(np.mean([dm[m] for dm in results])) for m in metrics}
        return {"local": results, "global": global_metrics}

    # -----------------------------------------------------------------
    # RL Interface
    # -----------------------------------------------------------------
    def reset(self, mode="train"):
        """Reset sampler state."""
        if not hasattr(self, "_seen"):
            self._seen = {"train": set(), "eval": set()}
        self._seen[mode] = set()
        print(f"🔄 Reset demo sampler for {mode} demos.")

    def remaining(self, mode="train"):
        """Number of unseen demo samples."""
        demos = self.train_demos if mode == "train" else self.eval_demos
        seen = self._seen.get(mode, set())
        return len(demos) - len(seen)

    def step(self, n_samples=1, mode="train", as_torch=False, device="cpu"):
        """Return random unseen demos for RL training."""
        if not hasattr(self, "_seen"):
            self._seen = {"train": set(), "eval": set()}

        demos = self.train_demos if mode == "train" else self.eval_demos
        seen = self._seen[mode]
        unseen = list(set(range(len(demos))) - seen)
        if len(unseen) == 0:
            raise StopIteration(f"All {mode} demos sampled. Call reset().")

        n_samples = min(n_samples, len(unseen))
        selected = random.sample(unseen, n_samples)
        seen.update(selected)
        self._seen[mode] = seen
        batch_demos = [demos[i] for i in selected]

        if as_torch:
            for d in batch_demos:
                for k in ["X", "y", "A"]:
                    d[k] = torch.as_tensor(d[k], dtype=torch.float32, device=device)
        return batch_demos

    # -----------------------------------------------------------------
    # Saved sample loading
    # -----------------------------------------------------------------
    def _load_sample(self, sample_id):
        """Load saved demo sample (npz or npy + json)."""
        base = (
            f"{self.cfg.demonstrator.dataset}_demo"
            f"_size{self.cfg.demonstrator.demo_size}"
            f"_glob{self.cfg.demonstrator.compute_global}"
            f"_norm{self.cfg.demonstrator.normalize}"
            f"_sample{sample_id}"
        )

        npz_path = self.cache_dir / f"{base}.npz"
        npy_path = self.cache_dir / f"{base}.npy"
        meta_path = self.cache_dir / f"{base}_meta.json"

        if npz_path.exists():
            print(f"📂 Loading compressed demo archive: {npz_path}")
            with np.load(npz_path, allow_pickle=True) as data:
                out = {k: data[k].item() if data[k].shape == () else data[k] for k in data.files}
            if "metadata" not in out:
                out["metadata"] = {"dataset": self.cfg.demonstrator.dataset, "sample_id": sample_id}
            return out

        if npy_path.exists():
            print(f"📂 Loading legacy demo files: {npy_path}")
            demos = np.load(npy_path, allow_pickle=True).item()
            if meta_path.exists():
                demos["metadata"] = safe_json_load(meta_path)
            return demos

        raise FileNotFoundError(f"No demo sample found for ID {sample_id}")


    def reload_if_cfg_changed(self):
        """
        Re-check metadata in cache and re-partition demos if config mismatch is detected.
        """
        demo_file = self._build_demo_name()
        npy_path = self.cache_dir / demo_file
        meta_path = npy_path.with_name(npy_path.stem + "_meta.json")

        regenerate = self._should_regenerate_cache(meta_path)
        if regenerate or not npy_path.exists():
            print("♻️ Regenerating demos due to config change...")
            self.create_demos(resample=False, to_torch=self.to_torch_flag)
        else:
            print("✅ Cached demos match current config.")
