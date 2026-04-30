import numpy as np
import json
import random
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Literal, Optional, Tuple, Dict, Union
from omegaconf import OmegaConf
from stochastic_superhuman_fairness.core.dataset_utils import load_adult, load_compas
from stochastic_superhuman_fairness.core.data_defaults import DEFAULT_DATA_CONFIGS
from stochastic_superhuman_fairness.core.fairness.fairness_metrics import METRIC_REGISTRY, zero_one_loss, compute_fairness_features
from stochastic_superhuman_fairness.core.fairness.subdominance import subdominance_loss_from_features, compute_beat_rates
from stochastic_superhuman_fairness.core.demonstrator_utils import AgreementStatsMixin
from stochastic_superhuman_fairness.core.utils_io import safe_json_dump, safe_json_load, to_pure
from stochastic_superhuman_fairness.core.utils import normalize_cfg, NamespaceDict, sample_logistic_model
from stochastic_superhuman_fairness.core.fairness.subdominance import subdominance_loss_from_features
from stochastic_superhuman_fairness.core.qp_solver import solve_stochastic_subdom_coupling, bj_from_beatrates_nocollapse

class Demonstrator(AgreementStatsMixin):
    def __init__(self, cfg, auto_create: bool = True, to_torch: bool = True):
        self.cfg = normalize_cfg(cfg)
        self.dcfg = self.cfg.get('demonstrator')
        self._resolve_defaults()
        self.protected_attrs = getattr(self.cfg.demonstrator, "protected_attrs", [])
        self.sensitive_attrs = getattr(self.cfg.demonstrator, "sensitive_attrs", [])
        self.cache_dir = Path(self.cfg.demonstrator.cache_dir) / self.cfg.demonstrator.dataset
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Diagnostics Dataset dir
        self.diagnostics_dir = self.cache_dir / "diagnostics"
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        self.shared_x = self.dcfg.get('shared_x', False)
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
            self._compute_standalone_demofeats()
            self._compute_standalone_demolabels()
        self.keep_only_requested_fairness_metrics(required_metrics=self.metrics)
        #  import ipdb;ipdb.set_trace()
        self.compute_demo_ranking()
        self.compute_intrademo_subdom()
        self.compute_intrademo_ot()
        self.compute_label_agreement_dict()

    
    # --------------------------------------------------
    def compute_label_agreement_dict(self,  
        return_disagreement_per_sample: bool = False,
        exclude_self: bool = True,
        ):
        kwargs = {'return_disagreement_per_sample': return_disagreement_per_sample,
                  'exclude_self': exclude_self,
                  }
        self.label_agreement_dict = {}
        self.label_agreement_dict['train'] = self.compute_label_agreement(self.train_demo_labels[self.demo_ranking_train], **kwargs)
        self.label_agreement_dict['eval'] = self.compute_label_agreement(self.eval_demo_labels[self.demo_ranking_eval], **kwargs)
        self.label_agreement_dict['train']['pairwise_agreement_sorted'] = self.label_agreement_dict['train'].pop('pairwise_agreement') 
        self.label_agreement_dict['eval']['pairwise_agreement_sorted'] = self.label_agreement_dict['eval'].pop('pairwise_agreement') 
        self.label_agreement_dict['train']['pairwise_agreement'] = self.compute_label_agreement(self.train_demo_labels[self.demo_ranking_train], **kwargs)['pairwise_agreement']
        self.label_agreement_dict['eval']['pairwise_agreement'] = self.compute_label_agreement(self.eval_demo_labels[self.demo_ranking_eval], **kwargs)['pairwise_agreement']

    def compute_intrademo_ot(self, alpha: float = None, beta = None, to_torch = True):
        device = 'cpu' if to_torch is False else 'cuda'
        S_OT = self.__dict__[f'train_intrademo_subdom_dict']['S']
        S_OT += np.eye(*S_OT.shape) * 1e10
        
        out = solve_stochastic_subdom_coupling(
            S_OT,
            solver=self.cfg.learner.default.train.stochastic.solver,
            weight_method="primal",
            normalize_subdom=False,
            demo_marginals=None,
            row_constraints=False,
        )
        self.intrademo_ot_dict = out
        self.intrademo_gamma = out['gamma_np']
        self.intrademo_gamma_torch = torch.tensor(out['gamma_np']).to(device)
        #  import ipdb;ipdb.set_trace()
    # --------------------------------------------------
    def compute_intrademo_subdom(self, alpha: float = None, beta = None, to_torch = True):
        '''Default alpha = all ones and beta 0. scalars are repeated to match feature dim K.
           Per rollout reduction is mean.
        '''
        keys = ['train', 'eval']
        device = 'cpu' if to_torch is False else 'cuda'
        for key in keys:
            feats = self.__dict__[f'{key}_demo_feats'] 
            self.__dict__[f'{key}_intrademo_subdom_dict'] = subdominance_loss_from_features(feats, feats)
            self.__dict__[f'{key}_intrademo_S_torch'] =  torch.tensor(self.__dict__[f'{key}_intrademo_subdom_dict']['S']).to(device)
            self.__dict__[f'{key}_intrademo_subdom_dict']['median'] = np.median(
                    self.__dict__[f'{key}_intrademo_subdom_dict']['per_rollout'])
            self.__dict__[f'{key}_intrademo_mean_subdom'] = self.__dict__[f'{key}_intrademo_subdom_dict']['per_rollout'].mean()

    # --------------------------------------------------

    #  def get_top_k(self, k : float, metric: Literal["beat_rate", "subdom"] = "beat_rate"):
    #      if (k>0) and (k <=1.):
    #          int_k = int(k*len(self.train_demo_feats))
    #      elif k >  self.num_demos_train:
    #          int_k = self.num_demos_train
    def _compute_standalone_demolabels(self):
        self.train_demo_labels = torch.stack([d["y_demo"] for d in self.train_demos])  # [D,K]      
        self.eval_demo_labels = torch.stack([d["y_demo"] for d in self.eval_demos])  # [D,K]      

    # --------------------------------------------------

    def _compute_standalone_demofeats(self):
        try:
            self.train_demo_feats = np.stack([d["fairness_feats"] for d in self.train_demos])  # [D,K]      
        except:
            import ipdb;ipdb.set_trace()
        self.eval_demo_feats = np.stack([d["fairness_feats"] for d in self.eval_demos])  # [D,K]      
        self.train_demo_means_sorted = Demonstrator.compute_sorted_demo_means(self.train_demo_feats)
        self.eval_demo_means_sorted = Demonstrator.compute_sorted_demo_means(self.eval_demo_feats)

    # --------------------------------------------------
    def compute_demo_ranking(self):
        self.num_demos_train = len(self.train_demo_feats)
        self.num_demos_eval = len(self.eval_demo_feats)
        self._compute_intra_demo_subdominance()
        self.beat_rates_train = compute_beat_rates(self.intra_S)
        self.demo_ranking_train = np.argsort(-self.beat_rates_train)  # descending
        self.beat_rates_eval = compute_beat_rates(self.intra_S_eval)
        self.demo_ranking_eval = np.argsort(-self.beat_rates_eval)  # descending

    def sort_demos_by_ranking(self):
        types = ['train', 'eval']
        for t in types:
            get = self__dict__[f'{t}_demos'].__getitem__
            self.__dict__[f'{t}_demos_sorted'] = [get(i) for i in self.__dict__[f'demo_ranking_{t}']]
        
    def get_rank_sorted_demos(self):
        if not hasattr(self, 'train_demos_sorted'):
            self.sort_demos_by_ranking()
        return self.train_demos_sorted, self.eval_demos_sorted

    def sort_demo_feats_by_ranking(self):
        types = ['train', 'eval']
        for t in types:
            get = self.__dict__[f'{t}_demo_feats'].__getitem__
            self.__dict__[f'{t}_demo_feats_sorted'] = [get(i) for i in self.__dict__[f'demo_ranking_{t}']]

    def get_demo_rankings(self):
        if not hasattr(self, 'demo_ranking_train'):
            self.compute_demo_ranking()
        return self.demo_ranking_train, self.demo_ranking_eval

    def get_rank_sorted_demo_feats(self):
        if not hasattr(self, 'train_demo_feats_sorted'):
            self.sort_demo_feats_by_ranking()
        return self.train_demo_feats_sorted, self.eval_demo_feats_sorted

    def _compute_intra_demo_subdominance(self):
        if not hasattr(self, 'train_demo_feats'):
           self._compute_standalone_demofeats()
        self.intra_S = subdominance_loss_from_features(
            self.train_demo_feats,  
            self.train_demo_feats, 
            agg='sum',
            alpha = 1.,
            beta = 0.
            )['S']
        self.intra_S_eval = subdominance_loss_from_features(
            self.eval_demo_feats,  
            self.eval_demo_feats, 
            agg='sum',
            alpha = 1.,
            beta = 0.
            )['S']

    def _resolve_defaults(self):
        dataset_name = self.cfg.demonstrator.dataset.lower()
        defaults = DEFAULT_DATA_CONFIGS.get(dataset_name, {})
        for key, value in defaults.items():
            if not hasattr(self.cfg.demonstrator, key):
                try:
                    setattr(self.cfg.demonstrator, key, value)
                except:
                    import ipdb;ipdb.set_trace()
        if not hasattr(self.cfg, "data") or not isinstance(self.cfg.data, NamespaceDict):
            self.cfg.data = NamespaceDict()
        for k in ["label_col", "protected_attrs", "sensitive_attrs", "normalize", "one_hot", "train_ratio"]:
            setattr(self.cfg.data, k, getattr(self.cfg.demonstrator, k, None))

    # --------------------------------------------------

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

    # --------------------------------------------------

    def _build_demo_name(self):
        cfg = self.cfg.demonstrator
        demotype = getattr(cfg, "demotype", "partition")
        n_models = getattr(cfg, "n_models", None)
        subset_ratio = getattr(cfg, "subset_ratio", None)
        subset_size = getattr(cfg, "subset_size", None)

        extra = f"_type{demotype}"
        if demotype == "lrdecisions":
            if n_models is not None: extra += f"_n{int(n_models)}"
            if subset_ratio is not None: extra += f"_sr{float(subset_ratio):.3g}"
            if subset_size is not None: extra += f"_ss{int(subset_size)}"

        return (
            f"{cfg.dataset}_demo{extra}"
            f"_size{cfg.demo_size}"
            f"_glob{cfg.compute_global}"
            f"_norm{cfg.normalize}"
            f"_sample{self.sample_id}"+ ".npz" if cfg.save_format == 'zip' else '.npy'
        )
    def to_torch(self, device="cpu"):
        """One-time in-place tensor materialization for train/eval demos."""
        if getattr(self, "_torch_device", None) == device:
            return
        for bucket in (self.train_demos or [], self.eval_demos or []):
            for d in bucket:
                for k in ("X", "y", "A", "y_demo"):
                    v = d.get(k, None)
                    if v is not None and not isinstance(v, torch.Tensor):
                        d[k] = torch.as_tensor(v, dtype=torch.float32, device=device)
        self._torch_device = device

    #-----------------------------------------------------------------------------------------------------------

    def get_metadata(self):
        return self.meta
    
    #-----------------------------------------------------------------------------------------------------------
    def get_baseline_fairness_features(self):
        if not hasattr(self, "baseline_fairness_features"):
                       self._compute_baseline_fairness_features()
        return self.baseline_fairness_features

    def _compute_baseline_fairness_features(self):
        self.baseline_fairness_features = {'oracle':self._compute_oracle_fairness(), 'majority': self._compute_majority_fairness(),
                                  'random': self._compute_random_fairness(),
                                  }

    #-----------------------------------------------------------------------------------------------------------

    def create_demos(self, resample=False, to_torch: bool = True):

        #  import ipdb;ipdb.set_trace()
        sample_override = getattr(self.cfg.demonstrator, "demo_sample", None)

        if sample_override:
            print(f" Loading requested demo sample: {sample_override}")
            out = self._load_sample(sample_override)

            if to_torch:
                self.to_torch(device=self.device)
            return out

        if resample:
            self.sample_id += 1

        dcfg = self.cfg.demonstrator
        demotype = getattr(dcfg, "demotype", "partition")

        demo_file = self._build_demo_name()
        np_path = self.cache_dir / demo_file
        meta_path = np_path.with_name(np_path.stem + "_meta.json") if dcfg.save_format != 'zip' else np_path

        overwrite = bool(getattr(dcfg, "overwrite", False))

        if np_path.exists() and not overwrite and not self._should_regenerate_cache(meta_path):

            if 'zip' in dcfg.save_format:
                out = np.load(np_path, allow_pickle=True)
                self.train_demos = out["train_demos"].tolist()
                self.eval_demos = out["eval_demos"].tolist()
                self.meta = out.get("metadata", {}).tolist()
                if self.meta == {}:
                    print("\nNo metadata found in npz!...\n")
            else:
                out = np.load(np_path, allow_pickle=True).item()
                self.meta = out.get("metadata", safe_json_load(meta_path) if meta_path.exists() else {})
                self.train_demos = out["train_demos"]
                self.eval_demos = out["eval_demos"]
            print(f"📂 Loaded cached demos from {np_path}")

            self.maybe_apply_diagnostics()
            if to_torch:
                self.to_torch(device=self.device)

            return out

        if self.dataset is None:
            self.dataset = self._load_dataset()

        ds = self.dataset


        Xtr, Xte = ds["X_train"], ds["X_test"]
        ytr, yte = ds["y_train"], ds["y_test"]
        Atr, Ate = ds.get("sensitive_train").astype(int), ds.get("sensitive_test").astype(int)

        if demotype == "lrdecisions":
            self.train_demos = self._create_demos_lrdecisions(Xtr, ytr, Atr, is_eval=False)
            self.eval_demos  = self._create_demos_lrdecisions(Xte, yte, Ate, is_eval=True)

        else:
            self.train_demos = self._partition(Xtr, ytr, Atr, resample=resample)
            self.eval_demos  = self._partition(Xte, yte, Ate, resample=resample)

        fairness = self._compute_fairness(self.train_demos)

        meta = {
            "dataset": dcfg.dataset,
            "demo_size": dcfg.demo_size,
            "compute_global": dcfg.compute_global,
            "normalize": dcfg.normalize,
            "sample_id": self.sample_id,
            "n_demos_train": len(self.train_demos),
            "n_demos_eval": len(self.eval_demos),
            "train_ratio": dcfg.train_ratio,
            "n_features": Xtr.shape[1],
            "protected_attrs": self.protected_attrs,
            "sensitive_attrs": self.sensitive_attrs,
            "metrics": dcfg.metrics,
            "normalize_mode": getattr(dcfg, "normalize_mode", "continuous"),
            "device": self.device,
            "demotype": demotype,
            "n_models": getattr(dcfg, "n_models", None),
            "subset_ratio": getattr(dcfg, "subset_ratio", None),
            "subset_size": getattr(dcfg, "subset_size", None),
            "baseline_fairness_features": self.get_baseline_fairness_features(),
            #  "lr_max_iter": getattr(dcfg, "lr_max_iter", None),
            #  "lr_C": getattr(dcfg, "lr_C", None),
            #  "lr_solver": getattr(dcfg, "lr_solver", None),
            #  "lr_n_jobs": getattr(dcfg, "lr_n_jobs", None),
        }

        self.meta = meta

        out = {"train_demos": self.train_demos, "eval_demos": self.eval_demos, "fairness": fairness, "metadata": meta}
        save_format = getattr(dcfg, "save_format", "separate")
        meta_pure = to_pure(meta)

        if save_format == "zip":
            np.savez_compressed(np_path.with_suffix(".npz"), **out)
        else:
            np.save(np_path, out)
            safe_json_dump(meta_pure, meta_path)

        print(f"💾 Saved sample {self.sample_id} demos to {np_path}")
        self.maybe_apply_diagnostics()
        if to_torch:
            self.to_torch(device=self.device)

        return out

    # --------------------------------------------------

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

    # --------------------------------------------------

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
                "zero_one_loss": zero_one_loss(y_i, y_i)
            })
        return demos

    def _compute_random_fairness(self):
        y = self.train_demos[0]["y"]
        if isinstance(y, torch.Tensor):
            rand_bin = (torch.rand_like(y, dtype=torch.float32) > 0.5).to(y.dtype)
        else:
            rand_bin = (np.random.random(y.shape) > 0.5).astype(y.dtype)
        #  rng = np.random.default_rng()
        #  rand_bin = (rng.random(self.train_demos[0]['y'].shape) > 0.5).astype(int)
        return compute_fairness_features(
                self.train_demos[0]['y'],
                rand_bin,
                self.train_demos[0]['A'],
                self.cfg.get('demonstrator').get('metrics'))

    def _compute_oracle_fairness(self):
        return compute_fairness_features(
                self.train_demos[0]['y'],
                self.train_demos[0]['y'],
                self.train_demos[0]['A'],
                self.cfg.get('demonstrator').get('metrics'))

    def _compute_majority_fairness(self):
        #  majority_labels = np.ones_like(self.train_demos[0]['y']) if self.train_demos[0]['y'].mean() >= 0.5 else np.zeros_like(self.train_demos[0]['y'])
        y = self.train_demos[0]["y"]
        majority_labels = y * 0 + (1 if (y.float().mean() if hasattr(y, "float") else y.mean()) >= 0.5 else 0)
        return compute_fairness_features(
                self.train_demos[0]['y'],
                majority_labels,
                self.train_demos[0]['A'],
                self.cfg.get('demonstrator').get('metrics'))

    def _compute_fairness(self, demos):
            metrics = self.cfg.demonstrator.metrics
            results = []
            for d in demos:
                y_true = d["y"]
                a = d["A"][:, 0] if d["A"] is not None else np.zeros_like(y_true)
                y_pred = d['y_demo']
                fairness_feats = compute_fairness_features(
                                y_true,
                                y_pred,
                                a,
                                metrics=metrics,
                            )
                if torch.is_tensor(fairness_feats):
                    fairness_feats = fairness_feats.detach().cpu().numpy()

                results.append(fairness_feats)
            global_metrics = {m: float(np.mean([dm[i] for dm in results])) for i,m in enumerate(metrics)}
            #  import ipdb;ipdb.set_trace()
            return {"local": results, "global": global_metrics}

    def _create_demos_lrdecisions(self, X, y, A, is_eval: bool = False):

        """demotype=lrdecisions: train n LR models on subsets, predict on full (X,y)."""


        if A is None:
            raise ValueError("demotype=lrdecisions requires sensitive A (cfg.demonstrator.sensitive_attrs).")

        cfg = self.cfg.demonstrator
        n_models = int(getattr(cfg, "n_models", 100))
        subset_ratio = getattr(cfg, "subset_ratio", None)
        subset_size = getattr(cfg, "subset_size", None)

        n = len(X)

        if subset_size is None:
            subset_ratio = 0.7 if subset_ratio is None else float(subset_ratio)
            subset_size = max(1, int(round(n * subset_ratio)))

        subset_size = min(int(subset_size), n)

        rng = np.random.default_rng(int(getattr(self.cfg, "seed", 0)) + (999 if is_eval else 0))
        demos = []
        model_type = 'evaluation' if is_eval else 'training'
        print(f"Generating {n_models}  Classifiers as {model_type} demonstrators")
        for m in range(n_models):
            print('\n'+'-'*30 + f"Fitting logistic demonstrator {m}/{n_models}.")

            # Sample a distinct logistic cclassifier
            lr = sample_logistic_model(cfg)
            # Train on subset of data
            idx = rng.choice(n, size=subset_size, replace=False)
            lr.fit(X[idx], y[idx])
            # Make decisions  with different thresholds for variablity
            y_demo = lr.predict_proba(X)[:, 1].astype(np.float32)  # decisions on full set, returns probs
            threshold = rng.uniform(0.35, 0.85)
            y_demo_zero_one = (lr.predict_proba(X)[:, 1] >= threshold).astype(np.float32)

            fairness_feats = compute_fairness_features(
                torch.as_tensor(y, dtype=torch.float32), # y_true
                torch.as_tensor(y_demo, dtype=torch.float32),
                torch.as_tensor(A, dtype=torch.float32),
                metrics=cfg.metrics,
                X = torch.as_tensor(X, dtype=torch.float32), # Currently unused
            ).detach().cpu().numpy()

            zero_one = zero_one_loss(y, y_demo_zero_one)
            demos.append({"indices": idx, "X": X, "y": y, "A": A, "y_demo": y_demo_zero_one, "fairness_feats": fairness_feats, "zero_one_loss": zero_one})

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
            if 'npz' in str(meta_path):
                old_meta = np.load(meta_path, allow_pickle = True)['metadata'].tolist() # turn to dict from array
            else:
                old_meta = safe_json_load(meta_path)
        except Exception as e:
            print(f"⚠️ Could not read old metadata ({e}), regenerating demos.")
            return True

        keys_to_check = [
            "dataset","train_ratio","normalize","normalize_mode","demo_size","compute_global",
            "protected_attrs","sensitive_attrs","metrics",
            "demotype","n_models","subset_ratio","subset_size",
            "lr_max_iter","lr_C","lr_solver","lr_n_jobs",
            "device",
        ]

        for k in keys_to_check:
            old_val = old_meta.get(k, None)
            new_val = (self.device if k == 'device' else getattr(self.cfg.demonstrator, k, None))
            if k == "metrics":

                old_set = set(old_val)
                new_set = set(new_val) if new_val is not None else set()
                # allow if new ⊆ old
                if not new_set.issubset(old_set):
                    print(f"⚠️ Metadata mismatch on '{k}': {old_val} → {new_val}")
                    return True
                continue  # valid subset → no mismatch
            if old_val != new_val:
                print(f"⚠️ Metadata mismatch on '{k}': {old_val} → {new_val}")
                return True

        print(f"🟢 Using cached demos for {self.cfg.demonstrator.dataset}")
        return False

    # --------------------------------------------------

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

        # -----------------------------------------------------------------
    # Static Methods
    # -----------------------------------------------------------------
    @staticmethod
    def compute_sorted_demo_means(demos: np.ndarray) -> np.ndarray:
        """
        Given unsorted demos (D,K), compute per-feature sorted cumulative means.

        For each feature k:
          - sort demos[:,k] ascending -> v
          - means[:,k] = cumsum(v) / (1..D)

        Returns
        -------
        means : (D,K) ndarray
            Column k is the cumulative mean of the sorted demo values for feature k.
            Each column is nondecreasing.
        """
        x = np.asarray(demos, dtype=float)
        if x.ndim != 2:
            raise ValueError("demos must be a 2D array of shape (D, K).")

        # sort each column independently
        xs = np.sort(x, axis=0)                         # (D,K)
        cumsum = np.cumsum(xs, axis=0)                  # (D,K)
        denom = np.arange(1, xs.shape[0] + 1)[:, None]  # (D,1)
        return cumsum / denom
    # -----------------------------------------------------------------
    # Learning Inteface
    # -----------------------------------------------------------------
    def target_key(self) -> str:
        # lrdecisions demos store decisions in y_demo; partition uses y
        return "y_demo" if getattr(self.cfg.demonstrator, "demotype", "partition") == "lrdecisions" else "y"

    def get_targets(self, d):
        # always returns the training target tensor/array for this demo
        k = self.target_key()
        return d[k] if k in d and d[k] is not None else d["y"]
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

    # --------------------------------------------------------------------------------

    def keep_only_requested_fairness_metrics(
        self,
        required_metrics=None,
        update_cfg: bool = True,
        recompute_baselines: bool = True,
    ):
        """
        Keep only the requested fairness metrics in each demo's fairness_feats.

        If the currently loaded demos already contain all requested metrics,
        slice fairness_feats using the metric order stored in self.meta["metrics"].

        If any requested metric is missing, recompute fairness_feats from scratch
        for both train and eval demos using the requested metrics.

        Assumptions
        -----------
        - self.meta["metrics"] order matches the order in d["fairness_feats"].
        - If a demo has y_demo, that should be used as prediction target.
          Otherwise use y.

        Parameters
        ----------
        required_metrics : list[str] | None
            Metrics to keep. If None, uses self.cfg.demonstrator.metrics.
        update_cfg : bool
            If True, also update self.cfg.demonstrator.metrics.
        recompute_baselines : bool
            If True, recompute baseline fairness features after metric change.
        """
        if self.train_demos is None or self.eval_demos is None:
            raise ValueError("No demos loaded. Call create_demos() or _load_sample() first.")

        if required_metrics is None:
            required_metrics = list(self.cfg.demonstrator.metrics)
        else:
            required_metrics = list(required_metrics)

        if len(required_metrics) == 0:
            raise ValueError("required_metrics cannot be empty.")

        loaded_metrics = []
        if hasattr(self, "meta") and self.meta is not None:
            loaded_metrics = list(self.meta.get("metrics", []) or [])

        # ------------------------------------------------------------------
        # Helper: recompute fairness feats for one demo
        # ------------------------------------------------------------------
        def _recompute_demo_fairness_feats(d, metrics):
            y_true = d["y"]
            y_pred = self.get_targets(d)
            A = d["A"]
            X = d.get("X", None)

            ff = compute_fairness_features(
                torch.as_tensor(y_true, dtype=torch.float32),
                torch.as_tensor(y_pred, dtype=torch.float32),
                torch.as_tensor(A, dtype=torch.float32),
                metrics=metrics,
                X=None if X is None else torch.as_tensor(X, dtype=torch.float32),
            )
            d["fairness_feats"] = ff.detach().cpu().numpy()
            return d

        # ------------------------------------------------------------------
        # Case 1: all requested metrics already exist -> slice only
        # ------------------------------------------------------------------
        can_slice = (
            len(loaded_metrics) > 0
            and all(m in loaded_metrics for m in required_metrics)
        )

        if can_slice:
            metric_to_idx = {m: i for i, m in enumerate(loaded_metrics)}
            keep_idx = [metric_to_idx[m] for m in required_metrics]

            def _slice_bucket(demos):
                for d in demos:
                    ff = d.get("fairness_feats", None)
                    if ff is None:
                        # fallback to recompute if a demo is malformed
                        _recompute_demo_fairness_feats(d, required_metrics)
                    else:
                        ff = np.asarray(ff)
                        if ff.ndim != 1 or ff.shape[0] < max(keep_idx) + 1:
                            _recompute_demo_fairness_feats(d, required_metrics)
                        else:
                            d["fairness_feats"] = ff[keep_idx]
                return demos

            self.train_demos = _slice_bucket(self.train_demos)
            self.eval_demos = _slice_bucket(self.eval_demos)

        # ------------------------------------------------------------------
        # Case 2: some requested metric missing -> recompute all requested
        # ------------------------------------------------------------------
        else:
            print(
                "⚠️ Requested metrics are not fully present in loaded demos. "
                "Recomputing fairness_feats from scratch."
            )

            self.train_demos = [
                _recompute_demo_fairness_feats(d, required_metrics)
                for d in self.train_demos
            ]
            self.eval_demos = [
                _recompute_demo_fairness_feats(d, required_metrics)
                for d in self.eval_demos
            ]

        # ------------------------------------------------------------------
        # Update metadata / cfg / cached demo feature arrays
        # ------------------------------------------------------------------
        if not hasattr(self, "meta") or self.meta is None:
            self.meta = {}

        self.meta["metrics"] = list(required_metrics)

        if update_cfg:
            self.cfg.demonstrator.metrics = list(required_metrics)
            self.metrics = list(required_metrics)

        # Recompute summary fairness dicts with the new metric set
        self.fairness = {
            "train": self._compute_fairness(self.train_demos),
            "eval": self._compute_fairness(self.eval_demos),
        }

        if recompute_baselines:
            self._compute_baseline_fairness_features()
            self.meta["baseline_fairness_features"] = self.baseline_fairness_features

        self._compute_standalone_demofeats()

        return {
            "train_demos": self.train_demos,
            "eval_demos": self.eval_demos,
            "metrics": required_metrics,
            "metadata": self.meta,
        }

    # -----------------------------------------------------------------
    # Saved sample loading
    # -----------------------------------------------------------------
    def _load_sample(self, sample_id):
        """Load saved demo sample (npz or npy + json)."""
        cfg = self.cfg.demonstrator
        demotype = getattr(cfg, "demotype", "partition")
        if demotype == "fullset":
            demotype = "lrdecisions"
        n_models = getattr(cfg, "n_models", None)
        subset_ratio = getattr(cfg, "subset_ratio", None)
        subset_size = getattr(cfg, "subset_size", None)

        extra = f"_type{demotype}"
        if demotype == "lrdecisions":
            if n_models is not None: extra += f"_n{int(n_models)}"
            if subset_ratio is not None: extra += f"_sr{float(subset_ratio):.3g}"
            if subset_size is not None: extra += f"_ss{int(subset_size)}"

        base = (
            f"{cfg.dataset}_demo{extra}"
            f"_size{cfg.demo_size}"
            f"_glob{cfg.compute_global}"
            f"_norm{cfg.normalize}"
            f"_sample{sample_id}"
        )

        npz_path = self.cache_dir / f"{base}.npz"
        npy_path = self.cache_dir / f"{base}.npy"
        meta_path = self.cache_dir / f"{base}_meta.json"

        if npz_path.exists():
            print(f"📂 Loading compressed demos from {npz_path}")
            data = dict(np.load(npz_path, allow_pickle=True))
            out = {k: data[k].item() if hasattr(data[k], "item") else data[k] for k in data.keys()}
        elif npy_path.exists():
            print(f"📂 Loading demos from {npy_path}")
            out = np.load(npy_path, allow_pickle=True).item()
        else:
            raise FileNotFoundError(f"No demo sample found at {npz_path} or {npy_path}")

        self.train_demos = out.get("train_demos", None)
        self.eval_demos = out.get("eval_demos", None)
        self.meta = out.get("metadata", safe_json_load(meta_path) if meta_path.exists() else {})
        return out
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


    # DIAGNOSTIC SET BUILDING
    #===========================================================================================================

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
        train_frac: float,
        random_mult: float,
        replace_train: bool,
        p_one: float,
    ):
        base = self._build_demo_base_name()
        return (
            f"{base}"
            f"_diag{split}"
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


    def _make_diagnostic_split(
        self,
        demos,
        train_frac: float = 0.5,
        random_mult: float = 0.5,
        replace_train: bool = False,
        p_one: float = 0.5,
        seed: int | None = None,
    ):
        if not (0.0 <= train_frac <= 1.0):
            raise ValueError("train_frac must be in [0,1]")
        if not (0.0 <= random_mult <= 10.0):
            raise ValueError("random_mult must be in [0,10]")
        if demos is None or len(demos) == 0:
            return []

        rng = np.random.default_rng(seed)
        n_total = len(demos)

        n_keep = int(round(train_frac * n_total))
        n_rand = int(round(random_mult * n_total))

        if n_keep > 0:
            chosen = rng.choice(n_total, size=n_keep, replace=replace_train)
            real_demos = [demos[i] for i in chosen]
        else:
            real_demos = []

        random_demos = []
        for rid in range(n_rand):
            src_id = int(rng.integers(0, n_total))
            src = demos[src_id]

            y_ref = src["y"]
            X_ref = src["X"]
            A_ref = src["A"]
            idx_ref = src.get("indices", None)

            y_demo = (rng.random(np.shape(y_ref)) < p_one).astype(np.float32)

            fairness_feats = compute_fairness_features(
                torch.as_tensor(y_ref, dtype=torch.float32),
                torch.as_tensor(y_demo, dtype=torch.float32),
                torch.as_tensor(A_ref, dtype=torch.float32),
                metrics=self.cfg.demonstrator.metrics,
                X=torch.as_tensor(X_ref, dtype=torch.float32),
            ).detach().cpu().numpy()

            random_demos.append({
                "indices": idx_ref,
                "X": X_ref,
                "y": y_ref,
                "A": A_ref,
                "y_demo": y_demo,
                "fairness_feats": fairness_feats,
                "zero_one_loss": zero_one_loss(y_ref, y_demo),
                "source_demo_id": src_id,
                "random_demo_id": rid,
                "is_random_demo": True,
            })

        for d in real_demos:
            d.setdefault("is_random_demo", False)

        mixed = real_demos + random_demos
        rng.shuffle(mixed)
        return mixed


    def _load_or_create_diagnostic_split(self, base_demos, split: str):
        params = self._diagnostic_defaults()

        name = self._build_diagnostic_name(
            split=split,
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

        diag_demos = self._make_diagnostic_split(
            demos=base_demos,
            train_frac=params["train_frac"],
            random_mult=params["random_mult"],
            replace_train=params["replace_train"],
            p_one=params["p_one"],
            seed=seed,
        )

        fairness = self._compute_fairness(diag_demos)
        meta = {
            **(self.meta if hasattr(self, "meta") and self.meta is not None else {}),
            "is_diagnostic": True,
            "diagnostic_split": split,
            "diagnostic_train_frac": params["train_frac"],
            "diagnostic_random_mult": params["random_mult"],
            "diagnostic_replace_train": params["replace_train"],
            "diagnostic_p_one": params["p_one"],
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
        """
        Post-processing step to be called after create_demos().
        Replaces self.train_demos / self.eval_demos with diagnostic versions
        only when the corresponding flags are enabled.
        """
        dcfg = self.cfg.demonstrator
        use_train = bool(getattr(dcfg, "use_diagnostic_train", False))
        use_eval = bool(getattr(dcfg, "use_diagnostic_eval", False))

        if not use_train and not use_eval:
            return {
                "train_demos": self.train_demos,
                "eval_demos": self.eval_demos,
                "metadata": self.meta,
            }

        base_train = self.train_demos
        base_eval = self.eval_demos

        if use_train:
            self.train_demos = self._load_or_create_diagnostic_split(base_train, split="train")

        if use_eval:
            self.eval_demos = self._load_or_create_diagnostic_split(base_eval, split="eval")

        if hasattr(self, "meta") and self.meta is not None:
            self.meta = {
                **self.meta,
                "use_diagnostic_train": use_train,
                "use_diagnostic_eval": use_eval,
                "n_demos_train": len(self.train_demos) if self.train_demos is not None else 0,
                "n_demos_eval": len(self.eval_demos) if self.eval_demos is not None else 0,
            }

        return {
            "train_demos": self.train_demos,
            "eval_demos": self.eval_demos,
            "metadata": self.meta,
        }
