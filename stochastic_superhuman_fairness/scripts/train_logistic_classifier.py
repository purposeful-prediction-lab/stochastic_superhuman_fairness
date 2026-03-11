# scripts/train_logistic_baseline.py
import argparse
from pathlib import Path
import numpy as np
import torch
import joblib
from omegaconf import OmegaConf

from stochastic_superhuman_fairness.core.baselines.logistic_regression import (
    train_logistic_from_demos,
    logistic_training_settings,
)
from stochastic_superhuman_fairness.core.fairness.fairness_metrics import compute_fairness_features
from stochastic_superhuman_fairness.core.demonstrator import Demonstrator
from stochastic_superhuman_fairness.core.dataset_utils import load_adult_csv, load_compas_csv


def _format_feat_suffix(fairness_feats):
    vals = np.asarray(fairness_feats).reshape(-1)
    return "_".join(f"{x:.4f}".replace(".", "p").replace("-", "m") for x in vals)


def build_default_save_path(model_name: str, fairness_feats):
    suffix = _format_feat_suffix(fairness_feats)
    return f"./data/checkpoints/logistic_model_{model_name}_performance_{suffix}.pkl"


def concat_demo_split(demos):
    X = np.concatenate([_to_numpy(d["X"]) for d in demos], axis=0)
    y = np.concatenate([_to_numpy(d["y"]).reshape(-1) for d in demos], axis=0)
    A = np.concatenate([_to_numpy(d["A"]).reshape(-1) for d in demos], axis=0)
    return X, y, A


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def infer_dataset_name(raw_dataset_path, dataset_name=None):
    if dataset_name is not None:
        dataset_name = dataset_name.lower()
        if dataset_name not in {"adult", "compas"}:
            raise ValueError("dataset_name must be one of {'adult', 'compas'}")
        return dataset_name

    p = str(raw_dataset_path).lower()
    if "adult" in p:
        return "adult"
    if "compas" in p:
        return "compas"

    raise ValueError(
        "Could not infer dataset from raw_dataset_path. "
        "Please provide --dataset_name adult or --dataset_name compas."
    )


def load_raw_dataset_split(args, return_name: bool = False):
    dataset_name = infer_dataset_name(args.raw_dataset_path, args.dataset_name)

    common_args = (
        args.normalize,
        args.one_hot,
        args.train_ratio,
        args.seed,
    )
    common_kwargs = dict( normalize_mode= args.normalize_mode)
    if dataset_name == "adult":
        #  import ipdb;ipdb.set_trace()
        data = load_adult_csv(
            args.raw_dataset_path,
            "income",
            args.protected_attrs,
            args.sensitive_attrs or ["sex"],
            *common_args,
            **common_kwargs,
        )
    elif dataset_name == "compas":
        data = load_compas_csv(
            args.raw_dataset_path,
            "two_year_recid",
            args.protected_attrs,
            args.sensitive_attrs or ["race"],
            *common_args,
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    train_demo = {
        "X": data["X_train"],
        "y": data["y_train"],
    }

    X_eval = data["X_test"]
    y_eval = data["y_test"]
    A_eval = data["sensitive_test"]
    if return_name:
        return train_demo, X_eval, y_eval, A_eval, dataset_name
    return train_demo, X_eval, y_eval, A_eval


def main():
    parser = argparse.ArgumentParser()

    # input source
    parser.add_argument("--cfg_path", type=str, default=None)
    parser.add_argument("--raw_dataset_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None, help="adult or compas if not inferable from path")

    # fairness
    DEFAULT_METRICS = ["D.DP", "D.EqOdds", "D.PRP", "D.Err", "L.ZeroOne"]
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, help="Fairness metrics to compute (default: all metrics).")

    # dataset loader args
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--no_normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=True)

    parser.add_argument("--one_hot", action="store_true")
    parser.add_argument("--no_one_hot", dest="one_hot", action="store_false")
    parser.set_defaults(one_hot=True)

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize_mode", type=str, default="continuous")
    parser.add_argument("--protected_attrs", nargs="*", default=None)
    parser.add_argument("--sensitive_attrs", nargs="*", default=None)

    # logistic args

    parser.add_argument("--preset_qual", choices=["poor", "intermediate", 'high'], default=None)
    parser.add_argument("--solver", type=str, default="lbfgs")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--n_jobs", type=int, default=None)
    parser.add_argument("--fit_intercept", action="store_true")
    parser.add_argument("--no_fit_intercept", dest="fit_intercept", action="store_false")
    parser.set_defaults(fit_intercept=True)
    parser.add_argument("--class_weight", type=str, default=None)
    parser.add_argument("--random_state", type=int, default=None)

    # save
    parser.add_argument("--save_path", type=str, default=None)

    args = parser.parse_args()

    if (args.cfg_path is None) == (args.raw_dataset_path is None):
        raise ValueError("Exactly one of --cfg_path or --raw_dataset_path must be provided.")

    # If a preset qual setting is given use those settings
    preset_tag = str(args.preset_qual) if args.preset_qual is not None else ''
    if preset_tag is not None:
        settings = logistic_training_settings(preset_tag)
    else:
        settings = {
            "solver": args.solver,
            "C": args.C,
            "max_iter": args.max_iter,
            "n_jobs": args.n_jobs,
            "fit_intercept": args.fit_intercept,
            "class_weight": args.class_weight,
            "random_state": args.random_state,
        }
    settings['save_path'] = None
    # --------------------------------------------------
    # Path 1: Demonstrator
    # --------------------------------------------------
    if args.cfg_path is not None:
        cfg = OmegaConf.load(args.cfg_path)

        print("📦 Initializing Demonstrator...")
        demo = Demonstrator(cfg)

        train_demos = demo.train_demos
        eval_demos = demo.eval_demos
        model_name = demo.dataset

        model = train_logistic_from_demos(
            train_demos,
            **settings,
            )

        X_eval, y_eval, A_eval = concat_demo_split(eval_demos)
        y_demo = model.predict(X_eval)

    # --------------------------------------------------
    # Path 2: Raw dataset loaders
    # --------------------------------------------------
    else:
        train_demo, X_eval, y_eval, A_eval, model_name = load_raw_dataset_split(args, return_name = True)
        model = train_logistic_from_demos(
            [train_demo],
            **settings,
        )

        y_demo = model.predict(X_eval)

    fairness_feats = compute_fairness_features(
        torch.as_tensor(y_eval, dtype=torch.float32),
        torch.as_tensor(y_demo, dtype=torch.float32),
        torch.as_tensor(A_eval, dtype=torch.float32),
        metrics=args.metrics,
        X=torch.as_tensor(X_eval, dtype=torch.float32),
    ).detach().cpu().numpy()

    save_path = args.save_path
    if save_path is None:
        save_path = build_default_save_path(model_name+preset_tag, fairness_feats,)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)

    print(f"Saved model to: {save_path.resolve()}")
    print("fairness_feats:", fairness_feats)


if __name__ == "__main__":
    main()
