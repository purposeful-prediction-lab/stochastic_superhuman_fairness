"""
core/dataset_utils.py

Primitive data loading and preprocessing utilities
for specific fairness datasets (Adult, COMPAS, etc.).
Each dataset loader returns a standardized dict structure.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def safe_train_test_split(X, y, protected_np, train_ratio=0.8, seed=42):
    """
    Splits X, y, (optional) protected attributes consistently.
    If protected_np is None or empty, returns None for its splits.
    """
    test_size = 1 - train_ratio

    if protected_np is None or len(protected_np) == 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, shuffle=True
        )
        prot_train = prot_test = None
    else:
        X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
            X, y, protected_np, test_size=test_size, random_state=seed, shuffle=True
        )
    y_train = np.asarray(y_train, dtype = np.float32).reshape(-1)
    y_test = np.asarray(y_test, dtype = np.float32).reshape(-1)
    return X_train, X_test, y_train, y_test, prot_train, prot_test

# =====================================================================
# 1. ADULT DATASET
# =====================================================================
def load_adult_cfg(cfg, data_dir="./data"):
    """Wrapper that extracts fields from cfg and calls load_adult_old."""
    data_cfg = getattr(cfg, "data", cfg)
    return load_adult(
        data_dir=data_dir,
        label_col=getattr(data_cfg, "label_col", "income"),
        protected_attrs=getattr(data_cfg, "protected_attrs", []),
        sensitive_attrs=getattr(data_cfg, "sensitive_attrs", []),
        normalize=getattr(data_cfg, "normalize", True),
        one_hot=getattr(data_cfg, "one_hot", True),
        train_ratio=getattr(data_cfg, "train_ratio", 0.8),
        seed=getattr(cfg, "seed", 42),
        normalize_mode=getattr(data_cfg, "normalize_mode", "continuous"),
    )

def load_adult(
    data_dir,
    label_col,
    protected_attrs,
    sensitive_attrs,
    normalize,
    one_hot,
    train_ratio,
    seed,
    normalize_mode="continuous",
):
    """
    Core Adult dataset loader (legacy-compatible signature).

    - Protected attributes: removed from X.
    - Sensitive attributes: retained in X and populate A.
    """
    dataset_path = Path(data_dir) / "adult"
    dataset_path.mkdir(parents=True, exist_ok=True)

    csv_path = dataset_path / "adult_raw.csv"
    if not csv_path.exists():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        df = pd.read_csv(url, header=None)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)

    df.columns = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"
    ]

    # subsets
    protected = df[protected_attrs].copy() if protected_attrs else pd.DataFrame()
    sensitive = df[sensitive_attrs].copy() if sensitive_attrs else pd.DataFrame()

    # remove protected columns from X but keep sensitive
    #  import ipdb;ipdb.set_trace()
    features = df.drop(columns=protected_attrs + [label_col]) if not protected.empty else df.drop(columns=[label_col])
    y = (df[label_col].str.strip() == ">50K").astype(np.float32)

    # one-hot encode
    if one_hot:
        features = pd.get_dummies(features, drop_first=True)

    for col in features.columns:
        if not np.issubdtype(features[col].dtype, np.number):
            features[col] = pd.Categorical(features[col]).codes

    # before splitting
    features = features.reset_index(drop=True)
    y = y.reset_index(drop=True)
    if not protected.empty:
        protected = protected.reset_index(drop=True)
    if not sensitive.empty:
        sensitive = sensitive.reset_index(drop=True)
    features = features.apply(pd.to_numeric, errors="coerce").fillna(0)

    # normalization (continuous only)
    unique_counts = features.nunique()
    is_cont = unique_counts > 2
    cont_cols = features.columns[is_cont]
    scaler = None
    if normalize and normalize_mode != "none":
        scaler = StandardScaler()
        if normalize_mode == "continuous":
            features.loc[:, cont_cols] = scaler.fit_transform(features[cont_cols])
        elif normalize_mode == "all":
            features.loc[:, :] = scaler.fit_transform(features.to_numpy())

    X = features.astype(np.float32).to_numpy()

    # encode protected / sensitive
    def encode(df_):
        if df_.empty:
            return None
        return (
            df_.apply(pd.Categorical)
            .apply(lambda s: s.cat.codes)
            .to_numpy(np.float32)
        )

    protected_np = encode(protected)
    sensitive_np = encode(sensitive)

    # split
    Xtr, Xte, ytr, yte, A_tr, A_te = safe_train_test_split(X, y, sensitive_np, train_ratio, seed)
    _, _, _, _, prot_tr, prot_te = safe_train_test_split(X, y, protected_np, train_ratio, seed)

    meta = {
        "dataset": "adult",
        "n_samples": len(X),
        "n_features": X.shape[1],
        "protected_attrs": protected_attrs,
        "sensitive_attrs": sensitive_attrs,
        "normalize_mode": normalize_mode,
    }

    return {
        "X_train": Xtr,
        "X_test": Xte,
        "y_train": ytr,
        "y_test": yte,
        "protected_train": prot_tr,
        "protected_test": prot_te,
        "sensitive_train": A_tr,
        "sensitive_test": A_te,
        "feature_names": list(features.columns),
        "scaler": scaler,
        "metadata": meta,
    }

def load_adult_old(data_dir, label_col, protected_attrs, sensitive_attrs, normalize, one_hot, train_ratio, seed, normalize_mode = "continuous"):
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / "adult_raw.csv"
    if not raw_path.exists():
        _download_adult(raw_path)

    df = pd.read_csv(raw_path, header=None, na_values=" ?", skipinitialspace=True)
    df.columns = _adult_column_names()
    df = _clean_adult(df)

    protected_attrs = [] if protected_attrs is None else protected_attrs
    protected = df[protected_attrs].copy()
    features = df.drop(columns=protected_attrs + [label_col])
    y = (df[label_col] == ">50K").astype(np.float32)

    # One-hot / categorical encoding
    if one_hot:
        features = pd.get_dummies(features, drop_first=True)

    # Ensure numeric
    for col in features.columns:
        if not np.issubdtype(features[col].dtype, np.number):
            features[col] = pd.Categorical(features[col]).codes

    features = features.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Determine column types
    unique_counts = features.nunique()
    is_cont = unique_counts > 2
    cont_cols = features.columns[is_cont]
    cat_cols = features.columns[~is_cont]

    X = features.astype(np.float32).copy()
    scaler = None
    scaled_col_indices = []

    # --- Normalization logic ---
    if normalize and normalize_mode != "none":
        scaler = StandardScaler()
        if normalize_mode == "continuous":
            X.loc[:, cont_cols] = scaler.fit_transform(X[cont_cols].to_numpy())
            scaled_col_indices = [features.columns.get_loc(c) for c in cont_cols]
        elif normalize_mode == "all":
            X.loc[:, :] = scaler.fit_transform(X.to_numpy())
            scaled_col_indices = list(range(X.shape[1]))

    X = X.to_numpy(np.float32)
    

    # Encode protected attributes
    protected_np = (protected.apply(pd.Categorical).apply(lambda s: s.cat.codes).to_numpy(np.float32))
    # reset indices to make array indexing safe
    import ipdb;ipdb.set_trace()

    X_train, X_test, y_train, y_test, prot_train, prot_test = safe_train_test_split(
        X, y, protected_np, train_ratio=train_ratio, seed=seed
    )
    y_train = np.array(y_train).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)
    prot_train = np.array(prot_train).astype(np.float32)
    prot_test = np.array(prot_test).astype(np.float32)


    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "protected_train": prot_train,
        "protected_test": prot_test,
        "feature_names": list(features.columns),
        "protected_attrs": protected_attrs,
        "scaler": scaler,
        "scaled_col_indices": scaled_col_indices,
        "cont_cols": list(cont_cols),
        "cat_cols": list(cat_cols),
        "metadata": {
            "dataset": "adult",
            "n_features": X.shape[1],
            "n_samples": len(X),
        },
    }


# =====================================================================
# 2. COMPAS DATASET
# =====================================================================

def load_compas_cfg(cfg, data_dir="./data"):
    """Wrapper to load COMPAS from a Hydra/OmegaConf config."""
    data_cfg = getattr(cfg, "data", cfg)
    return load_compas_old(
        data_dir=data_dir,
        label_col=getattr(data_cfg, "label_col", "two_year_recid"),
        protected_attrs=getattr(data_cfg, "protected_attrs", []),
        sensitive_attrs=getattr(data_cfg, "sensitive_attrs", []),
        normalize=getattr(data_cfg, "normalize", True),
        one_hot=getattr(data_cfg, "one_hot", True),
        train_ratio=getattr(data_cfg, "train_ratio", 0.8),
        seed=getattr(cfg, "seed", 42),
        normalize_mode=getattr(data_cfg, "normalize_mode", "continuous"),
    )

def load_compas(
    data_dir,
    label_col,
    protected_attrs,
    sensitive_attrs,
    normalize,
    one_hot,
    train_ratio,
    seed,
    normalize_mode="continuous",
):
    """
    Load COMPAS dataset (legacy-compatible signature).
    - Protected attrs: removed from X entirely.
    - Sensitive attrs: kept in X and used for A (sensitive variable).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / "compas_raw.csv"
    if not raw_path.exists():
        _download_compas(raw_path)

    df = pd.read_csv(raw_path)
    df = _clean_compas(df)

    # extract protected/sensitive
    protected = df[protected_attrs].copy() if protected_attrs else pd.DataFrame()
    sensitive = df[sensitive_attrs].copy() if sensitive_attrs else pd.DataFrame()

    # remove protected columns from features
    features = df.drop(columns=protected_attrs + [label_col]) if not protected.empty else df.drop(columns=[label_col])
    y = df[label_col].astype(np.float32)

    # one-hot encode categoricals
    if one_hot:
        features = pd.get_dummies(features, drop_first=True)
    for col in features.columns:
        if not np.issubdtype(features[col].dtype, np.number):
            features[col] = pd.Categorical(features[col]).codes
    features = features.apply(pd.to_numeric, errors="coerce").fillna(0)

    # normalize
    unique_counts = features.nunique()
    cont_cols = features.columns[unique_counts > 2]
    scaler = None
    if normalize and normalize_mode != "none":
        scaler = StandardScaler()
        if normalize_mode == "continuous":
            features.loc[:, cont_cols] = scaler.fit_transform(features[cont_cols])
        elif normalize_mode == "all":
            features.loc[:, :] = scaler.fit_transform(features.to_numpy())

    X = features.astype(np.float32).to_numpy()

    # encode helper
    def encode(df_):
        if df_.empty:
            return None
        return (
            df_.apply(pd.Categorical)
            .apply(lambda s: s.cat.codes)
            .to_numpy(np.float32)
        )

    protected_np = encode(protected)
    sensitive_np = encode(sensitive)

    # split (A = sensitive attributes)
    Xtr, Xte, ytr, yte, A_tr, A_te = safe_train_test_split(X, y, sensitive_np, train_ratio, seed)
    _, _, _, _, prot_tr, prot_te = safe_train_test_split(X, y, protected_np, train_ratio, seed)

    # reset indices
    ytr, yte = np.array(ytr, dtype=np.float32), np.array(yte, dtype=np.float32)
    prot_tr = np.array(prot_tr, dtype=np.float32) if prot_tr is not None else None
    prot_te = np.array(prot_te, dtype=np.float32) if prot_te is not None else None
    A_tr = np.array(A_tr, dtype=np.float32) if A_tr is not None else None
    A_te = np.array(A_te, dtype=np.float32) if A_te is not None else None

    meta = {
        "dataset": "compas",
        "n_features": X.shape[1],
        "n_samples": len(X),
        "protected_attrs": protected_attrs,
        "sensitive_attrs": sensitive_attrs,
        "normalize_mode": normalize_mode,
    }

    return {
        "X_train": Xtr,
        "X_test": Xte,
        "y_train": ytr,
        "y_test": yte,
        "protected_train": prot_tr,
        "protected_test": prot_te,
        "sensitive_train": A_tr,
        "sensitive_test": A_te,
        "feature_names": list(features.columns),
        "protected_attrs": protected_attrs,
        "scaler": scaler,
        "metadata": meta,
    }

def load_compas_old(data_dir, label_col, protected_attrs, normalize, one_hot, train_ratio, seed):
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / "compas_raw.csv"
    if not raw_path.exists():
        _download_compas(raw_path)

    df = pd.read_csv(raw_path)
    df = _clean_compas(df)

    protected = df[protected_attrs].copy()
    features = df.drop(columns=protected_attrs + [label_col])
    y = df[label_col].astype(np.float32)

    # One-hot / categorical encoding
    if one_hot:
        features = pd.get_dummies(features, drop_first=True)
    for col in features.columns:
        if not np.issubdtype(features[col].dtype, np.number):
            features[col] = pd.Categorical(features[col]).codes
    features = features.apply(pd.to_numeric, errors="coerce").fillna(0)

    X = features.astype(np.float32).to_numpy()
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    protected_np = protected.apply(pd.Categorical).apply(lambda s: s.cat.codes).to_numpy(np.float32)

    X_train, X_test, y_train, y_test, prot_train, prot_test = safe_train_test_split(
            X, y, protected_np, train_ratio=train_ratio, seed=seed
        )
    # reset indices to make array indexing safe
    y_train = np.array(y_train).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)
    prot_train = np.array(prot_train).astype(np.float32)
    prot_test = np.array(prot_test).astype(np.float32)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "protected_train": prot_train,
        "protected_test": prot_test,
        "feature_names": list(features.columns),
        "protected_attrs": protected_attrs,
        "scaler": scaler,
        "metadata": {
            "dataset": "compas",
            "n_features": X.shape[1],
            "n_samples": len(X),
        },
    }

def _download_adult(save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    df = pd.read_csv(url, header=None, na_values=" ?", skipinitialspace=True)
    df.to_csv(save_path, index=False)
    print(f"⬇️ Downloaded Adult dataset to {save_path}")


def _clean_adult(df):
    df = df.dropna()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
    return df


def _adult_column_names():
    return [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country",
        "income",
    ]
# ---------------------------------------------------------
# Helpers for COMPAS
# ---------------------------------------------------------
def _download_compas(save_path):
    """Download or load the COMPAS dataset from ProPublica GitHub."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    df = pd.read_csv(url)
    df.to_csv(save_path, index=False)
    print(f"⬇️ Downloaded COMPAS dataset to {save_path}")


def _clean_compas(df):
    """Lenient cleaning: keep label rows, fill others."""

    # Strip whitespace
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()

    # Drop rows with missing target only
    if "two_year_recid" in df.columns:
        df = df[~df["two_year_recid"].isna()]

    # Fill remaining NaNs with 0 or "Unknown"
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(0)

    # Optional: standard ProPublica subset for comparability
    keep_cols = [
        "sex", "race", "age", "age_cat", "juv_fel_count", "juv_misd_count",
        "juv_other_count", "priors_count", "c_charge_degree", "two_year_recid"
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    return df
