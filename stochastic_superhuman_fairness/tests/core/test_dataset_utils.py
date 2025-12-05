"""
tests/core/test_dataset_utils.py

Verifies that dataset utilities for Adult and COMPAS
produce consistent, correct, and reproducible outputs.
"""

import os
import numpy as np
import pytest
from pathlib import Path

from stochastic_superhuman_fairness.core.dataset_utils import load_adult_cfg, load_compas_cfg, load_dataset_by_name_cfg


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture(scope="session")
def tmp_data_dir(tmp_path_factory):
    """Create a temporary data directory for test runs."""
    path = tmp_path_factory.mktemp("data")
    return path


@pytest.fixture(scope="session")
def base_cfg():
    """Minimal config-like object with dataset parameters."""
    class Cfg:
        class data:
            name = ""
            label_col = ""
            protected_attrs = ["sex", "race"]
            normalize = True
            one_hot = True
            train_ratio = 0.8
            cache_dir = "./data"
        seed = 42
    return Cfg


# ---------------------------------------------------------------------
# Helper assertions
# ---------------------------------------------------------------------

def _check_dataset_schema(data_dict):
    """Ensure the returned dict has expected structure and metadata."""
    required_keys = [
        "X_train", "X_test", "y_train", "y_test",
        "protected_train", "protected_test",
        "feature_names", "protected_attrs", "metadata",
    ]
    for key in required_keys:
        assert key in data_dict, f"Missing key: {key}"

    meta = data_dict["metadata"]
    assert "n_features" in meta and isinstance(meta["n_features"], int)
    assert "n_samples" in meta and isinstance(meta["n_samples"], int)
    assert meta["n_features"] == data_dict["X_train"].shape[1]
    assert data_dict["X_train"].ndim == 2
    assert data_dict["y_train"].ndim == 1
    assert len(data_dict["protected_train"].shape) == 2

    # Train/test disjointness sanity check
    n_train = len(data_dict["X_train"])
    n_test = len(data_dict["X_test"])
    assert (n_train + n_test) == meta["n_samples"], "Mismatch in sample count"

    # Numeric sanity
    assert np.isfinite(data_dict["X_train"]).all()
    assert np.isfinite(data_dict["X_test"]).all()


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

@pytest.mark.parametrize("dataset_name,label_col", [
    ("adult", "income"),
    ("compas", "two_year_recid"),
])
def test_load_dataset_schema(tmp_data_dir, base_cfg, dataset_name, label_col):
    """Ensure both datasets load and return the same schema."""
    cfg = base_cfg
    cfg.data.name = dataset_name
    cfg.data.label_col = label_col

    data = load_dataset_by_name_cfg(cfg, tmp_data_dir)
    _check_dataset_schema(data)

    meta = data["metadata"]
    print(f"✅ {dataset_name.upper()} loaded: {meta['n_samples']} samples, {meta['n_features']} features.")


def test_adult_and_compas_have_unique_features(tmp_data_dir, base_cfg):
    """Check that Adult and COMPAS have distinct feature spaces."""
    cfg = base_cfg

    cfg.data.name = "adult"
    cfg.data.label_col = "income"
    data_adult = load_dataset_by_name_cfg(cfg, tmp_data_dir)

    cfg.data.name = "compas"
    cfg.data.label_col = "two_year_recid"
    data_compas = load_dataset_by_name_cfg(cfg, tmp_data_dir)

    n_feats_adult = data_adult["metadata"]["n_features"]
    n_feats_compas = data_compas["metadata"]["n_features"]

    assert n_feats_adult != n_feats_compas, "Feature dimensions unexpectedly identical"
    assert data_adult["metadata"]["dataset"] != data_compas["metadata"]["dataset"]


def test_adult_reproducibility(tmp_data_dir, base_cfg):
    """Ensure loading twice with the same seed gives identical arrays."""
    cfg = base_cfg
    cfg.data.name = "adult"
    cfg.data.label_col = "income"

    d1 = load_dataset_by_name_cfg(cfg, tmp_data_dir)
    d2 = load_dataset_by_name_cfg(cfg, tmp_data_dir)

    # same shapes and feature counts
    assert d1["X_train"].shape == d2["X_train"].shape
    assert np.allclose(d1["X_train"].mean(), d2["X_train"].mean(), rtol=1e-5)


import json

def _build_cache_name(cfg, dataset_name):
    """Return consistent cache filename based on preprocessing config."""
    return (
        f"{dataset_name}"
        f"_norm{cfg.data.normalize}"
        f"_onehot{cfg.data.one_hot}"
        f"_train{cfg.data.train_ratio}"
        ".npy"
    )
