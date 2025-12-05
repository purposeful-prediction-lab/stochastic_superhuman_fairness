import numpy as np
import pytest
from pathlib import Path
from omegaconf import OmegaConf

from stochastic_superhuman_fairness.core.demonstrator import Demonstrator
from stochastic_superhuman_fairness.core.utils_io import safe_json_load


@pytest.fixture(scope="function")
def tmp_cfg(tmp_path):
    cfg = OmegaConf.create({
        "seed": 123,
        "device":'cuda',
        "demonstrator": {
            "dataset": "adult",
            "demo_size": 10,
            "compute_global": True,
            "normalize": False,
            "overwrite": True,
            "metrics": ["D.DP", "D.Err"],
            "cache_dir": str(tmp_path),
            "protected_attrs": ["race"],
            "sensitive_attrs": ["sex"],
        }
    })
    return cfg


def _assert_metadata(meta, sample_id, mode):
    assert meta["dataset"] == "adult"
    assert meta["sample_id"] == sample_id
    assert "n_demos_train" in meta
    assert "n_demos_eval" in meta
    assert isinstance(meta["protected_attrs"], list)
    assert isinstance(meta["sensitive_attrs"], list)
    if mode == "zip":
        assert meta["n_features"] > 0


def test_create_and_reload_separate(tmp_cfg):
    tmp_cfg.demonstrator.save_format = "separate"

    demo = Demonstrator(tmp_cfg)
    out1 = demo.create_demos()
    meta1 = out1["metadata"]
    _assert_metadata(meta1, 1, "separate")

    npy = Path(tmp_cfg.demonstrator.cache_dir) / "adult" / f"adult_demo_size10_globTrue_normFalse_sample1.npy"
    meta = npy.with_name(npy.stem + "_meta.json")
    assert npy.exists() and meta.exists()

    tmp_cfg.demonstrator.demo_sample = 1
    demo2 = Demonstrator(tmp_cfg)
    out2 = demo2.create_demos()
    assert np.allclose(out1["train_demos"][0]["y"], out2["train_demos"][0]["y"])

    re_meta = safe_json_load(meta)
    assert re_meta["sample_id"] == 1


def test_resample_increments_sample_id(tmp_cfg):
    demo = Demonstrator(tmp_cfg)
    demo.create_demos()
    demo.resample()
    npy2 = Path(tmp_cfg.demonstrator.cache_dir) / "adult" / f"adult_demo_size10_globTrue_normFalse_sample2.npy"
    assert npy2.exists()
    meta2 = safe_json_load(npy2.with_name(npy2.stem + "_meta.json"))
    _assert_metadata(meta2, 2, "separate")


def test_create_and_reload_zip(tmp_cfg):
    tmp_cfg.demonstrator.save_format = "zip"
    demo = Demonstrator(tmp_cfg)
    out = demo.create_demos()
    meta = out["metadata"]
    _assert_metadata(meta, 1, "zip")

    npz = Path(tmp_cfg.demonstrator.cache_dir) / "adult" / "adult_demo_size10_globTrue_normFalse_sample1.npz"
    assert npz.exists()

    re = demo._load_sample(1)
    assert "metadata" in re and re["metadata"]["sample_id"] == 1


def test_in_memory_consistency(tmp_cfg):
    demo = Demonstrator(tmp_cfg)
    out = demo.create_demos()
    td, ed = demo.train_demos[0], demo.eval_demos[0]
    assert td["X"].shape[0] == tmp_cfg.demonstrator.demo_size
    assert "fairness_feats" in td
    assert isinstance(td["fairness_feats"], np.ndarray)


@pytest.mark.parametrize("dataset", ["adult", "compas"])
def test_normalization_behavior_consistency(tmp_cfg, dataset):
    tmp_cfg.demonstrator.dataset = dataset
    tmp_cfg.demonstrator.demo_size = 20
    tmp_cfg.demonstrator.metrics = ["D.DP"]
    tmp_cfg.demonstrator.compute_global = True

    tmp_cfg.demonstrator.normalize = False
    demo_raw = Demonstrator(tmp_cfg)
    data_raw = demo_raw.create_demos()
    X_raw_train = np.vstack([d["X"] for d in data_raw["train_demos"]])

    tmp_cfg.demonstrator.normalize = True
    demo_norm = Demonstrator(tmp_cfg)
    data_norm = demo_norm.create_demos()
    X_norm_train = np.vstack([d["X"] for d in data_norm["train_demos"]])

    assert abs(np.mean(X_norm_train)) < 0.05
    assert 0.8 < np.std(X_norm_train) < 1.2
    assert data_norm["metadata"]["normalize"]
    assert "global" in data_norm["fairness"]


def test_step_and_reset_behavior(tmp_cfg):
    tmp_cfg.demonstrator.normalize = True
    demo = Demonstrator(tmp_cfg)
    demo.create_demos()
    n_total = len(demo.train_demos)

    demos1 = demo.step(n_samples=5)
    demos2 = demo.step(n_samples=10)
    assert isinstance(demos1[0], dict)
    assert len(demo._seen["train"]) == 15

    while True:
        try:
            demo.step(n_samples=50)
        except StopIteration:
            break
    assert demo.remaining("train") == 0
    demo.reset("train")
    assert demo.remaining("train") == n_total


def test_iter_batches_returns_demo_lists(tmp_cfg):
    tmp_cfg.demonstrator.normalize = True
    demo = Demonstrator(tmp_cfg)
    demo.create_demos()
    batches = list(demo.iter_batches(batch_size=2))
    assert isinstance(batches[0], list)
    assert all(isinstance(d, dict) for d in batches[0])
    assert "fairness_feats" in batches[0][0]


def test_protected_removed_sensitive_kept(tmp_cfg):
    tmp_cfg.demonstrator.normalize = False
    tmp_cfg.demonstrator.protected_attrs = ["race"]
    tmp_cfg.demonstrator.sensitive_attrs = ["sex"]

    demo = Demonstrator(tmp_cfg)
    data = demo.create_demos()
    td = data["train_demos"][0]
    meta = data["metadata"]

    assert "A" in td and td["A"] is not None
    assert "fairness_feats" in td
    assert meta["sensitive_attrs"] == ["sex"]
    assert "race" in meta["protected_attrs"]
    assert np.isfinite(td["A"]).all()


def test_demo_fairness_feature_vector_shape(tmp_cfg):
    """
    Ensure each demo in the demonstrator stores a 1D fairness feature vector.
    The vector corresponds to the fairness metrics configured for the dataset.
    """
    tmp_cfg.demonstrator.normalize = False
    tmp_cfg.demonstrator.metrics = ["D.DP", "D.Err"]

    demo = Demonstrator(tmp_cfg)
    out = demo.create_demos()

    train_demos = out["train_demos"]

    # Check each demo has fairness_feats and that it is 1D
    for d in train_demos:
        fairness_feats = d.get("fairness_feats", None)
        assert fairness_feats is not None, "Demo missing fairness_feats"
        assert isinstance(fairness_feats, np.ndarray), "fairness_feats should be a NumPy array"
        assert fairness_feats.ndim == 1, (
            f"fairness_feats for demo not 1D (got shape {fairness_feats.shape})"
        )
        assert np.all(np.isfinite(fairness_feats)), "fairness_feats contain NaNs or infs"

    # Each vector length should equal number of fairness metrics
    expected_len = len(tmp_cfg.demonstrator.metrics)
    actual_len = train_demos[0]["fairness_feats"].shape[0]
    assert actual_len == expected_len, (
        f"Expected fairness vector length {expected_len}, got {actual_len}"
    )
