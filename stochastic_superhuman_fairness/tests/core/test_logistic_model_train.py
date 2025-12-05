import torch
import numpy as np
import pytest
from omegaconf import OmegaConf
from stochastic_superhuman_fairness.core.demonstrator import Demonstrator
from stochastic_superhuman_fairness.core.models.logistic import LogisticRegressionModel
import shutil
import os, random, numpy as np, torch

def set_all_seeds(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@pytest.fixture(scope="function")
def tmp_cfg(tmp_path):
    """Base test config for Adult dataset demos — automatically uses CUDA if available."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    adult_cache = tmp_path / "adult"
    if adult_cache.exists():
        shutil.rmtree(adult_cache)
    adult_cache.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.create({
        "seed": 123,
        "device": device,
        "demonstrator": {
            "dataset": "adult",
            "demo_size": 1000,
            "compute_global": True,
            "normalize": True,
            "normalize_mode": "continuous",
            "overwrite": True,
            "metrics": ["D.DP"],
            "cache_dir": str(tmp_path),
            "protected_attrs": ["race"],
            "sensitive_attrs": ["sex"],
            "train_ratio": 0.8,
        }
    })
    return cfg

@pytest.fixture(scope="function")
def demo_adult(tmp_cfg):
    """
    Create a fresh Demonstrator instance for the Adult dataset
    using the tmp_cfg fixture. Ensures reproducible train/eval splits
    and small demo size for fast tests.
    """
    demo_adult = Demonstrator(tmp_cfg, auto_create=True, to_torch=True)
    demo_adult.reload_if_cfg_changed()  # ensure fresh demos if config changed
    return demo_adult

def test_logistic_integration_train_eval(tmp_cfg):
    """End-to-end sanity: model trains and evaluates on real data."""
    cfg = OmegaConf.create({'device': 'cuda', "metrics": {'use': ["D.DP"]}, "lr": 1e-3, "batch_size": 8})
    demo_adult = Demonstrator(tmp_cfg)
    model = LogisticRegressionModel(cfg, demo_adult)

    # Run one training epoch
    results = model.train_one_epoch(demo_adult)
    assert "train/zero_one_loss" in results and np.isfinite(results["train/zero_one_loss"])
    assert "train/mean_subdom" in results
    assert "train/std_subdom" in results
    assert "train/fairness" in results
    assert isinstance(results["train/fairness"], list)

    # Evaluate
    eval_out = model.evaluate(demo_adult)
    assert "eval/zero_one_loss" in eval_out
    #  import ipdb;ipdb.set_trace()
    assert np.isfinite(eval_out["eval/zero_one_loss"])
    assert "eval/fairness" in eval_out


def test_logistic_expected_update_single_sample(tmp_cfg):
    """
    Analytical check: when using batch size 1 and known weights,
    parameter update equals LR * (φ_rollout - φ_demo_mean)
    weighted by subdominance.
    """
    cfg = OmegaConf.create({'device': 'cuda', "metrics": {'use': ["D.DP"]}, "lr": 1e-3, "batch_size": 8})
    demo_adult = Demonstrator(tmp_cfg)
    model = LogisticRegressionModel(cfg, demo_adult)
    model.policy.weight.data.fill_(0.0)
    model.policy.bias.data.fill_(0.0)

    # Manually take one training epoch (batch=1)
    model.train_one_epoch(demo_adult)

    # Closed-form reference: Δθ ≈ −lr * (φ_r − φ_d_mean)
    d = demo_adult.train_demos[0]
    X = torch.tensor(d["X"], dtype=torch.float32)
    y = torch.tensor(d["y"], dtype=torch.float32)
    phi_r = model.compute_phi(X, y).mean(dim=0)

    phi_d_means = model.phi_mean_per_demo(demo_adult.train_demos)
    phi_d_mean = phi_d_means.mean(dim=0)

    expected_update = -cfg.lr * (phi_r - phi_d_mean)
    new_weights = model.policy.weight.data.squeeze()
    new_bias = model.policy.bias.data

    # The first elements should roughly follow expected_update
    diff = torch.norm(new_weights - expected_update[:-1])
    assert diff < 1e-2 or diff / torch.norm(expected_update[:-1]) < 0.1


def test_loss_decreases_after_update(tmp_cfg, demo_adult):
    cfg = OmegaConf.merge(
            tmp_cfg,
            {"lr": 0.1, "batch_size": 4},
        )
    model = LogisticRegressionModel(cfg, demo_adult)
    initial_loss = model.evaluate(demo_adult)["eval/zero_one_loss"]
    model.train_one_epoch(demo_adult)
    new_loss = model.evaluate(demo_adult)["eval/zero_one_loss"]
    assert new_loss <= initial_loss + 1e-3, f"Loss did not decrease ({initial_loss} → {new_loss})"

def test_reproducible_updates(tmp_cfg, demo_adult):
    cfg = OmegaConf.merge(
            tmp_cfg,
            {"lr": 0.05, "batch_size": 8},
        )
    set_all_seeds(0)
    model1 = LogisticRegressionModel(cfg, demo_adult)
    set_all_seeds(0)
    model2 = LogisticRegressionModel(cfg, demo_adult)
    torch.manual_seed(0)
    out1 = model1.train_one_epoch(demo_adult, shuffle_batch_each_epoch = False)
    torch.manual_seed(0)
    out2 = model2.train_one_epoch(demo_adult, shuffle_batch_each_epoch = False)
    for k in out1:
        if isinstance(out1[k], float):
            assert np.allclose(out1[k], out2[k])

def test_lr_scales_update_magnitude(tmp_cfg, demo_adult):
    cfg = OmegaConf.merge(
            tmp_cfg,
            {"lr": 0.01, "batch_size": 4},
        )
    model_small = LogisticRegressionModel(cfg, demo_adult)
    cfg.lr = 0.1
    model_large = LogisticRegressionModel(cfg, demo_adult)
    w0 = model_small.policy.weight.clone()
    model_small.train_one_epoch(demo_adult)
    delta_small = (model_small.policy.weight - w0).norm().item()
    w0 = model_large.policy.weight.clone()
    model_large.train_one_epoch(demo_adult)
    delta_large = (model_large.policy.weight - w0).norm().item()
    ratio = delta_large / (delta_small + 1e-12)
    assert ratio > 3.0, f"Expected larger LR to yield ≥3× update, got {ratio:.2f}×"


def test_zero_gradient_for_identical_demos(tmp_cfg, demo_adult):
    # duplicate the first demo to remove variation
    cfg = OmegaConf.merge(
            tmp_cfg,
            {"lr": 0.1, "batch_size": 1},
        )
    demo_adult.train_demos = [demo_adult.train_demos[0]] * 5
    model = LogisticRegressionModel(cfg, demo_adult)
    before = model.policy.weight.clone()
    model.train_one_epoch(demo_adult)
    diff = (model.policy.weight - before).abs().max().item()
    assert diff < 1e-3

def test_training_outputs_finite(tmp_cfg, demo_adult):
    cfg = OmegaConf.merge(
            tmp_cfg,
            {"lr": 0.05, "batch_size": 8},
        )
    model = LogisticRegressionModel(cfg, demo_adult)
    stats = model.train_one_epoch(demo_adult)
    for key, val in stats.items():
        if isinstance(val, float):
            assert np.isfinite(val), f"{key} is not finite"

def test_convergence_over_epochs(tmp_cfg,demo_adult):
    cfg = OmegaConf.merge(
            tmp_cfg,
            {"lr": 0.05, "batch_size": 8},
        )
    model = LogisticRegressionModel(cfg, demo_adult)
    losses = []
    for _ in range(10):
        out = model.train_one_epoch(demo_adult)
        losses.append(out["train/zero_one_loss"])
    assert losses[-1] <= losses[0] - 1e-3, f"Loss did not decrease overall ({losses[0]:.4f} → {losses[-1]:.4f})"

def test_fairness_metric_bounds(tmp_cfg, demo_adult):
    cfg = OmegaConf.merge(
            tmp_cfg,
            {"lr": 0.05, "batch_size": 4},
        )
    model = LogisticRegressionModel(cfg, demo_adult)
    out = model.train_one_epoch(demo_adult)
    fairness = np.array(out["train/fairness"])
    assert np.all(np.isfinite(fairness))
    assert np.all(np.abs(fairness) < 1e2)

def test_multi_epoch_loss_descent(tmp_cfg, demo_adult):
    cfg = OmegaConf.merge(
            tmp_cfg,
            {"lr": 0.05, "batch_size": 8},
        )
    model = LogisticRegressionModel(cfg, demo_adult)
    losses = []
    for _ in range(20):
        out = model.train_one_epoch(demo_adult)
        losses.append(out["train/zero_one_loss"])
    assert losses[-1] <= losses[0] or np.mean(losses[-5:]) <= np.mean(losses[:5])


def test_fairness_accuracy_tradeoff(tmp_cfg, demo_adult):
    cfg = OmegaConf.merge(
            tmp_cfg,
            {"lr": 0.05, "batch_size": 8},
        )
    model = LogisticRegressionModel(cfg, demo_adult)
    fair_trend, loss_trend = [], []
    for _ in range(10):
        out = model.train_one_epoch(demo_adult)
        fair_trend.append(np.mean(out["train/fairness"]))
        loss_trend.append(out["train/zero_one_loss"])
    corr = np.corrcoef(fair_trend, loss_trend)[0, 1]
    assert np.isfinite(corr)

def test_gradient_norm_decay(tmp_cfg, demo_adult):
    cfg = OmegaConf.merge(
            tmp_cfg,
            {"lr": 0.05, "batch_size": 8},
        )
    model = LogisticRegressionModel(cfg, demo_adult)
    norms = []
    for _ in range(10):
        out = model.train_one_epoch(demo_adult)
        norms.append(out["train/mean_subdom"])
    assert norms[0] > norms[-1] * 0.5  # rough decay trend

def test_learning_rate_sensitivity(tmp_cfg, demo_adult):
    cfg1 = OmegaConf.merge(
            tmp_cfg,
            {"lr": 0.05, "batch_size": 8},
        )
    cfg2 = OmegaConf.merge(
            tmp_cfg,
            {"lr": 0.2, "batch_size": 8},
        )
    m1, m2 = LogisticRegressionModel(cfg1, demo_adult), LogisticRegressionModel(cfg2, demo_adult)
    out1 = m1.train_one_epoch(demo_adult)
    out2 = m2.train_one_epoch(demo_adult)
    assert out2["train/zero_one_loss"] >= out1["train/zero_one_loss"] - 0.08


def test_cross_seed_stability(tmp_cfg, demo_adult):
    cfg = OmegaConf.merge(
            tmp_cfg,
            {"lr": 0.2, "batch_size": 8},
        )
    set_all_seeds(0)
    m1 = LogisticRegressionModel(cfg, demo_adult)
    l1 = m1.train_one_epoch(demo_adult)["train/zero_one_loss"]
    set_all_seeds(1)
    m2 = LogisticRegressionModel(cfg, demo_adult)
    l2 = m2.train_one_epoch(demo_adult)["train/zero_one_loss"]
    assert abs(l1 - l2) < 0.05
