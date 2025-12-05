import hydra
from omegaconf import DictConfig, OmegaConf
import os

from stochastic_superhuman_fairness.core.demonstrator import Demonstrator
from stochastic_superhuman_fairness.core.learner import Learner
from stochastic_superhuman_fairness.core.logger import Logger
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

@hydra.main(
    version_base=None,
    #config_path="../configs",
    config_path=str(CONFIG_DIR),
    config_name="default"
)
def main(cfg: DictConfig):
    """
    Main training entry point.
    Loads config via Hydra and runs the full learner schedule.
    """
    print("🧩 Configuration:\n", OmegaConf.to_yaml(cfg))

    exp_name = cfg.get("exp_name", "fairness_experiment")
    logger = Logger(log_dir=cfg.get("log_dir", "./logs"), exp_name=exp_name)

    print("📦 Initializing Demonstrator...")
    demo = Demonstrator(cfg)

    print("🧠 Building Learner...")
    learner = Learner(cfg, demonstrator=demo, logger=logger)

    print("🚀 Starting training...")
    learner.run()

    logger.close()


if __name__ == "__main__":
    main()
