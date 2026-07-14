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

    exp_name = cfg.get("exp_name", None)
    if exp_name is None:
        exp_name = cfg.get("demonstrator", {}).get('dataset', 'fairness_experiment')
    logger = Logger(base_dir=cfg.get("log_dir", "./logs"), exp_name=exp_name, exp_tag = cfg.get("exp_tag", "")
)
    logger.register_interrupt_cleanup()
    try:
        print("📦 Initializing Demonstrator...")
        demo = Demonstrator(cfg)

        print("🧠 Building Learner...")
        learner = Learner(cfg, demonstrator=demo, logger=logger)

        print("🚀 Starting training...")
        learner.run()
        logger.save_config_as_txt(learner.model.cfg)
        logger.mark_completed()
        logger.close()
    except KeyboardInterrupt:
        print(f'Interrupted. Deleting Run dir...\n')
        raise

    #  import ipdb;ipdb.set_trace()

if __name__ == "__main__":
    main()
