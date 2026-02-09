import argparse
from omegaconf import OmegaConf

from stochastic_superhuman_fairness.core.demonstrator import Demonstrator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True, help="Path to experiment YAML config")
    ap.add_argument("--sample", type=int, default=None, help="Optional demo_sample override")
    ap.add_argument("--overwrite", action="store_true", help="Force regeneration even if cache matches")
    ap.add_argument("--no_torch", action="store_true", help="Do not materialize demos as torch tensors")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)

    # Ensure demonstrator exists
    if not hasattr(cfg, "demonstrator"):
        raise ValueError("Config missing cfg.demonstrator")

    # Ensure demotype is lrdecisions (alias fullset handled in Demonstrator if you kept it)
    if not hasattr(cfg.demonstrator, "demotype"):
        cfg.demonstrator.demotype = "lrdecisions"

    if args.sample is not None:
        cfg.demonstrator.demo_sample = args.sample
    if args.overwrite:
        cfg.demonstrator.overwrite = True

    demo = Demonstrator(cfg, auto_create=False, to_torch=not args.no_torch)
    out = demo.create_demos(to_torch=not args.no_torch)
    import ipdb;ipdb.set_trace()
    meta = out.get("metadata", {})
    try:
        meta = meta.tolist() # if npz turn to dict
    except:
        pass
    print("✅ Done.")
    print(f"dataset={meta.get('dataset')}, demotype={meta.get('demotype')}, "
          f"n_train_demos={meta.get('n_demos_train')}, n_eval_demos={meta.get('n_demos_eval')}")
    print(f"cache_dir={cfg.demonstrator.cache_dir}")


if __name__ == "__main__":
    main()
