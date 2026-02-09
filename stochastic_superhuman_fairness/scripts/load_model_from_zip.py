import argparse
import os
import shutil
import torch

from stochastic_superhuman_fairness.core.models.model_io_utils import (
    load_model_from_archive,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, help="Path to run.zip")
    ap.add_argument("--target_dir", required=True, help="Where to write loaded model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--config_path", default=None,
                    help="Optional override config file (json/yaml/toml)")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--no_safe_load", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    # -------------------------------------------------
    # Load model from archive
    # -------------------------------------------------

    model, cfg, demo = load_model_from_archive(
        archive_path=args.archive,
        device=args.device,
        phase_idx=args.phase,
        strict=args.strict,
        use_safe_load=not args.no_safe_load,
    )

    # -------------------------------------------------
    # Optional config override
    # -------------------------------------------------

    if args.config_path is not None:
        if not os.path.exists(args.config_path):
            raise FileNotFoundError(args.config_path)

        dst = os.path.join(args.target_dir, os.path.basename(args.config_path))
        shutil.copy(args.config_path, dst)

    # -------------------------------------------------
    # Save reloaded artifacts
    # -------------------------------------------------

    model_path = os.path.join(args.target_dir, "model_state.pt")
    torch.save(model.state_dict(), model_path)

    # Optional: save dist state if supported
    if hasattr(model, "get_dist_state"):
        dist_state = model.get_dist_state()
        if dist_state:
            torch.save(dist_state, os.path.join(args.target_dir, "dist_state.pt"))

    if args.summary:
        print("=== Load complete ===")
        print("archive:", args.archive)
        print("target :", args.target_dir)
        print("device :", args.device)
        print("model  :", type(model).__name__)
        print("saved  :", model_path)
    import ipdb;ipdb.set_trace()
    return model


if __name__ == "__main__":
    raise SystemExit(main())
