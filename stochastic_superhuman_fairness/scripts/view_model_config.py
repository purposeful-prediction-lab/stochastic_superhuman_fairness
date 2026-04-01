import argparse
import os
import shutil
import torch

from stochastic_superhuman_fairness.core.utils_io import (
        format_dict_text,
        save_dict_text,
        pretty_dict_print,
        )
from stochastic_superhuman_fairness.core.models.model_io_utils import (
    load_model_from_archive,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, help="Path to run.zip")
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--no_safe_load", action="store_true")
    ap.add_argument("--summary", action="store_true")

    args = ap.parse_args()


    # -------------------------------------------------
    # Load model from archive
    # -------------------------------------------------

    model, cfg, demo = load_model_from_archive(
        archive_path=args.archive,
        phase_idx=args.phase,
        strict=args.strict,
        use_safe_load=not args.no_safe_load,
    )
    pretty_cfg = format_dict_text(cfg)
    
    pretty_dict_print(cfg)
    if args.summary:
        print("=== Load complete ===")
        print("archive:", args.archive)
        print("model  :", type(model).__name__)


if __name__ == "__main__":
    raise SystemExit(main())
