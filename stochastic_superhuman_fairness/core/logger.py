import os
import json
import io
import torch
import numpy as np
from datetime import datetime
from collections.abc import Mapping
from types import SimpleNamespace
import zipfile
from stochastic_superhuman_fairness.core.utils import ns_to_dict

def _is_dictlike(x):
    return isinstance(x, Mapping) or isinstance(x, SimpleNamespace)

def flatten_record(d, parent_key="", sep=".", keep_path=True):
    items = {}
    # support SimpleNamespace / NamespaceDict
    it = (vars(d).items() if isinstance(d, SimpleNamespace) else d.items())
    for k, v in it:
        k = str(k)
        key = f"{parent_key}{sep}{k}" if (keep_path and parent_key) else k
        if _is_dictlike(v):
            items.update(flatten_record(v, key if keep_path else "", sep=sep, keep_path=keep_path))
        else:
            items[key] = v
    return items

def make_json_safe(x):
    # keep it simple: handle common numeric types / tensors
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        return x.item() if x.numel() == 1 else x.tolist()
    if isinstance(x, (np.floating, np.integer)):  # if you use numpy
        return x.item()
    return x

class Logger:
    """
    Simple experiment logger with checkpoint support.
    Logs per-epoch metrics and saves model states at each phase transition.
    """

    def __init__(self, log_dir="./logs", exp_name="run"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, exp_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.file_path = os.path.join(self.log_dir, f"training_{timestamp}.jsonl")
        self.ckpt_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self._file = open(self.file_path, "a", buffering=1)  # line-buffered
        self._write_header()
        print(f"🧾 Logging to {self.file_path}")

    # ----------------------------------------------------------
    def _write_header(self):
        meta = {
            "event": "init",
            "time": datetime.now().isoformat(),
            "msg": "Logger initialized.",
        }
        self._file.write(json.dumps(meta) + "\n")

    # ----------------------------------------------------------
    def log(self, record: dict, flatten: bool = True, keep_path: bool = True):
        record = {k: v for k, v in record.items() if v is not None}
        if flatten:
            record = flatten_record(record, keep_path=keep_path)
        record = {k: make_json_safe(v) for k, v in record.items()}
        record["time"] = datetime.now().isoformat()
        self._file.write(json.dumps(record) + "\n")
   #  def log(self, record: dict):
        """Log one training/eval record to console + file."""
        #  record = {k: v for k, v in record.items() if v is not None}
        #  record["time"] = datetime.now().isoformat()
        #  self._file.write(json.dumps(record) + "\n")

        # Pretty console summary
        if "algo" in record and "epoch" in record:
            tag = f"[{record['algo'].upper()} | Epoch {record['epoch']}]"
        elif record.get("event") == "phase_transition":
            tag = f"[→ PHASE {record['phase']}: {record['algo'].upper()}]"
        else:
            tag = "[LOG]"
        msg = f"{tag} { {k:v for k,v in record.items() if k not in ['time','algo','epoch','phase']} }"
        print(msg)

    # ----------------------------------------------------------
    def log_transition(self, phase_idx: int, algo: str):
        """Log model transition event."""
        self.log({
            "event": "phase_transition",
            "phase": phase_idx,
            "algo": algo,
            "msg": f"Switching to model: {algo}",
        })

    # ----------------------------------------------------------

    def save_checkpoint(self, model, phase_idx: int, algo: str, cfg):
        """
        Save a structured checkpoint:
          - model_state.pt   (tensors only)
          - config.json      (readable)
          - metadata.json
        all bundled into a single zip.
        """
        
        run_name = f"phase_{phase_idx}_{algo}"
        zip_path = os.path.join(self.ckpt_dir, f"{run_name}.zip")

        os.makedirs(self.ckpt_dir, exist_ok=True)

        # --- prepare artifacts ---
        model_state = model.state_dict()

        # cfg should already be NamespaceDict → convert to plain dict
        cfg_dict = ns_to_dict(cfg)

        metadata = {
            "phase": phase_idx,
            "algo": algo,
            "timestamp": datetime.now().isoformat(),
            "format_version": 1,
        }
        import ipdb;ipdb.set_trace()
        # --- write zip ---
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # model weights (tensors only → safe)
            buf = io.BytesIO()
            torch.save(model_state, buf)
            zf.writestr("model_state.pt", buf.getvalue())

            # config
            zf.writestr("config.json", json.dumps(cfg_dict, indent=2))

            # metadata
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))

            # optional: stochastic dist state
            if hasattr(model, "get_dist_state"):
                dist_state = model.get_dist_state()
                if dist_state:
                    buf = io.BytesIO()
                    torch.save(dist_state, buf)
                    zf.writestr("dist_state.pt", buf.getvalue())

        self.log({
            "event": "checkpoint",
            "phase": phase_idx,
            "algo": algo,
            "path": zip_path,
            "msg": f"Saved checkpoint for phase {phase_idx} ({algo})",
        })

        return zip_path
        # ----------------------------------------------------------
    def close(self):
        self._file.close()
        print("✅ Logger closed.")
