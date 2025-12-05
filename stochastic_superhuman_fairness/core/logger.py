import os
import json
import torch
from datetime import datetime


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
    def log(self, record: dict):
        """Log one training/eval record to console + file."""
        record = {k: v for k, v in record.items() if v is not None}
        record["time"] = datetime.now().isoformat()
        self._file.write(json.dumps(record) + "\n")

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
    def save_checkpoint(self, model, phase_idx: int, algo: str, cfg=None):
        """Save model state dict and optionally its config."""
        ckpt_path = os.path.join(self.ckpt_dir, f"phase_{phase_idx}_{algo}.pt")
        torch.save(
            {
                "model_state": model.state_dict(),
                "cfg": cfg,
                "timestamp": datetime.now().isoformat(),
            },
            ckpt_path,
        )
        self.log({
            "event": "checkpoint",
            "phase": phase_idx,
            "algo": algo,
            "path": ckpt_path,
            "msg": f"Saved checkpoint for phase {phase_idx} ({algo})",
        })
        return ckpt_path

    # ----------------------------------------------------------
    def close(self):
        self._file.close()
        print("✅ Logger closed.")
