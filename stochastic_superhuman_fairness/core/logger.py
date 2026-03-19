import os
import json
import shutil
import signal
import atexit
import io
import torch
import numpy as np
import time
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
    
    def __init__(self, base_dir="./runs", exp_name="experiment", seed=None):
        self.completed = False
        self._cleanup_registered = False
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.exp_name = exp_name
        self.exp_dir = os.path.join(self.base_dir, self.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        run_id = self._next_run_id_locked()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        run_name = f"{exp_name}_{run_id:03d}_{timestamp}"
        if seed is not None:
            run_name += f"_seed{seed}"

        self.run_dir = os.path.join(self.exp_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=False)

        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.plot_dir = os.path.join(self.run_dir, "plots")
        self.artifact_dir = os.path.join(self.run_dir, "artifacts")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.artifact_dir, exist_ok=True)

        self.file_path = os.path.join(self.run_dir, "metrics_log.jsonl")
        self._file = open(self.file_path, "a", buffering=1)
        self._write_header()

        print(f"🧾 Run #{run_id} directory created at: {self.run_dir}")

    # ==================================================
    # Header (keep yours)
    # ==================================================
    def _write_header(self):
        pass

    def mark_completed(self):
        self.completed = True

    def cleanup_run_dir(self):
        """Delete log dir if run did not complete."""
        if self.completed:
            return
        if self.run_dir and os.path.exists(self.run_dir):
            try:
                shutil.rmtree(self.run_dir)
                print(f"[logger] Deleted interrupted run directory: {self.run_dir}")
            except Exception as e:
                print(f"[logger] Failed to delete log directory: {e}")

    def register_interrupt_cleanup(self):
        """Register cleanup for Ctrl+C and process exit."""
        if self._cleanup_registered:
            return
        self._cleanup_registered = True

        atexit.register(self.cleanup_run_dir)

        def _handle_interrupt(signum, frame):
            self.cleanup_run_dir()
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _handle_interrupt)
    # ==================================================
    # Counter + Lock
    # ==================================================
    def _counter_path(self) -> str:
        return os.path.join(self.base_dir, "run_counter.txt")

    def _lock_path(self) -> str:
        return os.path.join(self.base_dir, "run_counter.lock")

    def _acquire_lock(self, timeout_s: float = 10.0, poll_s: float = 0.05) -> int:
        """
        Acquire lock by atomically creating a lockfile.
        Returns an OS file descriptor for the lockfile (must be closed).
        """
        lock_path = self._lock_path()
        deadline = time.time() + timeout_s

        while True:
            try:
                # Atomic create: succeeds for only one process
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                # Optional: write PID for debugging
                os.write(fd, str(os.getpid()).encode("utf-8"))
                return fd
            except FileExistsError:
                if time.time() >= deadline:
                    raise TimeoutError(f"Timed out acquiring lock: {lock_path}")
                time.sleep(poll_s)

    def _release_lock(self, fd: int) -> None:
        lock_path = self._lock_path()
        try:
            os.close(fd)
        finally:
            # Best-effort remove (if already removed, ignore)
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass

    def _next_run_id_locked(self) -> int:
        """
        Concurrency-safe counter update.
        If the counter file is missing/corrupted, it restarts from 1.
        """
        fd = self._acquire_lock(timeout_s=30.0, poll_s=0.05)
        try:
            path = self._counter_path()
            last = 0

            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        txt = f.read().strip()
                    last = int(txt)
                    if last < 0:
                        last = 0
                except Exception:
                    last = 0  # corrupted/empty -> reset

            run_id = last + 1

            # Write back
            with open(path, "w") as f:
                f.write(str(run_id))

            return run_id
        finally:
            self._release_lock(fd)
    #  def __init__(self, log_dir="./logs", exp_name="run"):
    #      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #      self.log_dir = os.path.join(log_dir, exp_name)
    #      os.makedirs(self.log_dir, exist_ok=True)
    #      self.file_path = os.path.join(self.log_dir, f"training_{timestamp}.jsonl")
    #      self.ckpt_dir = os.path.join(self.log_dir, "checkpoints")
    #      os.makedirs(self.ckpt_dir, exist_ok=True)
    #
    #      self._file = open(self.file_path, "a", buffering=1)  # line-buffered
    #      self._write_header()
    #      print(f"🧾 Logging to {self.file_path}")

    # ----------------------------------------------------------
    def _write_header(self):
        meta = {
            "event": "init",
            "time": datetime.now().isoformat(),
            "msg": "Logger initialized.",
        }
        self._file.write(json.dumps(meta) + "\n")

    # ----------------------------------------------------------
    def log(self, record: dict, flatten: bool = False, keep_path: bool = True, verbose: bool = True, 
            no_print: list = []):

        no_print += ['time','algo','epoch','phase', 'gamma_matrix']

        record = {k: v for k, v in record.items() if (v is not None) and (k != 'gamma_matrix')}
        if flatten:
            record = flatten_record(record, keep_path=keep_path)
        record = {k: make_json_safe(v) for k, v in record.items()}
        record["time"] = datetime.now().isoformat()
        self._file.write(json.dumps(record) + "\n")

        """Log one training/eval record to console + file."""
        if verbose:
            if "algo" in record and "epoch" in record:
                tag = f"[{record['algo'].upper()} | Epoch {record['epoch']}]"
            elif record.get("event") == "phase_transition":
                tag = f"[→ PHASE {record['phase']}: {record['algo'].upper()}]"
            else:
                tag = "[LOG]"
            msg = f"\n{tag} { {k:v for k,v in record.items() if (k not in no_print) and 'dir' not in k} }"
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
        #  import ipdb;ipdb.set_trace()
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
