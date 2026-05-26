"""
Logger for mmRec training.
 
Responsibilities:
    - Console logging with timestamp (via Python logging)
    - TensorBoard logging (optional, controlled by cfg.logging.use_tensorboard)
    - Unified interface: logger.log_scalar(), logger.log_epoch()
    - Auto-creates log dir; gracefully degrades if tensorboard not installed
 
Usage (in trainer/trainer.py):
    from utils.logger import Logger
    logger = Logger(cfg, run_name="taac2026_onetrans")
 
    # inside train loop
    logger.log_scalar("train/loss", loss, step=global_step)
    logger.log_scalar("train/lr",   lr,   step=global_step)
 
    # after each epoch
    logger.log_epoch(epoch, train_loss=0.312, val_auc=0.734)
 
    logger.close()
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional

from utils.config import DotDict

# ═══════════════════════════════════════════════════════════════════════════════
# Console handler setup
# ═══════════════════════════════════════════════════════════════════════════════

def _build_console_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Build a console logger.
    Format: 2024-06-01 12:00:00 | INFO | message
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-5s | %(message)s", 
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.propagate = False
    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# Logger
# ═══════════════════════════════════════════════════════════════════════════════

class Logger:
    """
    Unified logger: console + optional TensorBoard.
    """
    def __init__(self, cfg: DotDict, run_name: Optional[str] = None):
        self._console = _build_console_logger("mmRec")

        log_config = cfg.get("logging", {})
        use_tb = log_config.get("use_tensorboard", False)
        tb_dir = log_config.get("tensorboard_dir", "runs")

        self._writer = None
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                if run_name is None:
                    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
                log_path = os.path.join(tb_dir, run_name)
                os.makedirs(log_path, exist_ok=True)
                self._writer = SummaryWriter(log_dir=log_path)
                self.info(f"TensorBoard logging enabled at: {log_path}")
            except ImportError:
                self.warning("TensorBoard logging requested but torch.utils.tensorboard is not available. Falling back to console only.")

    # ------------------------------------------------------------------
    # Console shortcuts
    # ------------------------------------------------------------------
    
    def info(self, msg: str):
        self._console.info(msg)

    def warning(self, msg: str):
        self._console.warning(msg)

    def error(self, msg: str):
        self._console.error(msg)

    # ------------------------------------------------------------------
    # Scalar logging
    # ------------------------------------------------------------------

    def log_scalar(self, tag: str, value: float, step: int):
        if self._writer is not None:
            self._writer.add_scalar(tag, value, global_step=step)

    def log_scalars(self, tag_value: dict, step: int):
        for tag, value in tag_value.items():
            self.log_scalar(tag, value, step)
    
    # ------------------------------------------------------------------
    # Epoch summary (console + tensorboard)
    # ------------------------------------------------------------------

    def log_epoch(self, epoch: int, **metrics):
        parts = [f"Epoch {epoch:>3d}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")

            if self._writer is not None:
                tb_tag = f"Epoch/{k}"
                self._writer.add_scalar(tb_tag, v, global_step=epoch)

        self.info(" | ".join(parts))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.close()