"""
Checkpoint manager for mmRec.
 
Responsibilities:
    - Save best checkpoint (by val AUC) and last checkpoint
    - Auto-name checkpoint dir as checkpoints/{dataset}_{model}/
    - Load checkpoint and restore model + optimizer state
    - Log save/load events via Logger
 
Usage (in trainer/trainer.py):
    from utils.checkpoint import CheckpointManager
 
    ckpt = CheckpointManager(cfg, logger, dataset="taac2026", model="onetrans")
 
    # after each epoch
    ckpt.save(
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        val_auc=val_auc,
    )
 
    # to resume training
    start_epoch, best_auc = ckpt.load_best(model, optimizer)
"""

import os
import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Optional, Tuple
 
from utils.config import DotDict
from utils.logger import Logger


class CheckpointManager:
    """
    Manages best and last checkpoints for a single training run.
    """
    def __init__(
        self,
        cfg: DotDict,
        logger: Logger,
        dataset: str,
        model: str,
    ):
        ckpt_cfg = cfg.get("checkpoint", {})
        root_dir = ckpt_cfg.get("dir", "checkpoints")
        self.save_best = ckpt_cfg.get("save_best", True)
        self.save_last = ckpt_cfg.get("save_last", True)

        self.run_dir = os.path.join(root_dir, f"{dataset}_{model}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.best_path = os.path.join(self.run_dir, "best.pt")
        self.last_path = os.path.join(self.run_dir, "last.pt")

        self.logger = logger
        self._best_auc = -1.0
    
    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        val_auc: float,
        extra: Optional[dict] = None
    ):
        """
        Save checkpoint if val_auc is better than best so far.
        Always save last checkpoint.
        """
        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_auc": val_auc
        }

        if extra:
            payload.update(extra)
        
        if self.save_last:
            self._write(payload, self.last_path)
            self.logger.info(f"Saved last checkpoint at epoch {epoch} with val AUC {val_auc:.6f}")

        if self.save_best and val_auc > self._best_auc:
            self._best_auc = val_auc
            self._write(payload, self.best_path)
            self.logger.info(f"New best checkpoint at epoch {epoch} with val AUC {val_auc:.6f}")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        device: str = "cpu"
    ) -> Tuple[int, float]:
        """
        Load best checkpoint and restore model + optimizer state.
        Returns (start_epoch, best_auc).
        """
        return self._load(self.best_path, model, optimizer, device)
    
    def load_last(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        device: str = "cpu"
    ) -> Tuple[int, float]:
        """
        Load last checkpoint and restore model + optimizer state.
        Returns (start_epoch, val_auc).
        """
        return self._load(self.last_path, model, optimizer, device)
    
    def load_from_path(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        device: str = "cpu"
    ) -> Tuple[int, float]:
        """
        Load checkpoint from a specific path and restore model + optimizer state.
        Returns (start_epoch, val_auc).
        """
        return self._load(path, model, optimizer, device)
    
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write(payload: dict, path: str):
        tmp_path = path + ".tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    
    def _load(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optional[Optimizer],
        device: str
    ) -> Tuple[int, float]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
            
        ckpt = torch.load(path, map_location=device)

        model.load_state_dict(ckpt["model_state"])
        if optimizer is not None and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        
        epoch = ckpt.get("epoch", 0)
        val_auc = ckpt.get("val_auc", 0.0)
        self._best_auc = val_auc

        self.logger.info(f"Loaded checkpoint from {path} at epoch {epoch} with val AUC {val_auc:.6f}")
        return epoch + 1, val_auc
    
    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
 
    @property
    def best_auc(self) -> float:
        """Current best val AUC seen so far in this run."""
        return self._best_auc
 
    def has_best(self) -> bool:
        """True if a best.pt checkpoint exists on disk."""
        return os.path.exists(self.best_path)
 
    def has_last(self) -> bool:
        """True if a last.pt checkpoint exists on disk."""
        return os.path.exists(self.last_path)