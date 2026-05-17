"""Training helpers: seed, class imbalance weights, checkpointing."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.schema import NUM_EVENT_CLASSES


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def estimate_pos_weight(loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Per-class positive weight for BCE: neg_count / pos_count (capped)."""
    pos = torch.zeros(NUM_EVENT_CLASSES, device=device)
    total_segments = 0
    for batch in loader:
        y = batch[1].to(device)
        total_segments += y.shape[0] * y.shape[1]
        pos += y.sum(dim=(0, 1))
    neg = float(total_segments) - pos
    w = neg / torch.clamp(pos, min=1.0)
    w = torch.clamp(w, max=500.0)
    return w


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer | None) -> int:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", -1)) + 1
