#!/usr/bin/env python3
"""
Legacy: train ResNet18 + temporal Conv1D (not Qwen).

Use ``run_train.py`` to fine-tune Qwen2.5-VL instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_soccer import SoccerClipSegmentDataset, scan_clip_folders, train_val_split
from src.model_temporal import SegmentEventModel
from src.schema import NUM_EVENT_CLASSES
from src.train_utils import estimate_pos_weight, save_checkpoint, set_seed


def _collate(batch: list) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    ids = [b[2] for b in batch]
    return xs, ys, ids


def _load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="[Legacy] Train ResNet segment CNN")
    ap.add_argument("--config", type=Path, default=ROOT / "train_config_cnn.yaml")
    ap.add_argument("--mock_video", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg_path = args.config.resolve()
    if not cfg_path.is_file():
        raise SystemExit(
            f"Config not found: {cfg_path}\n"
            "Copy train_config.yaml CNN fields into train_config_cnn.yaml or create one."
        )
    cfg = _load_cfg(cfg_path)
    if args.mock_video:
        cfg["mock_video_if_missing"] = True
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)

    dataset_root = (ROOT / str(cfg.get("dataset_root", "dataset"))).resolve()
    skip = frozenset({str(cfg.get("annotator_dir_name", "annotator"))})
    all_records = scan_clip_folders(dataset_root, skip_dir_names=skip)
    mock_missing = bool(cfg.get("mock_video_if_missing", False))
    usable = [
        r
        for r in all_records
        if (r.video_path is not None and r.video_path.is_file()) or mock_missing
    ]
    max_clips = int(cfg.get("max_clips") or 0)
    if max_clips > 0:
        usable = usable[:max_clips]
    if not usable:
        raise SystemExit(1)

    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_records, val_records = train_val_split(
        usable, float(cfg.get("val_ratio", 0.15)), int(cfg.get("seed", 42))
    )

    train_ds = SoccerClipSegmentDataset(
        train_records,
        duration_sec=float(cfg.get("duration_sec", 30)),
        video_fps=float(cfg.get("video_fps", 25)),
        num_segments=int(cfg.get("num_segments", 30)),
        image_size=int(cfg.get("image_size", 224)),
        train=True,
        mock_video_if_missing=mock_missing,
    )
    val_ds = SoccerClipSegmentDataset(
        val_records if val_records else train_records[:1],
        duration_sec=float(cfg.get("duration_sec", 30)),
        video_fps=float(cfg.get("video_fps", 25)),
        num_segments=int(cfg.get("num_segments", 30)),
        image_size=int(cfg.get("image_size", 224)),
        train=False,
        mock_video_if_missing=mock_missing,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=_collate,
    )

    model = SegmentEventModel(
        num_classes=NUM_EVENT_CLASSES,
        pretrained=bool(cfg.get("pretrained_backbone", True)),
    ).to(device)
    pos_weight = estimate_pos_weight(train_loader, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )

    ckpt_dir = (ROOT / str(cfg.get("checkpoint_dir", "checkpoints/cnn"))).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(cfg.get("epochs", 25))

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * xb.shape[0]
            n += xb.shape[0]
        print(f"epoch {epoch+1}/{epochs}  train_loss={running/max(n,1):.4f}")

    save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epochs - 1, extra={"config": cfg})
    print(f"Saved {ckpt_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
