#!/usr/bin/env python3
"""
Train a temporal segment classifier on `dataset/<clip_id>/` JSON + video pairs.

Layout matches `dataset/annotator` expectations: one JSON and one video per clip folder.
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

from src.dataset_soccer import (
    SoccerClipSegmentDataset,
    scan_clip_folders,
    train_val_split,
)
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
    ap = argparse.ArgumentParser(description="Train segment event model on dataset/")
    ap.add_argument("--config", type=Path, default=ROOT / "train_config.yaml")
    ap.add_argument(
        "--mock_video",
        action="store_true",
        help="Use black frames when video is missing (debug only; overrides config).",
    )
    ap.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    args = ap.parse_args()
    cfg = _load_cfg(args.config.resolve())
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

    print(f"Scanned {len(all_records)} clip folders under {dataset_root}")
    with_vid = sum(1 for r in all_records if r.video_path and r.video_path.is_file())
    print(f"  Folders with a video file: {with_vid}")
    print(
        f"  Clips in this run: {len(usable)} "
        f"(mock_video_if_missing={mock_missing}"
        + (f", max_clips={max_clips}" if max_clips else "")
        + ")"
    )

    if not usable:
        print(
            "\nNo clips to train on. Add a video next to each JSON "
            "(see dataset/annotator app.js for supported extensions), "
            "or set train_config.yaml `mock_video_if_missing: true` only for debugging."
        )
        raise SystemExit(1)

    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_ratio = float(cfg.get("val_ratio", 0.15))
    train_records, val_records = train_val_split(usable, val_ratio, int(cfg.get("seed", 42)))

    duration_sec = float(cfg.get("duration_sec", 30.0))
    video_fps = float(cfg.get("video_fps", 25.0))
    num_segments = int(cfg.get("num_segments", 30))
    image_size = int(cfg.get("image_size", 224))
    batch_size = int(cfg.get("batch_size", 4))
    num_workers = int(cfg.get("num_workers", 0))

    train_ds = SoccerClipSegmentDataset(
        train_records,
        duration_sec=duration_sec,
        video_fps=video_fps,
        num_segments=num_segments,
        image_size=image_size,
        train=True,
        mock_video_if_missing=mock_missing,
    )
    val_ds = SoccerClipSegmentDataset(
        val_records if val_records else train_records[:1],
        duration_sec=duration_sec,
        video_fps=video_fps,
        num_segments=num_segments,
        image_size=image_size,
        train=False,
        mock_video_if_missing=mock_missing,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=len(train_ds) > batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )

    model = SegmentEventModel(
        num_classes=NUM_EVENT_CLASSES,
        pretrained=bool(cfg.get("pretrained_backbone", True)),
    ).to(device)

    pos_weight = estimate_pos_weight(train_loader, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    lr = float(cfg.get("lr", 1e-4))
    wd = float(cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    epochs = int(cfg.get("epochs", 25))
    ckpt_dir = (ROOT / str(cfg.get("checkpoint_dir", "checkpoints"))).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_every = int(cfg.get("save_every_epochs", 5))

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * xb.shape[0]
            n += xb.shape[0]
        train_loss = running / max(n, 1)

        model.eval()
        vsum = 0.0
        vn = 0
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                vloss = criterion(logits, yb)
                vsum += float(vloss) * xb.shape[0]
                vn += xb.shape[0]
        val_loss = vsum / max(vn, 1)

        print(f"epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if (epoch + 1) % save_every == 0 or epoch + 1 == epochs:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch+1:03d}.pt",
                model,
                optimizer,
                epoch,
                extra={"config": cfg, "train_clips": [r.clip_id for r in train_records]},
            )

    save_checkpoint(
        ckpt_dir / "last.pt",
        model,
        optimizer,
        epochs - 1,
        extra={"config": cfg},
    )
    print(f"Saved {ckpt_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
