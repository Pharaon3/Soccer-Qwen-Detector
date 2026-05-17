#!/usr/bin/env python3
"""
Tune per-class sigmoid thresholds on a validation split.

Example:
  python tune_thresholds.py --checkpoint checkpoints/last.pt \\
      --config train_config.yaml --val_split 0.15 --output thresholds.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_soccer import (
    load_clip_frames_for_model,
    load_clip_events,
    scan_clip_folders,
    train_val_split,
)
from src.eval_metrics import TimedEvent, compute_metrics
from src.model_inference import (
    load_trained_model,
    probs_to_raw_events,
    predict_segment_probs,
)
from src.schema import ALLOWED_EVENT_CLASSES


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _gt_to_timed_events(json_path: Path) -> list[TimedEvent]:
    pairs, _ = load_clip_events(json_path)
    return [TimedEvent(time_sec=ts, label=lab, raw={"time_sec": ts, "label": lab}) for ts, lab in pairs]


def _events_to_timed(raw_events: list[dict]) -> list[TimedEvent]:
    out: list[TimedEvent] = []
    for row in raw_events:
        lab = str(row.get("class", ""))
        ts = float(row.get("timestamp_sec", 0.0))
        out.append(TimedEvent(time_sec=ts, label=lab, raw=row))
    return out


def _eval_thresholds_on_val(
    clip_probs: list[tuple[str, np.ndarray, list[TimedEvent]]],
    thresholds: dict[str, float],
    num_segments: int,
    duration_sec: float,
    tolerance_sec: float,
) -> dict:
    all_preds: list[TimedEvent] = []
    all_gts: list[TimedEvent] = []
    for _clip_id, probs, gts in clip_probs:
        raw = probs_to_raw_events(probs, thresholds, num_segments, duration_sec)
        preds = _events_to_timed(raw)
        all_preds.extend(preds)
        all_gts.extend(gts)
    return compute_metrics(all_preds, all_gts, tolerance_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune per-class detection thresholds")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "train_config.yaml")
    parser.add_argument("--val_split", type=float, default=None, help="Override val_ratio from config")
    parser.add_argument("--output", type=Path, default=ROOT / "thresholds.yaml")
    parser.add_argument("--tolerance_sec", type=float, default=1.0)
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.05)
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    cfg = _load_config(cfg_path)
    dataset_root = (ROOT / str(cfg.get("dataset_root", "dataset"))).resolve()
    skip = frozenset({str(cfg.get("annotator_dir_name", "annotator"))})
    all_records = scan_clip_folders(dataset_root, skip_dir_names=skip)

    mock_missing = bool(cfg.get("mock_video_if_missing", False))
    usable = [
        r
        for r in all_records
        if (r.video_path is not None and r.video_path.is_file()) or mock_missing
    ]
    if not usable:
        raise SystemExit(
            f"No clips with videos under {dataset_root}. "
            "Add videos next to annotation JSON files before tuning."
        )

    val_ratio = float(args.val_split if args.val_split is not None else cfg.get("val_ratio", 0.15))
    _, val_records = train_val_split(usable, val_ratio, int(cfg.get("seed", 42)))
    if not val_records:
        val_records = usable[: max(1, len(usable) // 5)]

    duration_sec = float(cfg.get("duration_sec", 30.0))
    video_fps = float(cfg.get("video_fps", 25.0))
    num_segments = int(cfg.get("num_segments", 30))
    image_size = int(cfg.get("image_size", 224))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_trained_model(args.checkpoint.resolve(), device)

    clip_probs: list[tuple[str, np.ndarray, list[TimedEvent]]] = []
    for rec in val_records:
        if rec.video_path is None or not rec.video_path.is_file():
            if not mock_missing:
                print(f"Skipping {rec.clip_id}: no video")
                continue
            continue
        frames = load_clip_frames_for_model(
            rec.video_path,
            num_segments,
            image_size,
            duration_sec,
            video_fps,
        )
        probs = predict_segment_probs(model, frames, device)
        gts = _gt_to_timed_events(rec.json_path)
        clip_probs.append((rec.clip_id, probs, gts))

    if not clip_probs:
        raise SystemExit("No validation clips with videos available for tuning.")

    print(f"Tuning on {len(clip_probs)} validation clips")

    grid = np.arange(args.threshold_min, args.threshold_max + 1e-9, args.threshold_step)
    best_thresholds: dict[str, float] = {}
    per_class_report: dict[str, dict] = {}

    base_thresholds = {c: 0.5 for c in ALLOWED_EVENT_CLASSES}

    for cls in ALLOWED_EVENT_CLASSES:
        best_f1 = -1.0
        best_thr = 0.5
        best_metrics_row = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        for thr in grid:
            trial = dict(base_thresholds)
            trial[cls] = float(thr)
            metrics = _eval_thresholds_on_val(
                clip_probs, trial, num_segments, duration_sec, args.tolerance_sec
            )
            row = metrics["per_class"][cls]
            if row["f1"] > best_f1:
                best_f1 = row["f1"]
                best_thr = float(thr)
                best_metrics_row = {
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                }

        best_thresholds[cls] = round(best_thr, 4)
        per_class_report[cls] = {
            "threshold": best_thresholds[cls],
            **best_metrics_row,
        }
        print(
            f"{cls:<18} thr={best_thresholds[cls]:.2f}  "
            f"P={best_metrics_row['precision']:.4f}  "
            f"R={best_metrics_row['recall']:.4f}  "
            f"F1={best_metrics_row['f1']:.4f}"
        )

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "thresholds": best_thresholds,
        "tolerance_sec": args.tolerance_sec,
        "val_clips": len(clip_probs),
        "per_class": per_class_report,
    }
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)

    final_metrics = _eval_thresholds_on_val(
        clip_probs, best_thresholds, num_segments, duration_sec, args.tolerance_sec
    )
    print(f"\nCombined macro F1: {final_metrics['macro_f1']:.4f}")
    print(f"Combined micro F1: {final_metrics['micro_f1']:.4f}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
