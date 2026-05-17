#!/usr/bin/env python3
"""
Evaluate model predictions against ground-truth annotation JSON files.

Example:
  python run_eval.py --pred_dir outputs_model --gt_dir dataset --tolerance_sec 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_metrics import (
    build_error_report,
    compute_metrics,
    find_gt_json_for_clip,
    format_metrics_table,
    load_timed_events,
)
from src.schema import ALLOWED_EVENT_CLASSES


def _aggregate_metrics(per_clip_metrics: list[dict]) -> dict:
    """Sum TP/FP/FN across clips and recompute metrics."""
    totals = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in ALLOWED_EVENT_CLASSES}
    for m in per_clip_metrics:
        for cls, row in m["per_class"].items():
            totals[cls]["tp"] += row["tp"]
            totals[cls]["fp"] += row["fp"]
            totals[cls]["fn"] += row["fn"]

    preds: list[TimedEvent] = []
    gts: list[TimedEvent] = []
    # Rebuild from totals via synthetic approach — compute directly
    per_class: dict = {}
    f1_values: list[float] = []
    micro_tp = micro_fp = micro_fn = 0

    for cls in ALLOWED_EVENT_CLASSES:
        tp = totals[cls]["tp"]
        fp = totals[cls]["fp"]
        fn = totals[cls]["fn"]
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
        if tp + fp + fn > 0:
            f1_values.append(f1)

    micro_prec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_rec = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (
        (2 * micro_prec * micro_rec / (micro_prec + micro_rec))
        if (micro_prec + micro_rec) > 0
        else 0.0
    )
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0

    return {
        "totals": {
            "tp": micro_tp,
            "fp": micro_fp,
            "fn": micro_fn,
            "precision": round(micro_prec, 4),
            "recall": round(micro_rec, 4),
            "f1": round(micro_f1, 4),
        },
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate event detection predictions")
    parser.add_argument("--pred_dir", type=Path, required=True, help="Directory of prediction JSON")
    parser.add_argument("--gt_dir", type=Path, required=True, help="Dataset root with clip folders")
    parser.add_argument(
        "--tolerance_sec",
        type=float,
        default=1.0,
        help="Max |pred_time - gt_time| for a match (default: 1.0)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "outputs_eval",
        help="Directory for metrics.json and errors.json",
    )
    args = parser.parse_args()

    pred_dir = args.pred_dir.resolve()
    gt_dir = args.gt_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_dir.is_dir():
        raise SystemExit(f"pred_dir is not a directory: {pred_dir}")
    if not gt_dir.is_dir():
        raise SystemExit(f"gt_dir is not a directory: {gt_dir}")

    pred_files = sorted(p for p in pred_dir.glob("*.json") if p.is_file())
    if not pred_files:
        raise SystemExit(f"No prediction JSON files in {pred_dir}")

    per_clip_metrics: list[dict] = []
    error_reports: list[dict] = []
    skipped: list[str] = []

    for pred_path in pred_files:
        clip_id = pred_path.stem
        gt_path = find_gt_json_for_clip(gt_dir, clip_id)
        if gt_path is None:
            skipped.append(clip_id)
            continue

        preds = load_timed_events(pred_path)
        gts = load_timed_events(gt_path)
        clip_metrics = compute_metrics(preds, gts, args.tolerance_sec)
        clip_metrics["clip_id"] = clip_id
        per_clip_metrics.append(clip_metrics)
        error_reports.append(
            build_error_report(preds, gts, args.tolerance_sec, clip_id)
        )

    if not per_clip_metrics:
        raise SystemExit(
            f"No clips evaluated. Check that pred stems match gt folders under {gt_dir}. "
            f"Skipped (no GT): {skipped}"
        )

    aggregate = _aggregate_metrics(per_clip_metrics)
    metrics_payload = {
        "tolerance_sec": args.tolerance_sec,
        "num_clips": len(per_clip_metrics),
        "skipped_clips": skipped,
        "aggregate": aggregate,
        "per_clip": per_clip_metrics,
    }

    metrics_path = out_dir / "metrics.json"
    errors_path = out_dir / "errors.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    with errors_path.open("w", encoding="utf-8") as f:
        json.dump({"clips": error_reports}, f, indent=2, ensure_ascii=False)
        f.write("\n")

    aggregate_full = {
        "tolerance_sec": args.tolerance_sec,
        **aggregate,
        "per_class": aggregate["per_class"],
    }
    print(format_metrics_table(aggregate_full))
    print(f"\nEvaluated {len(per_clip_metrics)} clips.")
    if skipped:
        print(f"Skipped {len(skipped)} clips (no ground truth): {', '.join(skipped[:10])}"
              + (" ..." if len(skipped) > 10 else ""))
    print(f"Saved {metrics_path}")
    print(f"Saved {errors_path}")


if __name__ == "__main__":
    main()
