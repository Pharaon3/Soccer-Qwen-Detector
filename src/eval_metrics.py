"""Event-level evaluation: greedy time matching and precision/recall/F1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.schema import ALLOWED_SET, ALLOWED_EVENT_CLASSES, normalize_class_key


@dataclass(frozen=True)
class TimedEvent:
    """Normalized event for matching."""

    time_sec: float
    label: str
    raw: dict[str, Any]


def _event_time(row: dict[str, Any]) -> float:
    for key in ("timestamp_sec", "time_sec"):
        if key in row:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def _event_label(row: dict[str, Any]) -> str:
    row = normalize_class_key(row)
    lab = row.get("label") or row.get("class") or ""
    return str(lab).strip()


def load_timed_events(path: Path) -> list[TimedEvent]:
    """Load events from a prediction or ground-truth JSON file."""
    with path.open("r", encoding="utf-8") as f:
        root = json.load(f)
    if not isinstance(root, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    raw_events = root.get("events", [])
    if raw_events is None:
        raw_events = []
    if not isinstance(raw_events, list):
        raise ValueError(f'"events" must be an array in {path}')

    out: list[TimedEvent] = []
    for row in raw_events:
        if not isinstance(row, dict):
            continue
        lab = _event_label(row)
        if lab not in ALLOWED_SET:
            continue
        out.append(
            TimedEvent(
                time_sec=_event_time(row),
                label=lab,
                raw=row,
            )
        )
    return out


def match_events_greedy(
    predictions: list[TimedEvent],
    ground_truth: list[TimedEvent],
    tolerance_sec: float,
) -> tuple[list[tuple[TimedEvent, TimedEvent]], list[TimedEvent], list[TimedEvent]]:
    """
    Per-class greedy nearest-time matching.

    Each prediction matches at most one ground-truth event within ``tolerance_sec``.
    """
    tp: list[tuple[TimedEvent, TimedEvent]] = []
    fp: list[TimedEvent] = []
    fn: list[TimedEvent] = []

    by_class = set(p.label for p in predictions) | set(g.label for g in ground_truth)

    for cls in by_class:
        preds = sorted([p for p in predictions if p.label == cls], key=lambda x: x.time_sec)
        gts = sorted([g for g in ground_truth if g.label == cls], key=lambda x: x.time_sec)
        used_gt: set[int] = set()

        for pred in preds:
            best_j: int | None = None
            best_dt = tolerance_sec + 1.0
            for j, gt in enumerate(gts):
                if j in used_gt:
                    continue
                dt = abs(pred.time_sec - gt.time_sec)
                if dt <= tolerance_sec and dt < best_dt:
                    best_dt = dt
                    best_j = j
            if best_j is not None:
                used_gt.add(best_j)
                tp.append((pred, gts[best_j]))
            else:
                fp.append(pred)

        for j, gt in enumerate(gts):
            if j not in used_gt:
                fn.append(gt)

    return tp, fp, fn


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def compute_metrics(
    predictions: list[TimedEvent],
    ground_truth: list[TimedEvent],
    tolerance_sec: float,
) -> dict[str, Any]:
    """Compute per-class and aggregate precision/recall/F1."""
    tp_all, fp_all, fn_all = match_events_greedy(predictions, ground_truth, tolerance_sec)

    per_class: dict[str, Any] = {}
    micro_tp = len(tp_all)
    micro_fp = len(fp_all)
    micro_fn = len(fn_all)

    f1_values: list[float] = []

    for cls in ALLOWED_EVENT_CLASSES:
        tp_c = sum(1 for p, _ in tp_all if p.label == cls)
        fp_c = sum(1 for p in fp_all if p.label == cls)
        fn_c = sum(1 for g in fn_all if g.label == cls)
        prec = _safe_div(tp_c, tp_c + fp_c)
        rec = _safe_div(tp_c, tp_c + fn_c)
        f1 = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls] = {
            "tp": tp_c,
            "fp": fp_c,
            "fn": fn_c,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
        if tp_c + fp_c + fn_c > 0:
            f1_values.append(f1)

    macro_f1 = _safe_div(sum(f1_values), len(f1_values)) if f1_values else 0.0
    micro_prec = _safe_div(micro_tp, micro_tp + micro_fp)
    micro_rec = _safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = _safe_div(2 * micro_prec * micro_rec, micro_prec + micro_rec)

    return {
        "tolerance_sec": tolerance_sec,
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


def build_error_report(
    predictions: list[TimedEvent],
    ground_truth: list[TimedEvent],
    tolerance_sec: float,
    clip_id: str,
) -> dict[str, Any]:
    """Missed GT events and unmatched predictions for one clip."""
    tp, fp, fn = match_events_greedy(predictions, ground_truth, tolerance_sec)
    return {
        "clip_id": clip_id,
        "false_positives": [
            {
                "class": p.label,
                "timestamp_sec": p.time_sec,
                **{k: v for k, v in p.raw.items() if k not in ("class", "label", "timestamp_sec", "time_sec")},
            }
            for p in fp
        ],
        "false_negatives": [
            {
                "label": g.label,
                "time_sec": g.time_sec,
                **{k: v for k, v in g.raw.items() if k not in ("class", "label", "timestamp_sec", "time_sec")},
            }
            for g in fn
        ],
        "true_positives": len(tp),
    }


def format_metrics_table(metrics: dict[str, Any]) -> str:
    """Human-readable table for console output."""
    lines = [
        f"Tolerance: {metrics['tolerance_sec']}s",
        f"Micro F1: {metrics['micro_f1']:.4f}  |  Macro F1: {metrics['macro_f1']:.4f}",
        f"Totals — P: {metrics['totals']['precision']:.4f}  "
        f"R: {metrics['totals']['recall']:.4f}  F1: {metrics['totals']['f1']:.4f}",
        "",
        f"{'Class':<18} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>7} {'Rec':>7} {'F1':>7}",
        "-" * 58,
    ]
    for cls in ALLOWED_EVENT_CLASSES:
        row = metrics["per_class"][cls]
        if row["tp"] + row["fp"] + row["fn"] == 0:
            continue
        lines.append(
            f"{cls:<18} {row['tp']:>4} {row['fp']:>4} {row['fn']:>4} "
            f"{row['precision']:>7.4f} {row['recall']:>7.4f} {row['f1']:>7.4f}"
        )
    return "\n".join(lines)


def find_gt_json_for_clip(gt_dir: Path, clip_id: str) -> Path | None:
    """Resolve ground-truth JSON for a clip id under ``gt_dir/<clip_id>/``."""
    folder = gt_dir / clip_id
    if not folder.is_dir():
        return None
    preferred = folder / f"{clip_id}.json"
    if preferred.is_file():
        return preferred
    jsons = sorted(folder.glob("*.json"))
    return jsons[0] if jsons else None
