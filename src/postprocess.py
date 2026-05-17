"""Validate model JSON, clamp times, add frame_index, sort, and merge near-duplicates."""

from __future__ import annotations

import copy
from typing import Any

from src.schema import ALLOWED_SET, DEFAULT_MIN_GAP_SEC, normalize_class_key


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _as_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    return str(x)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def validate_and_clean_events(
    events: list[dict[str, Any]],
    duration_sec: float,
) -> list[dict[str, Any]]:
    """Drop unknown classes and clamp all time fields to [0, duration_sec]."""
    cleaned: list[dict[str, Any]] = []
    for raw in events:
        d = normalize_class_key(copy.deepcopy(raw))
        cls = _as_str(d.get("class"), "")
        if cls not in ALLOWED_SET:
            continue
        ts = _clamp(_as_float(d.get("timestamp_sec"), 0.0), 0.0, duration_sec)
        st = _clamp(_as_float(d.get("start_time_sec", ts), ts), 0.0, duration_sec)
        en = _clamp(_as_float(d.get("end_time_sec", ts), ts), 0.0, duration_sec)
        conf = _clamp(_as_float(d.get("confidence"), 0.0), 0.0, 1.0)
        expl = _as_str(d.get("explanation"), "")
        cleaned.append(
            {
                "class": cls,
                "timestamp_sec": ts,
                "start_time_sec": st,
                "end_time_sec": en,
                "confidence": conf,
                "explanation": expl,
            }
        )
    return cleaned


def add_frame_indices(events: list[dict[str, Any]], output_fps: float) -> list[dict[str, Any]]:
    """Add ``frame_index`` = round(timestamp_sec * output_fps) for each event."""
    out: list[dict[str, Any]] = []
    for e in events:
        row = dict(e)
        row["frame_index"] = int(round(_as_float(row["timestamp_sec"], 0.0) * output_fps))
        out.append(row)
    return out


def sort_by_timestamp(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(events, key=lambda x: _as_float(x.get("timestamp_sec"), 0.0))


def merge_near_duplicate_events(
    events: list[dict[str, Any]],
    min_gap_sec: dict[str, float],
) -> list[dict[str, Any]]:
    """
    Merge events of the same class whose timestamps fall within the class-specific gap.

    For each incoming event (in time order), any *kept* event of the same class whose
    ``timestamp_sec`` is within ``min_gap_sec`` forms a cluster with it. The whole
    cluster is replaced by the single highest-confidence event. This handles another
    class appearing between two nearby passes and avoids leaving triples unmerged.
    """
    if not events:
        return []
    ordered = sort_by_timestamp(events)
    merged: list[dict[str, Any]] = []
    for e in ordered:
        cls = _as_str(e.get("class"), "")
        gap = float(min_gap_sec.get(cls, DEFAULT_MIN_GAP_SEC.get(cls, 1.0)))
        ts = _as_float(e.get("timestamp_sec"), 0.0)
        cluster_idx = [
            j
            for j, k in enumerate(merged)
            if _as_str(k.get("class"), "") == cls
            and abs(_as_float(k.get("timestamp_sec"), 0.0) - ts) <= gap
        ]
        if not cluster_idx:
            merged.append(e)
            continue
        cluster_rows = [merged[j] for j in cluster_idx] + [e]
        winner = max(cluster_rows, key=lambda r: _as_float(r.get("confidence"), 0.0))
        for j in sorted(cluster_idx, reverse=True):
            del merged[j]
        merged.append(winner)
    return sort_by_timestamp(merged)


def postprocess_model_events(
    raw_events: list[dict[str, Any]],
    duration_sec: float,
    output_fps: float,
    min_gap_sec: dict[str, float],
) -> list[dict[str, Any]]:
    """Full pipeline: validate, sort, merge, re-sort, add frame indices."""
    step1 = validate_and_clean_events(raw_events, duration_sec)
    step2 = sort_by_timestamp(step1)
    step3 = merge_near_duplicate_events(step2, min_gap_sec)
    step4 = sort_by_timestamp(step3)
    return add_frame_indices(step4, output_fps)
