"""Convert annotator JSON labels into Qwen2.5-VL training target JSON."""

from __future__ import annotations

import json
from typing import Any

from src.schema import ALLOWED_SET


def build_training_target_json(
    video_id: str,
    events: list[tuple[float, str]],
    duration_sec: float,
) -> str:
    """
    Build the assistant response string for supervised fine-tuning.

    Matches the JSON schema requested at inference time.
    """
    out_events: list[dict[str, Any]] = []
    for ts, lab in sorted(events, key=lambda x: x[0]):
        if lab not in ALLOWED_SET:
            continue
        ts = float(max(0.0, min(ts, duration_sec)))
        window = 0.2
        out_events.append(
            {
                "class": lab,
                "timestamp_sec": round(ts, 3),
                "start_time_sec": round(max(0.0, ts - window), 3),
                "end_time_sec": round(min(duration_sec, ts + window), 3),
                "confidence": 1.0,
                "explanation": f"Soccer event: {lab.replace('_', ' ')}.",
            }
        )
    payload = {"video_id": video_id, "events": out_events}
    return json.dumps(payload, ensure_ascii=False)
