#!/usr/bin/env python3
"""
Convert Qwen inference JSON into training/annotator annotation format.

Examples:
  python convert_qwen_to_annotations.py --input outputs_qwen/clip.json \\
      --output dataset/clip_001/clip_001.json --min_confidence 0.5
  python convert_qwen_to_annotations.py --input_dir outputs_qwen/ \\
      --dataset_dir dataset/ --min_confidence 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.schema import ALLOWED_SET, normalize_class_key


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def convert_qwen_output(
    qwen_data: dict[str, Any],
    min_confidence: float,
) -> dict[str, Any]:
    """Transform one Qwen prediction file into annotator JSON."""
    raw_events = qwen_data.get("events", [])
    if raw_events is None:
        raw_events = []
    if not isinstance(raw_events, list):
        raise ValueError('"events" must be a JSON array')

    events_out: list[dict[str, Any]] = []
    for raw in raw_events:
        if not isinstance(raw, dict):
            continue
        row = normalize_class_key(raw)
        lab = row.get("class") or row.get("label") or ""
        if not isinstance(lab, str):
            lab = str(lab)
        lab = lab.strip()
        if lab not in ALLOWED_SET:
            continue

        conf = _as_float(row.get("confidence"), 1.0)
        if conf < min_confidence:
            continue

        ts = _as_float(row.get("timestamp_sec", row.get("time_sec")), 0.0)
        ann: dict[str, Any] = {
            "time_sec": round(ts, 4),
            "label": lab,
            "source": "qwen",
            "confidence": round(conf, 4),
        }
        expl = row.get("explanation")
        if expl is not None and str(expl).strip():
            ann["explanation"] = str(expl).strip()
        events_out.append(ann)

    events_out.sort(key=lambda e: e["time_sec"])
    return {
        "events": events_out,
        "environment": "unknown",
    }


def _write_annotation(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Qwen outputs to training annotations")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="Single Qwen output JSON")
    src.add_argument("--input_dir", type=Path, help="Directory of Qwen output JSON files")
    dst = parser.add_mutually_exclusive_group(required=True)
    dst.add_argument("--output", type=Path, help="Output annotation JSON (single-file mode)")
    dst.add_argument("--dataset_dir", type=Path, help="Dataset root for batch mode")
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.5,
        help="Minimum event confidence to keep (default: 0.5)",
    )
    args = parser.parse_args()

    if args.input is not None:
        if args.output is None:
            raise SystemExit("Single-file mode requires --output")
        in_path = args.input.resolve()
        if not in_path.is_file():
            raise SystemExit(f"Input not found: {in_path}")
        with in_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise SystemExit("Input JSON root must be an object")
        payload = convert_qwen_output(data, args.min_confidence)
        out_path = args.output.resolve()
        _write_annotation(out_path, payload)
        print(f"Wrote {out_path} ({len(payload['events'])} events)")
        return

    in_dir = args.input_dir.resolve()
    if not in_dir.is_dir():
        raise SystemExit(f"input_dir is not a directory: {in_dir}")
    if args.dataset_dir is None:
        raise SystemExit("Batch mode requires --dataset_dir")

    dataset_root = args.dataset_dir.resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    json_files = sorted(p for p in in_dir.glob("*.json") if p.is_file())
    if not json_files:
        raise SystemExit(f"No JSON files found in {in_dir}")

    total_events = 0
    for jp in json_files:
        if jp.name.endswith("_error.json"):
            continue
        with jp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"Skipping invalid JSON: {jp.name}")
            continue
        clip_id = jp.stem
        payload = convert_qwen_output(data, args.min_confidence)
        out_path = dataset_root / clip_id / f"{clip_id}.json"
        _write_annotation(out_path, payload)
        total_events += len(payload["events"])
        print(f"{jp.name} -> {out_path} ({len(payload['events'])} events)")

    print(f"Done. Wrote {len(json_files)} annotation files ({total_events} total events)")


if __name__ == "__main__":
    main()
