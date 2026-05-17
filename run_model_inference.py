#!/usr/bin/env python3
"""
Run local CNN inference on soccer clips using a trained checkpoint.

Examples:
  python run_model_inference.py --video data/videos/clip.mp4 --checkpoint checkpoints/last.pt \\
      --config train_config.yaml --output outputs_model/clip.json
  python run_model_inference.py --video_dir data/videos --checkpoint checkpoints/last.pt \\
      --config train_config.yaml --output_dir outputs_model/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_soccer import load_clip_frames_for_model
from src.model_inference import (
    load_inference_config,
    load_trained_model,
    run_local_inference_on_clip,
)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _process_one_video(
    video_path: Path,
    output_json: Path,
    cfg: dict,
    model: torch.nn.Module,
    device: torch.device,
    thresholds: dict[str, float],
    min_gap_sec: dict[str, float],
    output_fps: float,
) -> None:
    duration_sec = float(cfg.get("duration_sec", 30.0))
    video_fps = float(cfg.get("video_fps", 25.0))
    num_segments = int(cfg.get("num_segments", 30))
    image_size = int(cfg.get("image_size", 224))

    frames = load_clip_frames_for_model(
        video_path,
        num_segments=num_segments,
        image_size=image_size,
        duration_sec=duration_sec,
        video_fps=video_fps,
    )

    events = run_local_inference_on_clip(
        model,
        frames,
        device,
        num_segments=num_segments,
        duration_sec=duration_sec,
        output_fps=output_fps,
        thresholds=thresholds,
        min_gap_sec=min_gap_sec,
    )

    payload = {
        "video_id": video_path.stem,
        "fps": int(output_fps) if output_fps == int(output_fps) else output_fps,
        "duration_sec": duration_sec,
        "source": "local_model",
        "events": events,
    }

    _ensure_parent(output_json)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local CNN soccer event inference")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="Path to a single video file")
    src.add_argument("--video_dir", type=Path, help="Directory of videos to process")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint (.pt)")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "train_config.yaml",
        help="Training/inference YAML config",
    )
    out = parser.add_mutually_exclusive_group(required=True)
    out.add_argument("--output", type=Path, help="Output JSON for single-video mode")
    out.add_argument("--output_dir", type=Path, help="Output directory for batch mode")
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    cfg = _load_config(cfg_path)

    ckpt_path = args.checkpoint.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_trained_model(ckpt_path, device)
    thresholds, min_gap_sec, output_fps = load_inference_config(cfg)

    if args.video is not None:
        if args.output is None:
            raise SystemExit("Single-video mode requires --output")
        video_path = args.video.resolve()
        if not video_path.is_file():
            raise SystemExit(f"Video not found: {video_path}")
        out_path = args.output.resolve()
        _process_one_video(
            video_path, out_path, cfg, model, device, thresholds, min_gap_sec, output_fps
        )
        print(f"Wrote {out_path}")
        return

    if args.output_dir is None:
        raise SystemExit("Batch mode requires --output_dir")
    vdir = args.video_dir.resolve()
    if not vdir.is_dir():
        raise SystemExit(f"video_dir is not a directory: {vdir}")
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}
    videos = sorted(p for p in vdir.iterdir() if p.is_file() and p.suffix.lower() in exts)
    if not videos:
        raise SystemExit(f"No video files found in {vdir}")

    for vp in videos:
        target = out_dir / f"{vp.stem}.json"
        print(f"Processing {vp.name} -> {target.name}")
        _process_one_video(
            vp, target, cfg, model, device, thresholds, min_gap_sec, output_fps
        )
    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
