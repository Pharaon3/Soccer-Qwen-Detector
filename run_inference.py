#!/usr/bin/env python3
"""
CLI for soccer clip event detection with Qwen2.5-VL.

Backends:
  hf  — local Hugging Face model (fine-tuned LoRA adapter recommended)
  api — OpenAI-compatible remote server (vLLM, etc.)

Examples:
  python run_inference.py --backend hf --adapter checkpoints/qwen_lora --video clip.mp4 --output out.json
  python run_inference.py --backend api --video clip.mp4 --config config.yaml --output out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocess import postprocess_model_events
from src.prompt_builder import build_system_prompt, build_user_prompt
from src.qwen_hf_client import QwenHFClient, QwenHFConfig
from src.qwen_vl_client import QwenVLClient, QwenVLConfig
from src.utils import parse_model_json
from src.video_loader import load_sampled_frames


class _InferClient:
    def infer_events(self, frames, system_prompt: str, user_text: str) -> str:
        raise NotImplementedError


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _process_one_video(
    video_path: Path,
    output_json: Path,
    cfg: dict,
    client: _InferClient,
    source_tag: str,
) -> None:
    video_cfg = cfg.get("video", {})
    qwen_cfg = cfg.get("qwen", {})
    post_cfg = cfg.get("postprocess", {})

    fps = float(video_cfg.get("fps", 25))
    clip_seconds = float(video_cfg.get("clip_seconds", 30))
    sample_fps = float(video_cfg.get("sample_fps", 2))
    max_frames = int(video_cfg.get("max_frames", 32))
    max_image_side = int(video_cfg.get("max_image_side", 896))

    frames = load_sampled_frames(
        str(video_path),
        fps=fps,
        clip_seconds=clip_seconds,
        sample_fps=sample_fps,
        max_frames=max_frames,
        max_image_side=max_image_side,
    )
    if not frames:
        raise RuntimeError(f"No frames sampled from video: {video_path}")

    video_id = video_path.stem
    frame_desc = [(f.frame_index, f.timestamp_sec) for f in frames]
    user_prompt = build_user_prompt(
        video_id=video_id,
        frame_descriptions=frame_desc,
        duration_sec=clip_seconds,
    )
    system_prompt = build_system_prompt()

    raw_text = client.infer_events(frames, system_prompt, user_prompt)

    raw_sidecar = output_json.with_name(f"{output_json.stem}_raw.txt")
    _ensure_parent(raw_sidecar)
    raw_sidecar.write_text(raw_text, encoding="utf-8")

    try:
        parsed = parse_model_json(raw_text)
    except json.JSONDecodeError as exc:
        err_path = output_json.with_name(f"{output_json.stem}_error.txt")
        err_path.write_text(
            f"JSON decode failed: {exc}\n\n--- raw model output ---\n\n{raw_text}",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Model output was not valid JSON. See {err_path} for details."
        ) from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("Parsed JSON root must be an object.")

    events_raw = parsed.get("events", [])
    if events_raw is None:
        events_raw = []
    if not isinstance(events_raw, list):
        raise RuntimeError('"events" must be a JSON array.')

    min_gap = dict(post_cfg.get("min_gap_sec", {}))
    output_fps = float(post_cfg.get("output_fps", fps))

    events_out = postprocess_model_events(
        [e for e in events_raw if isinstance(e, dict)],
        duration_sec=clip_seconds,
        output_fps=output_fps,
        min_gap_sec=min_gap,
    )

    final_payload = {
        "video_id": video_id,
        "fps": int(output_fps) if output_fps == int(output_fps) else output_fps,
        "duration_sec": clip_seconds,
        "source": source_tag,
        "events": events_out,
    }

    _ensure_parent(output_json)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Soccer event detection with Qwen2.5-VL")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="Path to a single video file")
    src.add_argument("--video_dir", type=Path, help="Directory of videos to process")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml", help="YAML config")
    parser.add_argument(
        "--backend",
        choices=("hf", "api"),
        default=None,
        help="Inference backend: hf=local fine-tuned Qwen, api=OpenAI-compatible server",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="LoRA adapter directory (hf backend; default from config qwen.adapter_path)",
    )
    parser.add_argument(
        "--train_config",
        type=Path,
        default=ROOT / "train_config.yaml",
        help="Training config for hf defaults (model_id, adapter path)",
    )
    out = parser.add_mutually_exclusive_group(required=True)
    out.add_argument("--output", type=Path, help="Output JSON path for single-video mode")
    out.add_argument("--output_dir", type=Path, help="Output directory for batch mode")
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    cfg = _load_config(cfg_path)
    q = cfg.get("qwen", {})

    train_cfg: dict = {}
    train_cfg_path = args.train_config.resolve()
    if train_cfg_path.is_file():
        train_cfg = _load_config(train_cfg_path)

    backend = args.backend or str(q.get("backend", "hf"))
    source_tag = "qwen_finetuned" if backend == "hf" else "qwen_api"

    if backend == "hf":
        q_train = train_cfg.get("qwen", {}) or {}
        adapter = args.adapter
        if adapter is None:
            apath = q.get("adapter_path") or train_cfg.get("checkpoint_dir", "checkpoints/qwen_lora")
            adapter = ROOT / str(apath)
        adapter_str = str(adapter.resolve()) if adapter else None
        client: _InferClient = QwenHFClient(
            QwenHFConfig(
                model_id=str(q.get("model_id") or q_train.get("model_id", "Qwen/Qwen2.5-VL-7B-Instruct")),
                adapter_path=adapter_str,
                torch_dtype=str(q.get("torch_dtype", q_train.get("torch_dtype", "bfloat16"))),
                max_new_tokens=int(q.get("max_tokens", 4096)),
                temperature=float(q.get("temperature", 0.2)),
            )
        )
        print(f"Backend: Hugging Face  adapter={adapter_str}")
    else:
        client = QwenVLClient(
            QwenVLConfig(
                base_url=str(q.get("base_url", "http://localhost:8000/v1")),
                api_key=str(q.get("api_key", "EMPTY")),
                model_name=str(q.get("model_name", "Qwen/Qwen2.5-VL-72B-Instruct")),
                temperature=float(q.get("temperature", 0.2)),
                max_tokens=int(q.get("max_tokens", 4096)),
                timeout_sec=float(q.get("timeout_sec", 600)),
            )
        )
        print("Backend: OpenAI-compatible API")

    if args.video is not None:
        if args.output is None:
            raise SystemExit("Single-video mode requires --output")
        video_path = args.video.resolve()
        if not video_path.is_file():
            raise SystemExit(f"Video not found: {video_path}")
        out_path = args.output.resolve()
        _process_one_video(video_path, out_path, cfg, client, source_tag)
        print(f"Wrote {out_path}")
        print(f"Wrote {out_path.with_name(out_path.stem + '_raw.txt')}")
        return

    # Batch mode
    if args.output_dir is None:
        raise SystemExit("Batch mode requires --output_dir")
    vdir = args.video_dir.resolve()
    if not vdir.is_dir():
        raise SystemExit(f"video_dir is not a directory: {vdir}")
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    videos = sorted(
        p for p in vdir.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    if not videos:
        raise SystemExit(f"No video files found in {vdir}")

    for vp in videos:
        target = out_dir / f"{vp.stem}.json"
        print(f"Processing {vp.name} -> {target.name}")
        _process_one_video(vp, target, cfg, client, source_tag)
    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
