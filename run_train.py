#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-VL on soccer clip annotations (LoRA by default).

Each training sample: sampled video frames + detection prompt → target JSON from labels.
This trains Qwen itself (via LoRA adapters), not a separate ResNet/CNN head.

Example:
  python run_train.py --config train_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_qwen import QwenSoccerFineTuneDataset, scan_clip_folders, train_val_split
from src.qwen_finetune import load_qwen_for_training, train_qwen
from src.train_utils import set_seed


def _load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-VL on dataset/ clip annotations"
    )
    ap.add_argument("--config", type=Path, default=ROOT / "train_config.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    ap.add_argument(
        "--max_clips",
        type=int,
        default=None,
        help="Override max_clips for quick tests",
    )
    ap.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Override video.max_frames (lower if CUDA OOM)",
    )
    ap.add_argument(
        "--sample_fps",
        type=float,
        default=None,
        help="Override video.sample_fps",
    )
    args = ap.parse_args()
    cfg = _load_cfg(args.config.resolve())

    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.max_clips is not None:
        cfg["max_clips"] = int(args.max_clips)
    if args.max_frames is not None:
        cfg.setdefault("video", {})["max_frames"] = int(args.max_frames)
    if args.sample_fps is not None:
        cfg.setdefault("video", {})["sample_fps"] = float(args.sample_fps)

    dataset_root = (ROOT / str(cfg.get("dataset_root", "dataset"))).resolve()
    skip = frozenset({str(cfg.get("annotator_dir_name", "annotator"))})
    all_records = scan_clip_folders(dataset_root, skip_dir_names=skip)

    usable = [r for r in all_records if r.video_path is not None and r.video_path.is_file()]
    max_clips = int(cfg.get("max_clips") or 0)
    if max_clips > 0:
        usable = usable[:max_clips]

    print(f"Scanned {len(all_records)} clip folders under {dataset_root}")
    print(f"  Clips with video: {len(usable)}" + (f" (max_clips={max_clips})" if max_clips else ""))

    if not usable:
        raise SystemExit(
            f"\nNo clips to train on. Each folder under {dataset_root} needs:\n"
            "  - an annotation JSON (events with time_sec + label)\n"
            "  - a video file (.mp4, .webm, .mkv, .mov, .avi, .m4v)\n"
        )

    if not torch.cuda.is_available():
        print(
            "WARNING: CUDA not available. Qwen2.5-VL fine-tuning on CPU is extremely slow "
            "and may run out of memory. Use a GPU."
        )

    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_ratio = float(cfg.get("val_ratio", 0.15))
    train_records, val_records = train_val_split(usable, val_ratio, int(cfg.get("seed", 42)))

    vcfg = cfg.get("video", {}) or {}
    fps = float(vcfg.get("fps", 25))
    clip_seconds = float(vcfg.get("clip_seconds", 30))
    sample_fps = float(vcfg.get("sample_fps", 2))
    max_frames = int(vcfg.get("max_frames", 16))
    max_image_side = int(vcfg.get("max_image_side", 896))

    if max_frames > 48:
        print(
            f"WARNING: max_frames={max_frames} is high for Qwen2.5-VL 7B on a ~32GB GPU.\n"
            "  If you hit CUDA OOM, try: --max_frames 24 or set video.max_frames: 32 in train_config.yaml.\n"
            "  (240 frames in one prompt typically needs 80GB+ or chunked video mode.)"
        )

    train_ds = QwenSoccerFineTuneDataset(
        train_records,
        fps=fps,
        clip_seconds=clip_seconds,
        sample_fps=sample_fps,
        max_frames=max_frames,
        max_image_side=max_image_side,
    )
    val_ds = QwenSoccerFineTuneDataset(
        val_records if val_records else train_records[:1],
        fps=fps,
        clip_seconds=clip_seconds,
        sample_fps=sample_fps,
        max_frames=max_frames,
        max_image_side=max_image_side,
    )

    tcfg = cfg.get("training", {}) or {}
    micro_b = max(1, int(tcfg.get("batch_size", 1)))
    grad_a = int(tcfg.get("gradient_accumulation_steps", 8))
    print(f"Train clips: {len(train_ds)}  Val clips: {len(val_ds)}")
    print(
        f"Frames per clip: up to {max_frames} @ {sample_fps} sample_fps "
        f"(~{sample_fps * clip_seconds:.0f} samples, capped)"
    )
    if max_image_side > 0:
        print(f"Max image side before processor: {max_image_side}px (source may be 1920x1080)")
    print(f"Micro-batch: {micro_b} clips  |  grad_accum: {grad_a}  |  effective batch: {micro_b * grad_a}")

    model, processor = load_qwen_for_training(cfg, device)
    ckpt_dir = (ROOT / str(cfg.get("checkpoint_dir", "checkpoints/qwen_lora"))).resolve()

    train_qwen(model, processor, train_ds, val_ds, cfg, ckpt_dir, device)


if __name__ == "__main__":
    main()
