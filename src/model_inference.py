"""Local CNN inference: checkpoint loading, segment probs → events."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.model_temporal import SegmentEventModel
from src.postprocess import postprocess_local_events
from src.schema import ALLOWED_EVENT_CLASSES, CLASS_TO_IDX, DEFAULT_MIN_GAP_SEC, NUM_EVENT_CLASSES
def default_thresholds() -> dict[str, float]:
    return {c: 0.35 for c in ALLOWED_EVENT_CLASSES}


def load_inference_config(cfg: dict[str, Any]) -> tuple[dict[str, float], dict[str, float], float]:
    """Return (thresholds, min_gap_sec, output_fps) from train config."""
    inf = cfg.get("inference", {}) or {}
    thr_raw = dict(inf.get("thresholds", {}) or {})
    gaps_raw = dict(inf.get("min_gap_sec", {}) or {})

    thresholds = default_thresholds()
    for cls in ALLOWED_EVENT_CLASSES:
        if cls in thr_raw:
            thresholds[cls] = float(thr_raw[cls])

    min_gap = dict(DEFAULT_MIN_GAP_SEC)
    for cls, gap in gaps_raw.items():
        if cls in CLASS_TO_IDX:
            min_gap[cls] = float(gap)

    output_fps = float(cfg.get("video_fps", 25))
    return thresholds, min_gap, output_fps


def load_trained_model(
    checkpoint_path: Path,
    device: torch.device,
    pretrained_backbone: bool | None = None,
) -> SegmentEventModel:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = SegmentEventModel(num_classes=NUM_EVENT_CLASSES, pretrained=False)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        if pretrained_backbone is None:
            extra = ckpt.get("extra") or {}
            saved_cfg = extra.get("config") or {}
            pretrained_backbone = bool(saved_cfg.get("pretrained_backbone", True))
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_segment_probs(
    model: nn.Module,
    frames: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Run model on one clip.

    ``frames``: (S, 3, H, W) → returns (S, num_classes) sigmoid probabilities.
    """
    x = frames.unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    return probs.astype(np.float32)


def segment_center_timestamp(segment_index: int, num_segments: int, duration_sec: float) -> float:
    seg_len = duration_sec / float(num_segments)
    return (segment_index + 0.5) * seg_len


def probs_to_raw_events(
    probs: np.ndarray,
    thresholds: dict[str, float],
    num_segments: int,
    duration_sec: float,
) -> list[dict[str, Any]]:
    """Convert segment probabilities to instantaneous detection events."""
    events: list[dict[str, Any]] = []
    n_seg, n_cls = probs.shape
    if n_seg != num_segments:
        raise ValueError(f"Expected {num_segments} segments, got {n_seg}")

    for seg_i in range(n_seg):
        ts = segment_center_timestamp(seg_i, num_segments, duration_sec)
        for cls in ALLOWED_EVENT_CLASSES:
            ci = CLASS_TO_IDX[cls]
            conf = float(probs[seg_i, ci])
            if conf >= thresholds[cls]:
                events.append(
                    {
                        "class": cls,
                        "timestamp_sec": round(ts, 4),
                        "confidence": round(conf, 4),
                        "segment_index": seg_i,
                    }
                )
    return events


def run_local_inference_on_clip(
    model: nn.Module,
    frames: torch.Tensor,
    device: torch.device,
    *,
    num_segments: int,
    duration_sec: float,
    output_fps: float,
    thresholds: dict[str, float],
    min_gap_sec: dict[str, float],
) -> list[dict[str, Any]]:
    """Full local inference pipeline for one clip tensor."""
    probs = predict_segment_probs(model, frames, device)
    raw = probs_to_raw_events(probs, thresholds, num_segments, duration_sec)
    return postprocess_local_events(raw, duration_sec, output_fps, min_gap_sec)
