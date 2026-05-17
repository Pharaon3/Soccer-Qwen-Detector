"""Load video with OpenCV and sample frames for vision-language inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np
from PIL import Image


@dataclass
class SampledFrame:
    """One sampled frame with metadata and a PIL image."""

    frame_index: int
    timestamp_sec: float
    image: Image.Image


def _time_sample_indices(
    n_frames_video: int,
    video_fps: float,
    clip_seconds: float,
    sample_fps: float,
    max_frames: int,
) -> list[int]:
    """
    Sample frame indices at approximately ``sample_fps`` over ``clip_seconds``,
    capped by ``max_frames`` and video length.
    """
    if n_frames_video <= 0 or video_fps <= 0 or sample_fps <= 0:
        return []
    max_index = n_frames_video - 1
    effective_duration = min(clip_seconds, n_frames_video / video_fps)
    # Times at sample_fps: 0, 1/sample_fps, 2/sample_fps, ...
    indices: list[int] = []
    dt = 1.0 / sample_fps
    t = 0.0
    while t < effective_duration - 1e-9 and len(indices) < max_frames:
        idx = int(round(t * video_fps))
        idx = max(0, min(idx, max_index))
        indices.append(idx)
        t += dt
    # Remove consecutive duplicates (same physical frame)
    deduped: list[int] = []
    for idx in indices:
        if not deduped or deduped[-1] != idx:
            deduped.append(idx)
    return deduped[:max_frames]


def iter_sampled_frames(
    video_path: str,
    fps: float,
    clip_seconds: float,
    sample_fps: float,
    max_frames: int,
) -> Iterator[SampledFrame]:
    """
    Yield sampled frames with original frame index, timestamp_sec, and PIL image.

    ``timestamp_sec`` is frame_index / fps (aligned with the clip timeline).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    # Prefer config fps for timestamp mapping; still read frames by index.
    use_fps = fps if fps > 0 else (reported_fps if reported_fps > 0 else 25.0)

    indices = _time_sample_indices(n_frames, use_fps, clip_seconds, sample_fps, max_frames)

    try:
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
            timestamp_sec = float(frame_idx) / use_fps
            yield SampledFrame(frame_index=frame_idx, timestamp_sec=timestamp_sec, image=pil_image)
    finally:
        cap.release()


def load_sampled_frames(
    video_path: str,
    fps: float,
    clip_seconds: float,
    sample_fps: float,
    max_frames: int,
) -> list[SampledFrame]:
    return list(iter_sampled_frames(video_path, fps, clip_seconds, sample_fps, max_frames))
