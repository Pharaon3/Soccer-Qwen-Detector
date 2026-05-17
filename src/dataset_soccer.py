"""
Load `dataset/<clip_id>/*.json` + sibling video for temporal event training.

Annotator JSON format:
  { "events": [ { "time_sec": float, "label": str }, ... ], "environment": optional }
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.schema import ALLOWED_SET, CLASS_TO_IDX, NUM_EVENT_CLASSES


VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v")


@dataclass
class ClipRecord:
    clip_id: str
    folder: Path
    json_path: Path
    video_path: Path | None


def find_video_in_folder(folder: Path) -> Path | None:
    names = [p for p in folder.iterdir() if p.is_file()]
    for p in names:
        if p.suffix.lower() in VIDEO_EXTENSIONS:
            return p
    return None


def find_json_in_folder(folder: Path) -> Path | None:
    """Prefer `<folder_name>.json` if present, else first `*.json` (excludes odd temp files)."""
    json_files = sorted(
        p for p in folder.glob("*.json") if p.is_file() and p.suffix.lower() == ".json"
    )
    if not json_files:
        return None
    preferred = folder / f"{folder.name}.json"
    if preferred in json_files:
        return preferred
    return json_files[0]


def scan_clip_folders(dataset_root: Path, skip_dir_names: frozenset[str]) -> list[ClipRecord]:
    """
    Each immediate subfolder of ``dataset_root`` that contains a JSON is a clip.
    Skips ``annotator`` and other names in ``skip_dir_names``.
    """
    out: list[ClipRecord] = []
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir() or child.name in skip_dir_names:
            continue
        jp = find_json_in_folder(child)
        if jp is None:
            continue
        vid = find_video_in_folder(child)
        out.append(ClipRecord(clip_id=child.name, folder=child, json_path=jp, video_path=vid))
    return out


def load_clip_events(json_path: Path) -> tuple[list[tuple[float, str]], dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        root = json.load(f)
    if not isinstance(root, dict):
        return [], {}
    raw = root.get("events", [])
    pairs: list[tuple[float, str]] = []
    if isinstance(raw, list):
        for row in raw:
            if not isinstance(row, dict):
                continue
            lab = row.get("label", "")
            if not isinstance(lab, str):
                lab = str(lab) if lab is not None else ""
            lab = lab.strip()
            try:
                ts = float(row.get("time_sec", 0.0))
            except (TypeError, ValueError):
                ts = 0.0
            if lab in ALLOWED_SET:
                pairs.append((ts, lab))
    return pairs, root


def events_to_segment_targets(
    events: list[tuple[float, str]],
    num_segments: int,
    duration_sec: float,
) -> np.ndarray:
    """
    Build (num_segments, NUM_EVENT_CLASSES) multi-label targets.

    Segment ``i`` covers [i * duration/num_segments, (i+1) * duration/num_segments).
    """
    y = np.zeros((num_segments, NUM_EVENT_CLASSES), dtype=np.float32)
    if num_segments <= 0 or duration_sec <= 0:
        return y
    seg_len = duration_sec / float(num_segments)
    for ts, lab in events:
        ts = float(np.clip(ts, 0.0, duration_sec - 1e-6))
        idx = min(int(ts // seg_len), num_segments - 1)
        y[idx, CLASS_TO_IDX[lab]] = 1.0
    return y


def read_frame_at_time(video_path: Path, t_sec: float, config_fps: float) -> np.ndarray | None:
    """Return RGB uint8 HxWx3 or None on failure."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        use_fps = float(cap.get(cv2.CAP_PROP_FPS)) or config_fps
        if use_fps <= 0:
            use_fps = config_fps
        if n_frames > 0:
            duration = n_frames / use_fps
            t_clipped = float(np.clip(t_sec, 0.0, max(0.0, duration - 1.0 / use_fps)))
        else:
            t_clipped = max(0.0, t_sec)
        frame_idx = int(round(t_clipped * use_fps))
        if n_frames > 0:
            frame_idx = max(0, min(frame_idx, n_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


class SoccerClipSegmentDataset(Dataset):
    """
    One sample = full clip as ``num_segments`` frames (segment centers) + segment labels.
    """

    def __init__(
        self,
        records: list[ClipRecord],
        duration_sec: float,
        video_fps: float,
        num_segments: int,
        image_size: int,
        train: bool,
        mock_video_if_missing: bool,
    ) -> None:
        self.records = records
        self.duration_sec = duration_sec
        self.video_fps = video_fps
        self.num_segments = num_segments
        self.mock_video_if_missing = mock_video_if_missing

        aug = []
        if train:
            aug.append(transforms.RandomHorizontalFlip(p=0.5))
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                *aug,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def _segment_center_times(self) -> list[float]:
        seg_len = self.duration_sec / float(self.num_segments)
        return [(i + 0.5) * seg_len for i in range(self.num_segments)]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        rec = self.records[idx]
        events, _ = load_clip_events(rec.json_path)
        y = events_to_segment_targets(events, self.num_segments, self.duration_sec)
        target = torch.from_numpy(y)

        times = self._segment_center_times()
        frames: list[torch.Tensor] = []

        use_mock = rec.video_path is None or not rec.video_path.is_file()
        if use_mock and not self.mock_video_if_missing:
            raise FileNotFoundError(f"No video for clip {rec.clip_id}: {rec.folder}")

        for t in times:
            if use_mock:
                arr = np.zeros((360, 640, 3), dtype=np.uint8)
            else:
                assert rec.video_path is not None
                arr = read_frame_at_time(rec.video_path, t, self.video_fps)
                if arr is None:
                    arr = np.zeros((360, 640, 3), dtype=np.uint8)
            pil = Image.fromarray(arr)
            frames.append(self.transform(pil))

        x = torch.stack(frames, dim=0)
        return x, target, rec.clip_id


def train_val_split(
    records: list[ClipRecord],
    val_ratio: float,
    seed: int,
) -> tuple[list[ClipRecord], list[ClipRecord]]:
    rng = random.Random(seed)
    idxs = list(range(len(records)))
    rng.shuffle(idxs)
    if len(records) < 2:
        return records, []
    n_val = int(round(len(records) * val_ratio))
    n_val = max(1, min(n_val, len(records) - 1))
    val_set = set(idxs[:n_val])
    train = [records[i] for i in idxs if i not in val_set]
    val = [records[i] for i in idxs if i in val_set]
    return train, val
