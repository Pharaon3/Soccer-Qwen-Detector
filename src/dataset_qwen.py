"""Dataset for Qwen2.5-VL supervised fine-tuning on soccer clips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from src.annotation_to_qwen import build_training_target_json
from src.dataset_soccer import ClipRecord, load_clip_events, scan_clip_folders, train_val_split
from src.prompt_builder import build_system_prompt, build_user_prompt
from src.video_loader import load_sampled_frames

__all__ = [
    "ClipRecord",
    "QwenSoccerFineTuneDataset",
    "scan_clip_folders",
    "train_val_split",
]


@dataclass
class QwenClipSample:
    clip_id: str
    images: list[Image.Image]
    frame_descriptions: list[tuple[int, float]]
    system_prompt: str
    user_prompt: str
    target_json: str


class QwenSoccerFineTuneDataset(Dataset):
    """
    One training sample = sampled frames + prompts + target JSON from annotations.

    Uses the same frame sampling and prompt format as ``run_inference.py``.
    """

    def __init__(
        self,
        records: list[ClipRecord],
        *,
        fps: float,
        clip_seconds: float,
        sample_fps: float,
        max_frames: int,
        max_image_side: int = 0,
    ) -> None:
        self.records = records
        self.fps = fps
        self.clip_seconds = clip_seconds
        self.sample_fps = sample_fps
        self.max_frames = max_frames
        self.max_image_side = max_image_side
        self.system_prompt = build_system_prompt()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> QwenClipSample:
        rec = self.records[idx]
        if rec.video_path is None or not rec.video_path.is_file():
            raise FileNotFoundError(
                f"No video file for clip '{rec.clip_id}'. "
                f"Expected a video in: {rec.folder}"
            )

        frames = load_sampled_frames(
            str(rec.video_path),
            fps=self.fps,
            clip_seconds=self.clip_seconds,
            sample_fps=self.sample_fps,
            max_frames=self.max_frames,
            max_image_side=self.max_image_side,
        )
        if not frames:
            raise RuntimeError(f"No frames sampled from video: {rec.video_path}")

        events, _ = load_clip_events(rec.json_path)
        frame_desc = [(f.frame_index, f.timestamp_sec) for f in frames]
        images = [f.image for f in frames]
        user_prompt = build_user_prompt(
            video_id=rec.clip_id,
            frame_descriptions=frame_desc,
            duration_sec=self.clip_seconds,
        )
        target_json = build_training_target_json(
            rec.clip_id, events, self.clip_seconds
        )

        return QwenClipSample(
            clip_id=rec.clip_id,
            images=images,
            frame_descriptions=frame_desc,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            target_json=target_json,
        )

    def build_messages(self, sample: QwenClipSample) -> list[dict[str, Any]]:
        """Chat messages for Hugging Face processor (system / user / assistant)."""
        user_content: list[dict[str, Any]] = [
            {"type": "image", "image": img} for img in sample.images
        ]
        user_content.append({"type": "text", "text": sample.user_prompt})
        return [
            {"role": "system", "content": sample.system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": sample.target_json},
        ]
