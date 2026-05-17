"""OpenAI-compatible client for Qwen2.5-VL and similar vision-language models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from src.utils import data_url_from_jpeg_base64, pil_to_jpeg_base64
from src.video_loader import SampledFrame


@dataclass
class QwenVLConfig:
    base_url: str
    api_key: str
    model_name: str
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout_sec: float = 600.0


class QwenVLClient:
    """
    Thin wrapper around the OpenAI SDK for chat.completions with vision content.

    Replace this class later if you switch to another VLM with a different API.
    """

    def __init__(self, cfg: QwenVLConfig) -> None:
        self._cfg = cfg
        self._client = OpenAI(
            base_url=cfg.base_url.rstrip("/"),
            api_key=cfg.api_key,
            timeout=cfg.timeout_sec,
        )

    def infer_events(
        self,
        frames: list[SampledFrame],
        system_prompt: str,
        user_text: str,
    ) -> str:
        """
        Send frames (as base64 JPEG data URLs) plus text; return raw assistant string.
        """
        content: list[dict[str, Any]] = []
        for sf in frames:
            b64 = pil_to_jpeg_base64(sf.image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url_from_jpeg_base64(b64)},
                }
            )
        content.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        resp = self._client.chat.completions.create(
            model=self._cfg.model_name,
            messages=messages,
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
        )
        choice = resp.choices[0]
        msg = choice.message
        text = msg.content or ""
        return text
