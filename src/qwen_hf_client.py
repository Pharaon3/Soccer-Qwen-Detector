"""Local Hugging Face inference for Qwen2.5-VL (base or fine-tuned LoRA)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.qwen_finetune import load_qwen_for_inference
from src.video_loader import SampledFrame


@dataclass
class QwenHFConfig:
    model_id: str
    adapter_path: str | None = None
    torch_dtype: str = "bfloat16"
    max_new_tokens: int = 4096
    temperature: float = 0.2


class QwenHFClient:
    """Run Qwen2.5-VL locally via transformers (optionally with a LoRA adapter)."""

    def __init__(self, cfg: QwenHFConfig) -> None:
        self._cfg = cfg
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adapter = Path(cfg.adapter_path).resolve() if cfg.adapter_path else None
        self._model, self._processor = load_qwen_for_inference(
            model_id=cfg.model_id,
            adapter_path=adapter,
            device=self._device,
            torch_dtype=cfg.torch_dtype,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    def infer_events(
        self,
        frames: list[SampledFrame],
        system_prompt: str,
        user_text: str,
    ) -> str:
        user_content: list[dict[str, Any]] = [
            {"type": "image", "image": sf.image} for sf in frames
        ]
        user_content.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=[sf.image for sf in frames],
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(self._model.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self._cfg.max_new_tokens,
        }
        if self._cfg.temperature > 0:
            gen_kwargs["temperature"] = self._cfg.temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, prompt_len:]
        decoded = self._processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return decoded[0] if decoded else ""
