"""Load Qwen2.5-VL, apply LoRA, and run supervised fine-tuning."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.dataset_qwen import QwenClipSample, QwenSoccerFineTuneDataset


def _require_training_deps() -> tuple[Any, Any, Any, Any]:
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "Qwen fine-tuning requires: pip install transformers peft accelerate "
            "(and a CUDA-capable GPU is strongly recommended)."
        ) from exc
    return (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
        LoraConfig,
        get_peft_model,
    )


def resolve_torch_dtype(name: str) -> torch.dtype:
    key = (name or "bfloat16").lower()
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if key in ("fp16", "float16"):
        return torch.float16
    if key in ("fp32", "float32"):
        return torch.float32
    return torch.bfloat16


def load_qwen_for_training(cfg: dict[str, Any], device: torch.device) -> tuple[Any, Any]:
    """Load base Qwen2.5-VL and optionally wrap with LoRA."""
    (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
        LoraConfig,
        get_peft_model,
    ) = _require_training_deps()

    qcfg = cfg.get("qwen", {}) or {}
    model_id = str(qcfg.get("model_id", "Qwen/Qwen2.5-VL-7B-Instruct"))
    dtype = resolve_torch_dtype(str(qcfg.get("torch_dtype", "bfloat16")))

    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if bool(qcfg.get("load_in_4bit", False)):
        try:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"
        except ImportError as exc:
            raise ImportError(
                "4-bit loading requires: pip install bitsandbytes"
            ) from exc
    else:
        load_kwargs["device_map"] = "auto" if device.type == "cuda" else None

    print(f"Loading {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)

    if bool(qcfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if bool(qcfg.get("use_lora", True)):
        target_modules = qcfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        lora_config = LoraConfig(
            r=int(qcfg.get("lora_r", 16)),
            lora_alpha=int(qcfg.get("lora_alpha", 32)),
            lora_dropout=float(qcfg.get("lora_dropout", 0.05)),
            target_modules=list(target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if device.type == "cpu" and load_kwargs.get("device_map") is None:
        model = model.to(device)

    return model, processor


def build_training_batch(
    processor: Any,
    dataset: QwenSoccerFineTuneDataset,
    sample: QwenClipSample,
) -> dict[str, torch.Tensor]:
    """Tokenize one clip; mask prompt tokens so loss is on assistant JSON only."""
    messages = dataset.build_messages(sample)
    prompt_messages = messages[:-1]

    text_full = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    text_prompt = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = processor(
        text=[text_full],
        images=sample.images,
        padding=True,
        return_tensors="pt",
    )
    prompt_inputs = processor(
        text=[text_prompt],
        images=sample.images,
        padding=True,
        return_tensors="pt",
    )

    labels = model_inputs["input_ids"].clone()
    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels[:, :prompt_len] = -100
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100

    model_inputs["labels"] = labels
    return model_inputs


def _prune_epoch_checkpoints(ckpt_dir: Path, keep_last_n: int) -> None:
    """Delete old ``epoch_XXX`` folders; keep only the newest ``keep_last_n``."""
    if keep_last_n <= 0:
        return
    epoch_dirs = sorted(
        (p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.startswith("epoch_")),
        key=lambda p: p.name,
    )
    while len(epoch_dirs) > keep_last_n:
        old = epoch_dirs.pop(0)
        shutil.rmtree(old, ignore_errors=True)
        print(f"  Removed old checkpoint {old.name}")


def train_qwen(
    model: Any,
    processor: Any,
    train_ds: QwenSoccerFineTuneDataset,
    val_ds: QwenSoccerFineTuneDataset | None,
    cfg: dict[str, Any],
    ckpt_dir: Path,
    device: torch.device,
) -> None:
    """
    Fine-tuning loop.

    ``training.batch_size`` = clips averaged per backward (micro-batch; sequential forwards).
    Effective batch ≈ ``batch_size * gradient_accumulation_steps`` clips per optimizer step.
    """
    tcfg = cfg.get("training", {}) or {}
    epochs = int(tcfg.get("epochs", 10))
    lr = float(tcfg.get("lr", 2e-5))
    wd = float(tcfg.get("weight_decay", 0.01))
    micro_batch = max(1, int(tcfg.get("batch_size", 1)))
    grad_accum = int(tcfg.get("gradient_accumulation_steps", 8))
    max_grad_norm = float(tcfg.get("max_grad_norm", 1.0))
    save_every = int(tcfg.get("save_every_epochs", 1))
    keep_last_n = int(tcfg.get("keep_last_n_checkpoints", 2))
    log_every = int(tcfg.get("log_every_steps", 4))

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)

    def _collate_one(batch: list[QwenClipSample]) -> QwenClipSample:
        return batch[0]

    loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_one,
    )

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        accum_loss = 0.0
        micro_steps = 0
        micro_losses: list[torch.Tensor] = []

        for sample in loader:
            batch = build_training_batch(processor, train_ds, sample)
            batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch.items()}

            outputs = model(**batch)
            micro_losses.append(outputs.loss / micro_batch)

            if len(micro_losses) < micro_batch:
                continue

            step_loss = torch.stack(micro_losses).sum()
            (step_loss / grad_accum).backward()
            accum_loss += float(step_loss.detach())
            micro_losses = []
            micro_steps += 1

            if micro_steps % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % log_every == 0:
                    print(f"  step {global_step}  loss={accum_loss / grad_accum:.4f}")
                running_loss += accum_loss
                accum_loss = 0.0

        if micro_losses:
            step_loss = torch.stack(micro_losses).sum()
            (step_loss / grad_accum).backward()
            accum_loss += float(step_loss.detach())
            micro_steps += 1

        if micro_steps % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            running_loss += accum_loss

        train_loss = running_loss / max(len(loader), 1)
        val_loss = _eval_loss(model, processor, val_ds) if val_ds else None
        msg = f"epoch {epoch + 1}/{epochs}  train_loss≈{train_loss:.4f}"
        if val_loss is not None:
            msg += f"  val_loss={val_loss:.4f}"
        print(msg)

        if (epoch + 1) % save_every == 0 or epoch + 1 == epochs:
            save_dir = ckpt_dir / f"epoch_{epoch + 1:03d}"
            _save_adapter(model, processor, save_dir, cfg)
            _prune_epoch_checkpoints(ckpt_dir, keep_last_n)

    _save_adapter(model, processor, ckpt_dir, cfg)
    print(f"Saved final adapter to {ckpt_dir}")


@torch.no_grad()
def _eval_loss(
    model: Any,
    processor: Any,
    val_ds: QwenSoccerFineTuneDataset,
) -> float | None:
    if len(val_ds) == 0:
        return None
    model.eval()
    total = 0.0
    n = 0
    for i in range(len(val_ds)):
        sample = val_ds[i]
        batch = build_training_batch(processor, val_ds, sample)
        batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch.items()}
        outputs = model(**batch)
        total += float(outputs.loss.detach())
        n += 1
    model.train()
    return total / max(n, 1)


def _save_adapter(model: Any, processor: Any, save_dir: Path, cfg: dict[str, Any]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    meta = {"config": cfg, "model_type": "qwen2.5-vl-lora"}
    (save_dir / "soccer_qwen_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def load_qwen_for_inference(
    model_id: str,
    adapter_path: Path | None,
    device: torch.device,
    torch_dtype: str = "bfloat16",
) -> tuple[Any, Any]:
    """Load fine-tuned Qwen (base + optional LoRA adapter) for local inference."""
    (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
        _LoraConfig,
        _get_peft_model,
    ) = _require_training_deps()
    from peft import PeftModel

    dtype = resolve_torch_dtype(torch_dtype)
    if adapter_path is not None and adapter_path.is_dir():
        processor = AutoProcessor.from_pretrained(str(adapter_path), trust_remote_code=True)
    else:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    if adapter_path is not None and adapter_path.is_dir():
        meta_file = adapter_path / "soccer_qwen_meta.json"
        if meta_file.is_file():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            saved_id = meta.get("config", {}).get("qwen", {}).get("model_id")
            if saved_id:
                model_id = saved_id

        print(f"Loading base model {model_id} + adapter {adapter_path}")
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, str(adapter_path))
    else:
        print(f"Loading base model {model_id} (no adapter)")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )

    if device.type == "cpu" and not hasattr(model, "hf_device_map"):
        model = model.to(device)

    model.eval()
    return model, processor
