# Soccer Qwen VL Event Detector

End-to-end pipeline for **30-second soccer clips**. Detect **15 subnet44-style event classes** (not SoccerNet) by **fine-tuning Qwen2.5-VL** on your annotations.

## How it works

```
Annotations + videos  →  run_train.py (LoRA fine-tune Qwen2.5-VL)
                              ↓
                    checkpoints/qwen_lora/
                              ↓
              run_inference.py --backend hf  →  event JSON
```

Training **starts from Qwen2.5-VL** (default: `Qwen/Qwen2.5-VL-7B-Instruct`) and learns to output the same JSON event format as inference. It does **not** train a separate ResNet head.

Optional: `--backend api` uses a remote Qwen server (vLLM) without fine-tuning. Legacy CNN tools remain in `run_train_cnn.py` / `run_model_inference.py`.

## Install

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
```

GPU strongly recommended for training. For 4-bit QLoRA on smaller GPUs:

```bash
pip install bitsandbytes
```

Set `qwen.load_in_4bit: true` in `train_config.yaml`.

## Dataset layout

```
dataset/<clip_id>/<clip_id>.json   # events: [{ "time_sec", "label" }, ...]
dataset/<clip_id>/<video>.mp4
```

Use `dataset/annotator/` (browser tool) to create or edit labels.

## Workflows

### 1. Fine-tune Qwen2.5-VL (main training)

```bash
python run_train.py --config train_config.yaml
```

- Loads `Qwen/Qwen2.5-VL-7B-Instruct` + LoRA adapters
- Each sample: sampled frames + prompt → target JSON from your labels
- Saves adapter to `checkpoints/qwen_lora/`

Quick test on a subset:

```bash
python run_train.py --max_clips 4 --epochs 1
```

Tune VRAM in `train_config.yaml`:

- `video.max_frames` — fewer frames = less memory (default **16**)
- `qwen.model_id` — e.g. `Qwen/Qwen2.5-VL-3B-Instruct` for smaller GPUs
- `qwen.load_in_4bit: true` — QLoRA

### 2. Run inference (fine-tuned model)

```bash
python run_inference.py --backend hf --adapter checkpoints/qwen_lora \
  --video data/videos/clip.mp4 --output outputs/clip.json
```

Default backend in `config.yaml` is `hf`. Batch:

```bash
python run_inference.py --backend hf --video_dir data/videos --output_dir outputs/
```

### 3. Remote API (no fine-tuning)

```bash
python run_inference.py --backend api --config config.yaml \
  --video data/videos/clip.mp4 --output outputs/clip.json
```

### 4. Convert Qwen API labels → dataset (optional bootstrap)

If you used the API to pseudo-label before training:

```bash
python convert_qwen_to_annotations.py --input_dir outputs_qwen/ --dataset_dir dataset/ --min_confidence 0.5
```

### 5. Evaluate

```bash
python run_eval.py --pred_dir outputs --gt_dir dataset --tolerance_sec 1.0
```

## Event classes

`pass`, `pass_received`, `recovery`, `tackle`, `interception`, `ball_out_of_play`, `clearance`, `take_on`, `substitution`, `block`, `aerial_duel`, `shot`, `save`, `foul`, `goal`

Defined in `src/schema.py`.

## Config files

| File | Purpose |
|------|---------|
| `train_config.yaml` | Qwen model id, LoRA, video sampling, training hyperparameters |
| `config.yaml` | Inference backend (`hf` / `api`), adapter path, postprocess |

## Project layout

| File | Role |
|------|------|
| `run_train.py` | **Fine-tune Qwen2.5-VL (LoRA)** |
| `run_inference.py` | Inference (`--backend hf` or `api`) |
| `src/qwen_finetune.py` | Model load, LoRA, training loop |
| `src/dataset_qwen.py` | Training dataset (frames + target JSON) |
| `src/qwen_hf_client.py` | Local HF inference |
| `src/qwen_vl_client.py` | OpenAI-compatible API client |
| `run_train_cnn.py` | Legacy ResNet training (optional) |
| `run_model_inference.py` | Legacy CNN inference (optional) |

## Output JSON

```json
{
  "video_id": "clip_001",
  "fps": 25,
  "duration_sec": 30,
  "source": "qwen_finetuned",
  "events": [
    {
      "class": "pass",
      "timestamp_sec": 12.4,
      "frame_index": 310,
      "confidence": 0.82,
      "explanation": "..."
    }
  ]
}
```

## Limitations

- Fine-tuning 7B+ VLMs needs a capable GPU (16GB+ VRAM typical with LoRA and 16 frames).
- Reduce `video.max_frames` or use 3B / 4-bit if you hit OOM.
- Always review labels; model quality depends on annotation quality and diversity.
