# Soccer Qwen VL Event Detector

Inference-only pipeline for **30-second soccer clips** (e.g. 25 FPS, 750 frames). It samples frames, sends them to **Qwen2.5-VL-72B** (or another model) via an **OpenAI-compatible** vision API, and writes **structured JSON** event predictions using a fixed label set (subnet44-style; **not** SoccerNet).

## Install

```bash
cd soccer_qwen_detector
python -m venv .venv
```

Activate the venv (Windows PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configure the Qwen endpoint

Edit `config.yaml`:

- `qwen.base_url` — OpenAI-compatible base URL, e.g. `http://localhost:8000/v1`
- `qwen.api_key` — API key or placeholder such as `EMPTY` for local servers
- `qwen.model_name` — model id served by the endpoint, e.g. `Qwen/Qwen2.5-VL-72B-Instruct`

Video sampling (defaults match a 30 s, 25 FPS clip):

- `video.fps`, `video.clip_seconds`, `video.sample_fps`, `video.max_frames` — default sampling is about **2 FPS** (≈60 frames over 30 s), capped by `max_frames`.

Postprocessing:

- `postprocess.output_fps` — used for `frame_index = round(timestamp_sec * output_fps)` (default **25**)
- `postprocess.min_gap_sec` — per-class minimum time gap for merging near-duplicate detections

## Run: single video

From the `soccer_qwen_detector` directory (so `src` imports resolve):

```bash
python run_inference.py --video path/to/video.mp4 --config config.yaml --output outputs/clip_001.json
```

This writes:

- `outputs/clip_001.json` — validated, postprocessed predictions
- `outputs/clip_001_raw.txt` — raw model output for debugging

## Run: folder of videos

```bash
python run_inference.py --video_dir data/videos --config config.yaml --output_dir outputs/
```

Each input `stem` produces `outputs/{stem}.json` and `outputs/{stem}_raw.txt`.

## Output shape

Final JSON includes `video_id`, `fps`, `duration_sec`, and `events` with `frame_index` and clamped times. Allowed event classes are fixed in code (`src/schema.py`).

## Limitations

- **Vision-language models hallucinate.** Qwen2.5-VL may report events that did not occur or miss subtle ones.
- Outputs are **not** reliable as exact per-frame ground truth without human review or extra constraints.
- Best uses: **pseudo-labeling**, **annotation assistance**, **explanation**, or as one signal in a larger system.
- A production detector should combine this with **tracking**, a **temporal classifier**, calibration, and **threshold tuning**.

## Dataset layout (`dataset/`)

Clip folders (same layout as `dataset/annotator` in the browser tool):

- `dataset/<clip_id>/<clip_id>.json` (or any `*.json` in that folder) with:
  - `events`: list of `{ "time_sec": number, "label": string }` using the same classes as `src/schema.py`
  - optional `environment`: one of `child`, `night`, `snow`, `dry` (ignored by training for now)
- A **video file** next to the JSON (`.mp4`, `.webm`, `.mkv`, `.mov`, `.avi`, `.m4v`)

The `annotator/` subfolder is skipped when scanning for clips.

## Train (supervised, local CNN)

Trains a **ResNet18 + temporal conv** model on **30 temporal segments** per clip (multi-label BCE per segment and class). This is separate from Qwen inference: it learns from your JSON labels once videos are present.

1. Install deps (includes PyTorch). For GPU builds, use the official PyTorch install command for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).
2. Edit `train_config.yaml` (`dataset_root`, `epochs`, `batch_size`, `max_clips` for quick tests, etc.).
3. Run:

```bash
python run_train.py --config train_config.yaml
```

Checkpoints are written under `checkpoints/` (`last.pt` and periodic `epoch_XXX.pt`). If no clip folder contains a video, training exits with a hint; for **debugging only** you may set `mock_video_if_missing: true` (black frames — not meaningful learning).

## Project layout

- `run_inference.py` — Qwen / OpenAI-compatible VL inference CLI
- `run_train.py` — supervised training on `dataset/`
- `config.yaml` — VL inference settings
- `train_config.yaml` — training settings
- `src/video_loader.py` — OpenCV read + temporal sampling + PIL images
- `src/qwen_vl_client.py` — OpenAI-compatible multimodal chat client (swap for another VLM later)
- `src/prompt_builder.py` — soccer event prompt + class definitions
- `src/postprocess.py` — class validation, time clamping, frame index, merge by `min_gap_sec`
- `src/schema.py` — allowed classes, indices, default merge gaps
- `src/utils.py` — JPEG base64 + robust JSON extraction
- `src/dataset_soccer.py` — scan clip folders, read frames, build segment targets
- `src/model_temporal.py` — `SegmentEventModel`
- `src/train_utils.py` — seed, pos-weight estimate, checkpoint save
