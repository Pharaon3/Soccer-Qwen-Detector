"""General helpers: image encoding, JSON extraction from model text."""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any

from PIL import Image


def pil_to_jpeg_base64(image: Image.Image, quality: int = 85) -> str:
    """Encode a PIL image as base64 JPEG string (no data URL prefix)."""
    buf = io.BytesIO()
    rgb = image.convert("RGB")
    rgb.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def data_url_from_jpeg_base64(b64: str) -> str:
    return f"data:image/jpeg;base64,{b64}"


def extract_json_object(text: str) -> str | None:
    """
    Try to isolate a JSON object from model output that may include markdown
    fences or leading/trailing commentary.
    """
    if not text:
        return None
    s = text.strip()

    # Code block ```json ... ```
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
    if fence:
        inner = fence.group(1).strip()
        if inner.startswith("{") and inner.endswith("}"):
            return inner

    # First '{' to last '}' greedy balance-ish: simple scan
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def parse_model_json(text: str) -> dict[str, Any]:
    """Parse JSON from model text; raises json.JSONDecodeError if unrecoverable."""
    candidate = extract_json_object(text)
    if candidate is None:
        raise json.JSONDecodeError("No JSON object found in model output", text, 0)
    return json.loads(candidate)
