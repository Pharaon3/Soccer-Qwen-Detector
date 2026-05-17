"""Shared constants for soccer event detection (subnet44-style labels)."""

from __future__ import annotations

from typing import Any


# Allowed event classes only — do not use SoccerNet or other label sets.
ALLOWED_EVENT_CLASSES: tuple[str, ...] = (
    "pass",
    "pass_received",
    "recovery",
    "tackle",
    "interception",
    "ball_out_of_play",
    "clearance",
    "take_on",
    "substitution",
    "block",
    "aerial_duel",
    "shot",
    "save",
    "foul",
    "goal",
)

ALLOWED_SET: frozenset[str] = frozenset(ALLOWED_EVENT_CLASSES)

NUM_EVENT_CLASSES: int = len(ALLOWED_EVENT_CLASSES)
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(ALLOWED_EVENT_CLASSES)}
IDX_TO_CLASS: dict[int, str] = {i: c for i, c in enumerate(ALLOWED_EVENT_CLASSES)}

# Default merge gaps (seconds) if config omits a key
DEFAULT_MIN_GAP_SEC: dict[str, float] = {
    "pass": 0.8,
    "pass_received": 0.8,
    "recovery": 1.0,
    "tackle": 1.2,
    "interception": 1.0,
    "ball_out_of_play": 3.0,
    "clearance": 1.0,
    "take_on": 1.2,
    "substitution": 5.0,
    "block": 1.0,
    "aerial_duel": 1.0,
    "shot": 2.0,
    "save": 2.0,
    "foul": 2.0,
    "goal": 5.0,
}


def normalize_class_key(d: dict[str, Any]) -> dict[str, Any]:
    """Normalize alternate keys to ``class``."""
    if "class" in d:
        return d
    if "class_" in d:
        out = {**d}
        out["class"] = out.pop("class_")
        return out
    return d
