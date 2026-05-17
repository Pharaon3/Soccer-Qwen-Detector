"""Build prompts for soccer event detection with Qwen2.5-VL (or similar VLMs)."""

from __future__ import annotations

from src.schema import ALLOWED_EVENT_CLASSES


CLASS_DEFINITIONS = """
pass:
A player intentionally sends the ball to a teammate.

pass_received:
A teammate controls or receives a deliberate pass.

recovery:
A player gains possession after neither team clearly had controlled possession.

tackle:
A player challenges an opponent to stop progression or win the ball.

interception:
A player cuts off or blocks an opponent's pass before it reaches the target.

ball_out_of_play:
The ball clearly crosses the sideline or goal line and play stops or restarts.

clearance:
A defending player kicks, heads, or plays the ball away from a dangerous area.

take_on:
An attacking player attempts to dribble past an opponent.

substitution:
A player replacement occurs or is clearly indicated.

block:
A player blocks a shot, cross, or pass with their body or foot.

aerial_duel:
Two or more players contest the ball in the air.

shot:
A player attempts to score by shooting toward goal.

save:
The goalkeeper prevents or attempts to prevent a shot from entering the goal.

foul:
Illegal contact or unfair play occurs and likely stops or affects play.

goal:
The ball enters the goal and a goal is clearly scored.
""".strip()


def build_user_prompt(
    video_id: str,
    frame_descriptions: list[tuple[int, float]],
    duration_sec: float,
) -> str:
    """
    Build the user prompt listing frames in temporal order with index and time.

    ``frame_descriptions`` is a list of (frame_index, timestamp_sec).
    """
    lines = [
        f'video_id: "{video_id}"',
        f"The clip is about {duration_sec:.1f} seconds. Frames are in strict temporal order.",
        "Attached images correspond to the following (frame_index, timestamp_sec):",
    ]
    for fi, ts in frame_descriptions:
        lines.append(f"  - frame_index={fi}, timestamp_sec={ts:.3f}")
    lines.extend(
        [
            "",
            "TASK:",
            "Analyze the frames as a short soccer video sequence.",
            "Detect ONLY events whose class is one of the allowed labels below.",
            "Do NOT invent events. If nothing confident is visible, return an empty events list.",
            "Use timestamp_sec based on WHEN the event is visible in this frame sequence "
            "(interpolate between listed timestamps if needed).",
            "confidence must be a number from 0 to 1.",
            "Include a short explanation for each event.",
            "Include start_time_sec and end_time_sec when useful; for instant events they may be equal.",
            "",
            "ALLOWED event classes (exact strings):",
            ", ".join(ALLOWED_EVENT_CLASSES),
            "",
            "CLASS DEFINITIONS:",
            CLASS_DEFINITIONS,
            "",
            "OUTPUT RULES:",
            "- Return ONLY valid JSON. No markdown, no commentary, no code fences.",
            "- Top-level keys: video_id (string), events (array).",
            "- Each event object must include:",
            '  "class", "timestamp_sec", "start_time_sec", "end_time_sec", '
            '"confidence", "explanation"',
            "- If no event is visible: {\"video_id\": <same as input>, \"events\": []}",
            "",
            "Example shape (values illustrative):",
            '{ "video_id": "clip_001", "events": [ { "class": "pass", "timestamp_sec": 12.4, '
            '"start_time_sec": 12.2, "end_time_sec": 12.6, "confidence": 0.82, '
            '"explanation": "Player in blue intentionally passes the ball to a teammate." } ] }',
        ]
    )
    return "\n".join(lines)


def build_system_prompt() -> str:
    return (
        "You are an expert soccer video analyst. "
        "You receive temporally ordered video frames as images. "
        "You must follow the user's instructions exactly and output JSON only when asked."
    )
