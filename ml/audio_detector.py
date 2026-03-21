#!/usr/bin/env python3
"""
Audio Event Detector for Mobile Insights.

Extracts audio from gameplay videos and uses OpenAI Whisper to detect
MLBB announcer events (kills, objectives, turrets, etc.).
"""
import os
import re
import subprocess
import tempfile

# Whisper is optional — gracefully degrade if not installed
try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


# MLBB announcer patterns → event type + importance weight
_EVENT_PATTERNS: list[tuple[str, str, float]] = [
    # Multi-kills (highest value)
    (r"\bsavage\b", "savage", 100.0),
    (r"\bmaniac\b", "maniac", 95.0),
    (r"\btriple\s*kill\b", "triple_kill", 90.0),
    (r"\bdouble\s*kill\b", "double_kill", 85.0),
    (r"\bfirst\s*blood\b", "first_blood", 80.0),
    # Kill streaks
    (r"\blegendary\b", "legendary", 85.0),
    (r"\bgodlike\b", "godlike", 80.0),
    (r"\bkilling\s*spree\b", "killing_spree", 70.0),
    # Objectives (before generic kill patterns to avoid false matches)
    (r"\blord\b.*\bslain\b", "lord", 85.0),
    (r"\bturtle\b.*\bslain\b", "turtle", 75.0),
    (r"\blord\b", "lord", 70.0),
    (r"\bturtle\b", "turtle", 60.0),
    # Single kills
    (r"\b(?:has been |been |enemy |ally )?slain\b", "kill", 60.0),
    (r"\bexecuted\b", "kill", 55.0),
    (r"\bshutdown\b", "kill", 65.0),
    # Turrets
    (r"\bturret\b.*\bdestroyed\b", "turret", 50.0),
    (r"\btower\b.*\bdestroyed\b", "turret", 50.0),
    (r"\bturret\b", "turret", 40.0),
    # Game events
    (r"\bvictory\b", "game_end", 70.0),
    (r"\bdefeat\b", "game_end", 70.0),
]


def detect_audio_events(
    video_path: str,
    whisper_model: str = "tiny",
    ffmpeg_timeout: int = 300,
) -> list[dict]:
    """Detect MLBB announcer events from video audio using Whisper.

    Extracts the audio track, transcribes it with Whisper, and matches
    known MLBB announcer phrases to classify game events.

    Args:
        video_path: Path to the video file.
        whisper_model: Whisper model size (tiny, base, small, medium, large).

    Returns:
        List of dicts with 'timestamp', 'score', 'type', and 'text'.
        Returns empty list if Whisper is not available or audio extraction fails.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not WHISPER_AVAILABLE:
        return []

    # Extract audio to temporary WAV file
    audio_path = _extract_audio(video_path, timeout=ffmpeg_timeout)
    if audio_path is None:
        return []

    try:
        model = whisper.load_model(whisper_model)
        result = model.transcribe(audio_path, language="en", word_timestamps=False)

        events: list[dict] = []
        for segment in result.get("segments", []):
            text = segment.get("text", "").strip()
            start = segment.get("start", 0.0)

            matched = _classify_text(text)
            if matched:
                event_type, score = matched
                events.append({
                    "timestamp": round(start, 2),
                    "score": score,
                    "type": event_type,
                    "text": text,
                })

        return events
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)


def _extract_audio(video_path: str, timeout: int = 300) -> str | None:
    """Extract audio from video to a temporary WAV file using ffmpeg.

    Args:
        video_path: Path to the video file.
        timeout: Maximum seconds to wait for ffmpeg (default: 300).

    Returns:
        Path to the temporary WAV file, or None if extraction fails.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    try:
        subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                "-y", tmp.name,
            ],
            capture_output=True,
            timeout=timeout,
            check=True,
        )
        return tmp.name
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        return None


def _classify_text(text: str) -> tuple[str, float] | None:
    """Match transcribed text against known MLBB announcer patterns.

    Returns:
        Tuple of (event_type, score) for the highest-priority match, or None.
    """
    lower = text.lower()
    for pattern, event_type, score in _EVENT_PATTERNS:
        if re.search(pattern, lower):
            return event_type, score
    return None
