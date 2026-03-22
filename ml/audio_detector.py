#!/usr/bin/env python3
"""
Audio Event Detector for Mobile Insights.

Extracts audio from gameplay videos and uses OpenAI Whisper to detect
MLBB announcer events (kills, objectives, turrets, etc.).
Falls back to RMS energy analysis when Whisper is unavailable.
"""
import os
import re
import subprocess
import sys
import tempfile
import wave

import numpy as np

# Whisper is optional — gracefully degrade if not installed
try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


# MLBB announcer patterns → event type + importance weight
# Patterns are intentionally flexible to account for Whisper transcription
# variations (punctuation, accents, background noise from game SFX).
_EVENT_PATTERNS: list[tuple[str, str, float]] = [
    # Multi-kills (highest value)
    (r"\bsavage\b", "savage", 100.0),
    (r"\bmaniac\b", "maniac", 95.0),
    (r"\btriple\s*kill\b", "triple_kill", 90.0),
    (r"\btriple\b", "triple_kill", 87.0),
    (r"\bdouble\s*kill\b", "double_kill", 85.0),
    (r"\bdouble\b", "double_kill", 82.0),
    (r"\bfirst\s*blood\b", "first_blood", 80.0),
    # Kill streaks
    (r"\blegendary\b", "legendary", 85.0),
    (r"\bgodlike\b", "godlike", 80.0),
    (r"\bkilling\s*spree\b", "killing_spree", 70.0),
    (r"\bspree\b", "killing_spree", 65.0),
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
    debug: bool = False,
) -> list[dict]:
    """Detect MLBB announcer events from video audio using Whisper.

    Extracts the audio track, transcribes it with Whisper, and matches
    known MLBB announcer phrases to classify game events.

    Args:
        video_path: Path to the video file.
        whisper_model: Whisper model size (tiny, base, small, medium, large).
        ffmpeg_timeout: Maximum seconds to wait for ffmpeg.
        debug: When True, print Whisper segments to stderr for inspection.

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
        segments = _transcribe_audio(audio_path, whisper_model)

        if debug:
            for seg in segments:
                print(
                    f"[debug] Whisper segment [{seg.get('start', 0.0):.2f}s]: "
                    f"{seg.get('text', '').strip()!r}",
                    file=sys.stderr,
                )

        events: list[dict] = []
        for segment in segments:
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


def detect_volume_events(
    video_path: str,
    window_sec: float = 1.0,
    hop_sec: float = 0.5,
    rms_threshold: float = 0.05,
    cooldown_sec: float = 2.0,
    ffmpeg_timeout: int = 300,
) -> list[dict]:
    """Detect loud game moments via RMS energy analysis.

    Acts as a fallback signal that fires on any loud audio spike —
    catches events that Whisper may miss (music stingers, SFX bursts).
    Works regardless of Whisper availability.

    Args:
        video_path: Path to the video file.
        window_sec: Analysis window size in seconds.
        hop_sec: Step size between consecutive windows in seconds.
        rms_threshold: Minimum normalised RMS (0–1) to flag as an event.
        cooldown_sec: Minimum seconds between consecutive events.
        ffmpeg_timeout: Maximum seconds to wait for ffmpeg.

    Returns:
        List of dicts with 'timestamp', 'score', 'type', and 'rms'.
        Returns empty list if audio extraction fails.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    audio_path = _extract_audio(video_path, timeout=ffmpeg_timeout)
    if audio_path is None:
        return []

    try:
        return _analyze_rms(audio_path, window_sec, hop_sec, rms_threshold, cooldown_sec)
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)


def _transcribe_audio(audio_path: str, model_name: str) -> list[dict]:
    """Transcribe audio using Whisper and return segments.

    Args:
        audio_path: Path to the WAV audio file.
        model_name: Whisper model size to use.

    Returns:
        List of segment dicts with 'start' and 'text' keys.
    """
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, language="en", word_timestamps=False)
    return result.get("segments", [])


def _analyze_rms(
    audio_path: str,
    window_sec: float,
    hop_sec: float,
    rms_threshold: float,
    cooldown_sec: float,
) -> list[dict]:
    """Compute windowed RMS energy and return timestamps of loud spikes.

    Args:
        audio_path: Path to a 16 kHz mono PCM WAV file.
        window_sec: Window length in seconds.
        hop_sec: Hop size in seconds.
        rms_threshold: Minimum normalised RMS to flag.
        cooldown_sec: Minimum gap between flagged events.

    Returns:
        List of event dicts.
    """
    try:
        with wave.open(audio_path, "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except Exception as exc:
        print(f"[warn] Could not read WAV file for RMS analysis: {exc}", file=sys.stderr)
        return []

    # Convert raw bytes to normalised float32
    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        print(
            f"[warn] Unsupported WAV sample width {sampwidth} bytes; skipping volume analysis.",
            file=sys.stderr,
        )
        return []

    # Mix down to mono
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    window_size = max(1, int(window_sec * sample_rate))
    hop_size = max(1, int(hop_sec * sample_rate))

    events: list[dict] = []
    last_event_time = -cooldown_sec

    for i in range(0, len(samples) - window_size + 1, hop_size):
        window = samples[i : i + window_size]
        rms = float(np.sqrt(np.mean(window**2)))
        timestamp = round(i / sample_rate, 2)

        if rms >= rms_threshold and (timestamp - last_event_time) >= cooldown_sec:
            events.append({
                "timestamp": timestamp,
                "score": round(rms * 100.0, 2),
                "type": "loud_moment",
                "rms": round(rms, 4),
            })
            last_event_time = timestamp

    return events


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
