#!/usr/bin/env python3
"""
Kill-Feed OCR Detector for Mobile Insights.

Analyzes the top-right region of gameplay video frames using OCR
to detect kill events from the kill-feed overlay, providing
high-confidence "kill" timestamps for the highlight pipeline.
"""
import argparse
import json
import os
import re
import sys
from typing import Optional

import cv2
import numpy as np

# Kill feed occupies the top-right corner of the screen
_KILLFEED_X_START = 0.60
_KILLFEED_Y_START = 0.00
_KILLFEED_X_END   = 1.00
_KILLFEED_Y_END   = 0.20

# OCR sampling rate: kill feed entries stay visible ~3 seconds so 1-2 FPS is enough
_DEFAULT_SAMPLE_FPS = 1.5

# Same (killer, victim) pair within this window is treated as one event
_DEDUP_WINDOW_SEC = 3.0

# Minimum OCR text length worth parsing
_MIN_TEXT_LEN = 4


def _crop_killfeed(frame: np.ndarray) -> np.ndarray:
    """Crop the kill-feed region from a video frame.

    Extracts the top-right portion of the frame (x: 60%–100%, y: 0%–20%)
    where the kill feed is typically displayed in MLBB.

    Args:
        frame: BGR video frame as a NumPy array (H × W × 3).

    Returns:
        Cropped BGR sub-image as a NumPy array.
    """
    h, w = frame.shape[:2]
    x_start = int(w * _KILLFEED_X_START)
    x_end   = int(w * _KILLFEED_X_END)
    y_start = int(h * _KILLFEED_Y_START)
    y_end   = int(h * _KILLFEED_Y_END)
    return frame[y_start:y_end, x_start:x_end]


def _preprocess_for_ocr(region: np.ndarray) -> np.ndarray:
    """Preprocess a kill-feed crop for OCR accuracy.

    Upscales the image 2×, converts to grayscale, then applies a binary
    threshold to isolate the white text common in mobile game kill feeds.

    Args:
        region: BGR crop of the kill-feed area.

    Returns:
        Preprocessed single-channel binary image as a NumPy array.
    """
    h, w = region.shape[:2]
    upscaled = cv2.resize(region, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    return binary


def _parse_kill_text(text: str) -> Optional[tuple[str, str, Optional[str]]]:
    """Parse a single OCR text line for a kill event.

    Normalizes whitespace, strips non-printable characters, then applies a
    regex to detect "killer → victim" patterns including common kill-feed
    verbs (``killed``, ``slain``) and arrow notations (``->``, ``→``).

    Args:
        text: Raw OCR output line to parse.

    Returns:
        A ``(killer, victim, assist)`` tuple when a valid kill pattern is
        detected and both names are at least 2 characters long.
        ``assist`` is always ``None`` in the current implementation
        (reserved for future enhancement).
        Returns ``None`` if no pattern is found or names are too short.
    """
    text = text.strip()
    # Remove non-printable / non-ASCII characters
    text = re.sub(r'[^\x20-\x7E]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    if len(text) < _MIN_TEXT_LEN:
        return None

    pattern = (
        r'([A-Za-z][A-Za-z\s\'-]{1,20}?)'
        r'\s+(?:killed|slain|>|->|→)\s+'
        r'([A-Za-z][A-Za-z\s\'-]{1,20})'
    )
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        killer = match.group(1).strip()
        victim = match.group(2).strip()
        if len(killer) >= 2 and len(victim) >= 2:
            return (killer, victim, None)

    return None


def detect_killfeed(
    video_path: str,
    sample_fps: float = _DEFAULT_SAMPLE_FPS,
    dedup_window: float = _DEDUP_WINDOW_SEC,
    max_events: int = 50,
) -> list[dict]:
    """Detect kill events from the kill-feed overlay using OCR.

    Samples video frames at ``sample_fps`` rate, crops the top-right
    kill-feed region, preprocesses for OCR, and parses kill patterns from
    each line of the recognized text.  Duplicate ``(killer, victim)`` pairs
    within ``dedup_window`` seconds are suppressed to avoid noise from a
    kill entry lingering on screen across multiple sampled frames.

    Args:
        video_path: Path to the input video file.
        sample_fps: Frames per second to sample for OCR (default: 1.5).
            Must be greater than zero.
        dedup_window: Seconds to suppress repeated ``(killer, victim)``
            pairs (default: 3.0).
        max_events: Maximum number of kill events to return (default: 50).
            Must be greater than zero.

    Returns:
        List of dicts ordered by appearance, each containing:

        - ``time`` (float): Timestamp of the kill event in seconds.
        - ``killer`` (str): Name of the killing player.
        - ``victim`` (str): Name of the killed player.
        - ``assist`` (str, optional): Assisting player name (omitted when
          not detected).

    Raises:
        ImportError: If ``pytesseract`` is not installed.
        FileNotFoundError: If ``video_path`` does not exist on disk.
        RuntimeError: If the video cannot be opened or reports invalid FPS.
        ValueError: If ``sample_fps`` is not positive or ``max_events`` is not
            positive.
    """
    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be greater than zero, got {sample_fps}")
    if max_events <= 0:
        raise ValueError(f"max_events must be greater than zero, got {max_events}")

    try:
        import pytesseract as _pytesseract
    except ImportError as exc:
        raise ImportError(
            "pytesseract is not installed. "
            "Install it with: pip install pytesseract  "
            "You may also need to install the Tesseract OCR binary: "
            "https://github.com/tesseract-ocr/tesseract"
        ) from exc

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError("Video has invalid FPS")

    # Number of source frames to skip between OCR samples
    frame_step = max(1, round(fps / sample_fps))

    events: list[dict] = []
    # Track last-seen timestamp for each (killer, victim) pair
    recent: dict[tuple[str, str], float] = {}
    frame_idx = 0
    _tesseract_error_logged = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            current_time = frame_idx / fps
            crop = _crop_killfeed(frame)
            processed = _preprocess_for_ocr(crop)

            try:
                ocr_text: str = _pytesseract.image_to_string(
                    processed, config='--psm 6 --oem 1'
                )
            except _pytesseract.TesseractNotFoundError as exc:
                # Tesseract binary is missing — no point continuing
                cap.release()
                raise RuntimeError(
                    "Tesseract binary not found. "
                    "Install it from https://github.com/tesseract-ocr/tesseract"
                ) from exc
            except Exception as exc:
                if not _tesseract_error_logged:
                    print(f"[warn] OCR error on frame {frame_idx}: {exc}", file=sys.stderr)
                    _tesseract_error_logged = True
                frame_idx += 1
                continue

            for line in ocr_text.splitlines():
                parsed = _parse_kill_text(line)
                if parsed is None:
                    continue

                killer, victim, assist = parsed
                key = (killer, victim)

                # Suppress duplicate entries still visible on screen
                last_seen = recent.get(key)
                if last_seen is not None and current_time - last_seen < dedup_window:
                    continue

                recent[key] = current_time
                event: dict = {
                    "time": round(current_time, 2),
                    "killer": killer,
                    "victim": victim,
                }
                if assist is not None:
                    event["assist"] = assist

                events.append(event)

                if len(events) >= max_events:
                    cap.release()
                    return events

        frame_idx += 1

    cap.release()
    return events


def main() -> None:
    """CLI entry point for the kill-feed OCR detector."""
    parser = argparse.ArgumentParser(description="Kill-Feed OCR Detector")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=_DEFAULT_SAMPLE_FPS,
        help=f"Frames per second to sample for OCR (default: {_DEFAULT_SAMPLE_FPS})",
    )
    parser.add_argument(
        "--dedup-window",
        type=float,
        default=_DEDUP_WINDOW_SEC,
        help=f"Seconds to suppress duplicate kills (default: {_DEDUP_WINDOW_SEC})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )
    args = parser.parse_args()

    try:
        kills = detect_killfeed(
            args.video,
            sample_fps=args.sample_fps,
            dedup_window=args.dedup_window,
        )
    except ImportError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    result = json.dumps({"kills": kills}, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result + "\n")
    else:
        print(result)


if __name__ == "__main__":
    main()
