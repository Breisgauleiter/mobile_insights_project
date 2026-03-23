#!/usr/bin/env python3
"""
Game-Timer Stats Extractor for Mobile Insights.

Reads the MLBB game timer from the top-center HUD area to build a mapping
between video timestamps and in-game time.  Enables correlating events
(highlights, kills, stats) to game phases (early / mid / late game).
"""
import argparse
import json
import os
import re
import sys
from typing import Optional

import cv2
import numpy as np

# Timer is displayed at the top-center of the MLBB HUD
_TIMER_X_START = 0.38
_TIMER_X_END   = 0.62
_TIMER_Y_START = 0.00
_TIMER_Y_END   = 0.08

# Sample every 5 seconds — the timer is predictable between samples
_DEFAULT_SAMPLE_INTERVAL = 5.0

# Game-phase thresholds in seconds
_EARLY_GAME_END = 5 * 60    # 0:00 – 5:00  → early
_MID_GAME_END   = 15 * 60   # 5:01 – 15:00 → mid
                              # > 15:00      → late


def _crop_timer_region(frame: np.ndarray) -> np.ndarray:
    """Crop the game-timer region from a video frame.

    Extracts the top-center portion of the frame (x: 38 %–62 %, y: 0 %–8 %)
    where the MLBB game timer is displayed in MM:SS format.

    Args:
        frame: BGR video frame as a NumPy array (H × W × 3).

    Returns:
        Cropped BGR sub-image as a NumPy array.
    """
    h, w = frame.shape[:2]
    x_start = int(w * _TIMER_X_START)
    x_end   = int(w * _TIMER_X_END)
    y_start = int(h * _TIMER_Y_START)
    y_end   = int(h * _TIMER_Y_END)
    return frame[y_start:y_end, x_start:x_end]


def _preprocess_for_ocr(region: np.ndarray) -> np.ndarray:
    """Preprocess a timer crop for OCR accuracy.

    Upscales the image 2×, converts to grayscale, then applies a binary
    threshold to isolate the white timer text on the semi-transparent HUD.

    Args:
        region: BGR crop of the timer area.

    Returns:
        Preprocessed single-channel binary image as a NumPy array.
    """
    h, w = region.shape[:2]
    upscaled = cv2.resize(region, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    return binary


def _parse_timer_text(text: str) -> Optional[int]:
    """Parse a MM:SS or M:SS game-timer string from OCR output.

    Scans the raw OCR text for the first ``MM:SS`` pattern and converts it
    to a total number of seconds.  Rejects values where seconds ≥ 60.

    Args:
        text: Raw OCR output string that may contain the game timer.

    Returns:
        Total in-game seconds as an integer if a valid timer pattern is found,
        or ``None`` if no valid timer is detected.
    """
    match = re.search(r'\b(\d{1,2}):(\d{2})\b', text)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        if seconds < 60:
            return minutes * 60 + seconds
    return None


def _seconds_to_mmss(total_seconds: int) -> str:
    """Format total seconds as a MM:SS string.

    Args:
        total_seconds: Non-negative integer number of seconds.

    Returns:
        String in ``MM:SS`` format, e.g. ``"02:47"``.
    """
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def _game_phase(game_seconds: int) -> str:
    """Determine the game phase for a given in-game timestamp.

    Args:
        game_seconds: In-game time in seconds.

    Returns:
        One of ``"early"``, ``"mid"``, or ``"late"``.
    """
    if game_seconds <= _EARLY_GAME_END:
        return "early"
    if game_seconds <= _MID_GAME_END:
        return "mid"
    return "late"


def extract_timer_mapping(
    video_path: str,
    sample_interval: float = _DEFAULT_SAMPLE_INTERVAL,
) -> dict:
    """Extract a video-time → game-time mapping from an MLBB recording.

    Samples the video at ``sample_interval`` second intervals, runs OCR on
    the top-center timer region, and builds an ordered list of
    ``{video_time, game_time, game_seconds, phase}`` entries.

    Additionally detects:

    - **game_start**: The video timestamp at which the first valid timer
      reading appears (loading screen → in-game transition).
    - **game_end**: The video timestamp of the last valid timer reading
      (before the victory/defeat screen replaces the HUD).

    Args:
        video_path: Path to the input video file.
        sample_interval: Seconds between OCR samples (default: 5.0).
            Must be greater than zero.

    Returns:
        A dict with the following keys:

        - ``"timeline"`` (list[dict]): Ordered list of mapping entries,
          each containing ``"video_time"`` (float), ``"game_time"`` (str),
          ``"game_seconds"`` (int), and ``"phase"`` (str).
        - ``"game_start_video_time"`` (float | None): Video timestamp of
          the first valid timer reading, or ``None`` if never detected.
        - ``"game_end_video_time"`` (float | None): Video timestamp of
          the last valid timer reading, or ``None`` if never detected.

    Raises:
        ImportError: If ``pytesseract`` is not installed.
        FileNotFoundError: If ``video_path`` does not exist on disk.
        RuntimeError: If the video cannot be opened or has invalid FPS.
        ValueError: If ``sample_interval`` is not positive.
    """
    if sample_interval <= 0:
        raise ValueError(
            f"sample_interval must be greater than zero, got {sample_interval}"
        )

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

    # Number of source frames between OCR samples
    frame_step = max(1, round(fps * sample_interval))

    timeline: list[dict] = []
    _ocr_error_logged = False
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            video_time = frame_idx / fps
            crop = _crop_timer_region(frame)
            processed = _preprocess_for_ocr(crop)

            try:
                ocr_text: str = _pytesseract.image_to_string(
                    processed, config='--psm 7 --oem 1'
                )
            except _pytesseract.TesseractNotFoundError as exc:
                cap.release()
                raise RuntimeError(
                    "Tesseract binary not found. "
                    "Install it from https://github.com/tesseract-ocr/tesseract"
                ) from exc
            except Exception as exc:
                if not _ocr_error_logged:
                    print(
                        f"[warn] OCR error on frame {frame_idx}: {exc}",
                        file=sys.stderr,
                    )
                    _ocr_error_logged = True
                frame_idx += 1
                continue

            game_seconds = _parse_timer_text(ocr_text)
            if game_seconds is not None:
                timeline.append({
                    "video_time": round(video_time, 2),
                    "game_time": _seconds_to_mmss(game_seconds),
                    "game_seconds": game_seconds,
                    "phase": _game_phase(game_seconds),
                })

        frame_idx += 1

    cap.release()

    game_start = timeline[0]["video_time"] if timeline else None
    game_end   = timeline[-1]["video_time"] if timeline else None

    return {
        "timeline": timeline,
        "game_start_video_time": game_start,
        "game_end_video_time": game_end,
    }


def get_game_time_at(timeline: list[dict], video_time: float) -> Optional[int]:
    """Look up the approximate in-game time for a given video timestamp.

    Uses linear interpolation between the two nearest timeline entries to
    estimate the in-game seconds for any video timestamp within the mapped
    range.  Returns ``None`` for timestamps outside the mapped range.

    Args:
        timeline: Ordered list of timeline dicts as returned by
            :func:`extract_timer_mapping` under the ``"timeline"`` key.
        video_time: The video timestamp in seconds to look up.

    Returns:
        Estimated in-game time in seconds (int), or ``None`` if the
        timestamp is outside the recorded timeline.
    """
    if not timeline:
        return None

    if video_time < timeline[0]["video_time"] or video_time > timeline[-1]["video_time"]:
        return None

    for i in range(len(timeline) - 1):
        left  = timeline[i]
        right = timeline[i + 1]
        if left["video_time"] <= video_time <= right["video_time"]:
            span_video = right["video_time"] - left["video_time"]
            if span_video == 0:
                return left["game_seconds"]
            ratio = (video_time - left["video_time"]) / span_video
            interp = left["game_seconds"] + ratio * (
                right["game_seconds"] - left["game_seconds"]
            )
            return round(interp)

    # Exact match with the last entry
    return timeline[-1]["game_seconds"]


def main() -> None:
    """CLI entry point for the game-timer extractor."""
    parser = argparse.ArgumentParser(description="MLBB Game-Timer OCR Extractor")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=_DEFAULT_SAMPLE_INTERVAL,
        help=f"Seconds between OCR samples (default: {_DEFAULT_SAMPLE_INTERVAL})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )
    args = parser.parse_args()

    try:
        result = extract_timer_mapping(
            args.video,
            sample_interval=args.sample_interval,
        )
    except ImportError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    output = json.dumps(result, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
