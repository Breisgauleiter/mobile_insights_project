#!/usr/bin/env python3
"""
Gold Value Extractor for Mobile Insights.

Analyzes MLBB (Mobile Legends: Bang Bang) gameplay video using OCR to
extract gold values for both teams over time, providing a gold timeline
for the highlight pipeline.
"""
import argparse
import json
import os
import re
import sys
from typing import Optional

import cv2
import numpy as np

# Gold display occupies the very top of the screen
_GOLD_Y_START = 0.00
_GOLD_Y_END   = 0.08
_TEAM1_X_START = 0.15
_TEAM1_X_END   = 0.40
_TEAM2_X_START = 0.60
_TEAM2_X_END   = 0.85

# OCR sampling rate: gold values change slowly so 1 FPS is sufficient
_DEFAULT_SAMPLE_FPS = 1.0

# Regex to extract the first numeric token (digits, commas, dots)
_GOLD_REGEX = re.compile(r'(\d[\d,\.]*)')

# Tesseract OCR configuration for single-line gold value extraction
_TESSERACT_CONFIG = '--psm 7 --oem 1'


def _crop_gold_region(
    frame: np.ndarray,
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
) -> np.ndarray:
    """Crop a gold-value region from a video frame using fractional coordinates.

    Args:
        frame: BGR video frame as a NumPy array (H × W × 3).
        x_start: Left boundary as a fraction of frame width (0..1).
        x_end: Right boundary as a fraction of frame width (0..1).
        y_start: Top boundary as a fraction of frame height (0..1).
        y_end: Bottom boundary as a fraction of frame height (0..1).

    Returns:
        Cropped BGR sub-image as a NumPy array.
    """
    h, w = frame.shape[:2]
    x1 = int(w * x_start)
    x2 = int(w * x_end)
    y1 = int(h * y_start)
    y2 = int(h * y_end)
    return frame[y1:y2, x1:x2]


def _preprocess_for_ocr(region: np.ndarray) -> np.ndarray:
    """Preprocess a gold region crop for OCR accuracy.

    Upscales the image 2×, converts to grayscale, then applies a binary
    threshold to isolate the white digits common in MLBB gold displays.

    Args:
        region: BGR crop of the gold display area.

    Returns:
        Preprocessed single-channel binary image as a NumPy array.
    """
    h, w = region.shape[:2]
    upscaled = cv2.resize(region, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    return binary


def _parse_gold_value(text: str) -> Optional[int]:
    """Parse a gold value from raw OCR text output.

    Strips non-printable and non-ASCII characters, collapses whitespace,
    then uses a regex to extract the first numeric token.  Values above
    100 000 are rejected as OCR noise.

    Args:
        text: Raw OCR output string to parse.

    Returns:
        Integer gold value, or ``None`` if no valid value is found.
    """
    # Remove non-printable / non-ASCII characters entirely
    text = re.sub(r'[^\x20-\x7E]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    match = _GOLD_REGEX.search(text)
    if not match:
        return None

    raw = match.group(1).replace(',', '').replace('.', '')
    try:
        value = int(raw)
    except ValueError:
        return None

    # Sanity check: gold values above 100 000 are almost certainly OCR errors
    if value > 100_000:
        return None

    return value


def extract_gold_timeline(
    video_path: str,
    sample_fps: float = _DEFAULT_SAMPLE_FPS,
    max_samples: int = 3600,
) -> list[dict]:
    """Extract gold values for both teams at regular intervals.

    Samples video frames at ``sample_fps`` rate, crops the top-center
    gold display for each team, preprocesses for OCR, and parses the
    numeric gold value.  Only samples where at least one gold value was
    detected are included in the result.

    Args:
        video_path: Path to the input video file.
        sample_fps: Frames per second to sample for OCR (default: 1.0).
            Must be greater than zero.
        max_samples: Maximum number of timeline entries to return
            (default: 3600).  Must be greater than zero.

    Returns:
        List of dicts ordered by time, each containing:

        - ``time`` (float): Timestamp of the sample in seconds.
        - ``team1_gold`` (int, optional): Gold value of team 1 (omitted
          when OCR failed for this team).
        - ``team2_gold`` (int, optional): Gold value of team 2 (omitted
          when OCR failed for this team).
        - ``gold_diff`` (int): ``team1_gold`` minus ``team2_gold``
          (missing values treated as 0).

    Raises:
        ImportError: If ``pytesseract`` is not installed.
        FileNotFoundError: If ``video_path`` does not exist on disk.
        RuntimeError: If the video cannot be opened, reports invalid FPS,
            or the Tesseract binary is not found.
        ValueError: If ``sample_fps`` is not positive or ``max_samples``
            is not positive.
    """
    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be greater than zero, got {sample_fps}")
    if max_samples <= 0:
        raise ValueError(f"max_samples must be greater than zero, got {max_samples}")

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

    entries: list[dict] = []
    frame_idx = 0
    _tesseract_error_logged = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            current_time = frame_idx / fps
            crop1 = _crop_gold_region(
                frame, _TEAM1_X_START, _TEAM1_X_END, _GOLD_Y_START, _GOLD_Y_END
            )
            crop2 = _crop_gold_region(
                frame, _TEAM2_X_START, _TEAM2_X_END, _GOLD_Y_START, _GOLD_Y_END
            )
            processed1 = _preprocess_for_ocr(crop1)
            processed2 = _preprocess_for_ocr(crop2)

            try:
                text1: str = _pytesseract.image_to_string(
                    processed1, config=_TESSERACT_CONFIG
                )
                text2: str = _pytesseract.image_to_string(
                    processed2, config=_TESSERACT_CONFIG
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
                    print(
                        f"[warn] OCR error on frame {frame_idx}: {exc}",
                        file=sys.stderr,
                    )
                    _tesseract_error_logged = True
                frame_idx += 1
                continue

            v1 = _parse_gold_value(text1)
            v2 = _parse_gold_value(text2)

            if v1 is not None or v2 is not None:
                entry: dict = {
                    "time": round(current_time, 2),
                    "gold_diff": (v1 or 0) - (v2 or 0),
                }
                if v1 is not None:
                    entry["team1_gold"] = v1
                if v2 is not None:
                    entry["team2_gold"] = v2
                entries.append(entry)

                if len(entries) >= max_samples:
                    cap.release()
                    return entries

        frame_idx += 1

    cap.release()
    return entries


def main() -> None:
    """CLI entry point for the gold timeline extractor."""
    parser = argparse.ArgumentParser(description="MLBB Gold Value Extractor")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=_DEFAULT_SAMPLE_FPS,
        help=f"Frames per second to sample for OCR (default: {_DEFAULT_SAMPLE_FPS})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )
    args = parser.parse_args()

    try:
        timeline = extract_gold_timeline(
            args.video,
            sample_fps=args.sample_fps,
        )
    except ImportError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    result = json.dumps({"gold_timeline": timeline}, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result + "\n")
    else:
        print(result)


if __name__ == "__main__":
    main()
