#!/usr/bin/env python3
"""
Highlight Detector for Mobile Insights MVP.

Analyzes video files using frame-difference analysis to detect
scenes with high visual activity as potential highlights.
"""
import argparse
import json
import os

import cv2
import numpy as np


def detect_highlights(
    video_path: str,
    threshold: float = 15.0,
    cooldown_sec: float = 2.0,
    max_highlights: int = 10,
    skip_frames: int = 0,
    max_width: int = 0,
) -> list[dict]:
    """Detect highlights in a video using frame-difference analysis.

    Compares consecutive grayscale frames to find moments of high
    visual activity (e.g., fast movement, scene changes).

    Args:
        video_path: Path to the video file.
        threshold: Minimum mean pixel difference to consider a highlight.
        cooldown_sec: Minimum seconds between two highlights.
        max_highlights: Maximum number of highlights to return.
        skip_frames: Analyze every Nth frame (0 = no skipping).
        max_width: Downscale frames to this width in pixels (0 = no scaling).

    Returns:
        List of dicts with 'timestamp' (float, seconds) and 'score' (float).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError("Video has invalid FPS")

    prev_gray = None
    candidates: list[dict] = []
    frame_idx = 0

    step = max(1, skip_frames) if skip_frames > 0 else 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames for performance (still count them for correct timestamps)
        if step > 1 and frame_idx % step != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Downscale for performance
        if max_width > 0 and gray.shape[1] > max_width:
            scale = max_width / gray.shape[1]
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        if prev_gray is not None and gray.shape == prev_gray.shape:
            diff = cv2.absdiff(prev_gray, gray)
            score = float(np.mean(diff))

            if score >= threshold:
                timestamp = round(frame_idx / fps, 2)
                candidates.append({"timestamp": timestamp, "score": round(score, 2)})

        prev_gray = gray

    cap.release()

    # Merge nearby candidates using cooldown period
    highlights = _merge_candidates(candidates, cooldown_sec)

    # Sort by score descending and limit
    highlights.sort(key=lambda h: h["score"], reverse=True)
    return highlights[:max_highlights]


def _merge_candidates(
    candidates: list[dict], cooldown_sec: float
) -> list[dict]:
    """Merge candidates that are within the cooldown period, keeping the highest score."""
    if not candidates:
        return []

    merged: list[dict] = [candidates[0]]
    for c in candidates[1:]:
        if c["timestamp"] - merged[-1]["timestamp"] < cooldown_sec:
            # Keep the one with the higher score
            if c["score"] > merged[-1]["score"]:
                merged[-1] = c
        else:
            merged.append(c)

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Frame-Difference Highlight Detector")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--threshold", type=float, default=15.0, help="Activity threshold (default: 15.0)")
    parser.add_argument("--cooldown", type=float, default=2.0, help="Cooldown between highlights in seconds")
    parser.add_argument("--max", type=int, default=10, help="Maximum number of highlights")
    parser.add_argument("--skip-frames", type=int, default=0, help="Analyze every Nth frame (0 = all)")
    parser.add_argument("--max-width", type=int, default=0, help="Downscale frames to this width (0 = original)")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: stdout)")
    args = parser.parse_args()

    highlights = detect_highlights(
        args.video, args.threshold, args.cooldown, args.max,
        skip_frames=args.skip_frames, max_width=args.max_width,
    )

    if args.format == "json":
        result = json.dumps({"highlights": highlights}, indent=2)
    else:
        lines = [f"Highlight at {h['timestamp']}s (score: {h['score']})" for h in highlights]
        result = "\n".join(lines) if lines else "No highlights detected."

    if args.output:
        with open(args.output, "w") as f:
            f.write(result + "\n")
    else:
        print(result)


if __name__ == "__main__":
    main()
