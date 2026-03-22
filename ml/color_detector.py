#!/usr/bin/env python3
"""
Color-Burst Detector for Mobile Insights.

Analyzes color histogram changes between frames to detect visually
significant moments like explosions, skill effects, and team fights.
"""
import os

import cv2
import numpy as np


# HSV color ranges for MLBB-relevant effects
_COLOR_RANGES = {
    "red": ((0, 80, 80), (10, 255, 255)),       # Kill effects, damage
    "red2": ((170, 80, 80), (180, 255, 255)),    # Red wraparound in HSV
    "gold": ((15, 100, 100), (35, 255, 255)),    # Gold/yellow skill effects
    "blue": ((100, 80, 80), (130, 255, 255)),    # Blue team skills, mana
    "purple": ((130, 60, 60), (160, 255, 255)),  # Purple effects (ults)
}


def detect_color_bursts(
    video_path: str,
    burst_threshold: float = 4.0,
    cooldown_sec: float = 2.0,
    max_events: int = 20,
    skip_frames: int = 0,
    max_width: int = 0,
) -> list[dict]:
    """Detect color burst events in a video by analyzing HSV histogram changes.

    Compares color channel pixel counts between consecutive frames and flags
    moments where specific colors spike significantly (indicating effects,
    kills, skills, or explosions).

    Args:
        video_path: Path to the video file.
        burst_threshold: Minimum ratio of color increase to flag an event.
        cooldown_sec: Minimum seconds between events.
        max_events: Maximum number of events to return.
        skip_frames: Analyze every Nth frame (0 = no skipping).
        max_width: Downscale frames to this width (0 = no scaling).

    Returns:
        List of dicts with 'timestamp', 'score', and 'dominant_color'.
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

    step = max(1, skip_frames) if skip_frames > 0 else 1
    prev_counts: dict[str, float] | None = None
    candidates: list[dict] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if step > 1 and frame_idx % step != 0:
            frame_idx += 1
            continue

        # Downscale for performance
        if max_width > 0 and frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        total_pixels = float(hsv.shape[0] * hsv.shape[1])

        # Count pixels in each color range as fraction of total
        counts: dict[str, float] = {}
        for name, (lower, upper) in _COLOR_RANGES.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            counts[name] = float(np.count_nonzero(mask)) / total_pixels

        if prev_counts is not None:
            best_ratio = 0.0
            best_color = ""
            for name in counts:
                prev_val = prev_counts.get(name, 0.0)
                curr_val = counts[name]
                # Calculate spike ratio (avoid division by zero)
                if prev_val > 0.001:
                    ratio = curr_val / prev_val
                elif curr_val > 0.02:
                    ratio = burst_threshold + 1.0
                else:
                    ratio = 0.0

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_color = name

            if best_ratio >= burst_threshold and best_color:
                timestamp = round(frame_idx / fps, 2)
                # Normalize the color name (merge red + red2)
                color_name = "red" if best_color == "red2" else best_color
                score = round(best_ratio * 10, 2)
                candidates.append({
                    "timestamp": timestamp,
                    "score": score,
                    "dominant_color": color_name,
                })

        prev_counts = counts
        frame_idx += 1

    cap.release()

    # Merge nearby candidates
    merged = _merge_color_candidates(candidates, cooldown_sec)
    merged.sort(key=lambda h: h["score"], reverse=True)
    return merged[:max_events]


def _merge_color_candidates(
    candidates: list[dict], cooldown_sec: float
) -> list[dict]:
    """Merge color-burst candidates within the cooldown window."""
    if not candidates:
        return []

    merged: list[dict] = [candidates[0]]
    for c in candidates[1:]:
        if c["timestamp"] - merged[-1]["timestamp"] < cooldown_sec:
            if c["score"] > merged[-1]["score"]:
                merged[-1] = c
        else:
            merged.append(c)

    return merged
