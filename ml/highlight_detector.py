#!/usr/bin/env python3
"""
Highlight Detector for Mobile Insights.

Multi-signal pipeline that combines frame-difference analysis,
color-burst detection, and audio event recognition (Whisper)
to detect gameplay highlights with event-type classification.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

from audio_detector import detect_audio_events
from color_detector import detect_color_bursts


def detect_frame_diff(
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
        List of dicts with 'timestamp', 'score', 'type', and 'sources'.
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

        # Skip frames for performance (still count them for correct timestamps)
        if step > 1 and frame_idx % step != 0:
            frame_idx += 1
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
                candidates.append({
                    "timestamp": timestamp,
                    "score": round(score, 2),
                    "type": "action",
                    "sources": ["frame_diff"],
                })

        prev_gray = gray
        frame_idx += 1

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


def detect_highlights(
    video_path: str,
    threshold: float = 15.0,
    cooldown_sec: float = 2.0,
    max_highlights: int = 10,
    skip_frames: int = 0,
    max_width: int = 0,
    whisper_model: str = "tiny",
    enable_audio: bool = True,
    enable_color: bool = True,
) -> list[dict]:
    """Run the multi-signal highlight detection pipeline.

    Combines frame-difference, color-burst, and audio event detectors
    into a single ranked list of highlights with event types.

    Args:
        video_path: Path to the video file.
        threshold: Frame-diff activity threshold.
        cooldown_sec: Minimum seconds between highlights.
        max_highlights: Maximum highlights to return.
        skip_frames: Analyze every Nth frame (0 = all).
        max_width: Downscale frames to this width (0 = original).
        whisper_model: Whisper model size for audio detection.
        enable_audio: Whether to run audio detection.
        enable_color: Whether to run color-burst detection.

    Returns:
        List of dicts with 'timestamp', 'score', 'type', and 'sources'.
    """
    all_events: list[dict] = []

    # 1. Frame-difference detector
    frame_events = detect_frame_diff(
        video_path, threshold, cooldown_sec,
        max_highlights=max_highlights * 2,
        skip_frames=skip_frames, max_width=max_width,
    )
    all_events.extend(frame_events)

    # 2. Audio event detector (Whisper)
    if enable_audio:
        try:
            audio_events = detect_audio_events(video_path, whisper_model)
            for e in audio_events:
                e["sources"] = ["audio"]
            all_events.extend(audio_events)
        except (ImportError, FileNotFoundError, RuntimeError, OSError) as exc:
            print(f"[warn] Audio detection skipped: {exc}", file=sys.stderr)

    # 3. Color-burst detector
    if enable_color:
        try:
            color_events = detect_color_bursts(
                video_path,
                cooldown_sec=cooldown_sec,
                skip_frames=skip_frames,
                max_width=max_width,
            )
            for e in color_events:
                e["type"] = "effect"
                e["sources"] = ["color"]
            all_events.extend(color_events)
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            print(f"[warn] Color detection skipped: {exc}", file=sys.stderr)

    # Combine events that are close in time
    combined = _combine_events(all_events, cooldown_sec)

    # Sort by score descending and limit
    combined.sort(key=lambda h: h["score"], reverse=True)
    return combined[:max_highlights]


def _combine_events(events: list[dict], cooldown_sec: float) -> list[dict]:
    """Combine events from multiple detectors that overlap in time.

    Single-pass merge over time-sorted events. When events fall within
    the cooldown window they are merged: audio events take priority for
    type classification, scores are boosted once when multiple distinct
    sources are present, and sources are combined.
    """
    if not events:
        return []

    events.sort(key=lambda e: e["timestamp"])

    combined: list[dict] = []
    current = dict(events[0])

    for event in events[1:]:
        if event["timestamp"] - current["timestamp"] < cooldown_sec:
            # Merge into current cluster
            current["timestamp"] = event["timestamp"]
            current["score"] = max(
                current.get("score", 0.0), event.get("score", 0.0)
            )
            # Combine sources
            cur_sources = current.get("sources") or []
            for src in event.get("sources") or []:
                if src not in cur_sources:
                    cur_sources.append(src)
            current["sources"] = cur_sources
            # Audio events provide the most reliable type
            if "audio" in (event.get("sources") or []):
                current["type"] = event["type"]
        else:
            # Finalize current cluster: boost if multiple sources
            _finalize_cluster(current)
            combined.append(current)
            current = dict(event)

    _finalize_cluster(current)
    combined.append(current)
    return combined


def _finalize_cluster(cluster: dict) -> None:
    """Apply score boost if the cluster has multiple distinct sources."""
    sources = set(cluster.get("sources") or [])
    if len(sources) > 1:
        cluster["score"] = cluster.get("score", 0.0) * 1.2
    cluster["score"] = min(round(cluster["score"], 2), 100.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Signal Highlight Detector")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--threshold", type=float, default=15.0, help="Activity threshold (default: 15.0)")
    parser.add_argument("--cooldown", type=float, default=2.0, help="Cooldown between highlights in seconds")
    parser.add_argument("--max", type=int, default=10, help="Maximum number of highlights")
    parser.add_argument("--skip-frames", type=int, default=0, help="Analyze every Nth frame (0 = all)")
    parser.add_argument("--max-width", type=int, default=0, help="Downscale frames to this width (0 = original)")
    parser.add_argument("--whisper-model", type=str, default="tiny", help="Whisper model size (default: tiny)")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio detection")
    parser.add_argument("--no-color", action="store_true", help="Disable color-burst detection")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: stdout)")
    args = parser.parse_args()

    highlights = detect_highlights(
        args.video, args.threshold, args.cooldown, args.max,
        skip_frames=args.skip_frames, max_width=args.max_width,
        whisper_model=args.whisper_model,
        enable_audio=not args.no_audio,
        enable_color=not args.no_color,
    )

    if args.format == "json":
        result = json.dumps({"highlights": highlights}, indent=2)
    else:
        lines = []
        for h in highlights:
            sources = ", ".join(h.get("sources", []))
            event_type = h.get("type", "action")
            lines.append(
                f"Highlight at {h['timestamp']}s "
                f"(score: {h['score']}, type: {event_type}, sources: {sources})"
            )
        result = "\n".join(lines) if lines else "No highlights detected."

    if args.output:
        with open(args.output, "w") as f:
            f.write(result + "\n")
    else:
        print(result)


if __name__ == "__main__":
    main()
