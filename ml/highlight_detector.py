#!/usr/bin/env python3
"""
Highlight Detector for Mobile Insights.

Multi-signal pipeline that combines frame-difference analysis,
color-burst detection, audio event recognition (Whisper), and
kill-feed OCR to detect gameplay highlights with event-type classification.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

from audio_detector import detect_audio_events, detect_volume_events
from color_detector import detect_color_bursts
from killfeed_detector import detect_killfeed


def _report_progress(percent: int, stage: str) -> None:
    """Write a progress update to stderr as JSON for the server to parse."""
    print(json.dumps({"progress": percent, "stage": stage}), file=sys.stderr, flush=True)


def detect_frame_diff(
    video_path: str,
    threshold: float = 15.0,
    cooldown_sec: float = 2.0,
    max_highlights: int = 10,
    skip_frames: int = 0,
    max_width: int = 0,
    progress_callback=None,
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

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_gray = None
    candidates: list[dict] = []
    frame_idx = 0

    step = max(1, skip_frames) if skip_frames > 0 else 1

    last_reported_pct = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Report progress within frame-diff stage
        if progress_callback and total_frames > 0 and frame_idx % 100 == 0:
            pct = int(frame_idx / total_frames * 100)
            if pct != last_reported_pct:
                progress_callback(pct)
                last_reported_pct = pct

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
    enable_volume: bool = True,
    enable_killfeed: bool = True,
    debug: bool = False,
    kills_out: list[dict] | None = None,
    objective_events: list[dict] | None = None,
) -> list[dict]:
    """Run the multi-signal highlight detection pipeline.

    Combines frame-difference, color-burst, audio event, volume-based,
    kill-feed OCR, and optional minimap objective event detectors into a
    single ranked list of highlights with event-type classification.

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
        enable_volume: Whether to run volume-based fallback detection.
        enable_killfeed: Whether to run kill-feed OCR detection.
        debug: When True, print Whisper transcription segments to stderr.
        kills_out: Optional list that will be populated with raw kill events
            ``{time, killer, victim}`` detected by the kill-feed OCR step.
            Pass an empty list to receive structured kill data without a
            second call to ``detect_killfeed``.
        objective_events: Optional list of objective events produced by the
            minimap tracker (turret destructions, Lord/Turtle takes).  Each
            entry must have a ``'time'`` key.  Passed events are scored at
            80.0 with type ``'objective'`` and source ``'minimap'``.

    Returns:
        List of dicts with 'timestamp', 'score', 'type', and 'sources'.
    """
    all_events: list[dict] = []

    # 1. Frame-difference detector
    _report_progress(0, "frame_diff")
    frame_events = detect_frame_diff(
        video_path, threshold, cooldown_sec,
        max_highlights=max_highlights * 2,
        skip_frames=skip_frames, max_width=max_width,
        progress_callback=lambda pct: _report_progress(int(pct * 0.2), "frame_diff"),
    )
    all_events.extend(frame_events)
    _report_progress(20, "frame_diff")

    # 2. Audio event detector (Whisper)
    if enable_audio:
        _report_progress(20, "audio")
        try:
            audio_events = detect_audio_events(video_path, whisper_model, debug=debug)
            for e in audio_events:
                e["sources"] = ["audio"]
            all_events.extend(audio_events)
        except (ImportError, FileNotFoundError, RuntimeError, OSError) as exc:
            print(f"[warn] Audio detection skipped: {exc}", file=sys.stderr)
    _report_progress(80, "audio")

    # 3. Volume-based fallback detector (RMS energy)
    if enable_volume:
        _report_progress(80, "volume")
        try:
            volume_events = detect_volume_events(
                video_path,
                cooldown_sec=cooldown_sec,
            )
            for e in volume_events:
                e["sources"] = ["volume"]
            all_events.extend(volume_events)
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            print(f"[warn] Volume detection skipped: {exc}", file=sys.stderr)

    # 4. Color-burst detector
    if enable_color:
        _report_progress(85, "color")
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

    # 5. Kill-feed OCR detector
    if enable_killfeed:
        _report_progress(90, "killfeed")
        try:
            kill_events = detect_killfeed(video_path)
            if kills_out is not None:
                kills_out.extend(kill_events)
            for e in kill_events:
                all_events.append({
                    "timestamp": e["time"],
                    "score": 85.0,
                    "type": "kill",
                    "sources": ["killfeed"],
                })
        except (ImportError, FileNotFoundError, RuntimeError, OSError) as exc:
            print(f"[warn] Kill-feed detection skipped: {exc}", file=sys.stderr)

    # 6. Objective events (turret/Lord/Turtle from minimap analysis)
    if objective_events:
        for oe in objective_events:
            all_events.append({
                "timestamp": oe["time"],
                "score": 80.0,
                "type": "objective",
                "sources": ["minimap"],
            })

    _report_progress(100, "done")

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
    cluster["score"] = round(cluster["score"], 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Signal Highlight Detector")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--threshold", type=float, default=15.0, help="Activity threshold (default: 15.0)")
    parser.add_argument("--cooldown", type=float, default=2.0, help="Cooldown between highlights in seconds")
    parser.add_argument("--max", type=int, default=30, help="Maximum number of highlights")
    parser.add_argument("--skip-frames", type=int, default=0, help="Analyze every Nth frame (0 = all)")
    parser.add_argument("--max-width", type=int, default=0, help="Downscale frames to this width (0 = original)")
    parser.add_argument("--whisper-model", type=str, default="tiny", help="Whisper model size (default: tiny)")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio detection")
    parser.add_argument("--no-color", action="store_true", help="Disable color-burst detection")
    parser.add_argument("--no-volume", action="store_true", help="Disable volume-based fallback detection")
    parser.add_argument("--no-killfeed", action="store_true", help="Disable kill-feed OCR detection")
    parser.add_argument("--debug", action="store_true", help="Print Whisper transcription segments to stderr")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: stdout)")
    args = parser.parse_args()

    enable_killfeed = not args.no_killfeed

    # kills_out collects raw kill events from inside detect_highlights (no double call)
    kills: list[dict] = []
    highlights = detect_highlights(
        args.video, args.threshold, args.cooldown, args.max,
        skip_frames=args.skip_frames, max_width=args.max_width,
        whisper_model=args.whisper_model,
        enable_audio=not args.no_audio,
        enable_color=not args.no_color,
        enable_volume=not args.no_volume,
        enable_killfeed=enable_killfeed,
        debug=args.debug,
        kills_out=kills if enable_killfeed else None,
    )

    if args.format == "json":
        result = json.dumps({"highlights": highlights, "kills": kills}, indent=2)
    else:
        lines = []
        for h in highlights:
            sources = ", ".join(h.get("sources", []))
            event_type = h.get("type", "action")
            lines.append(
                f"Highlight at {h['timestamp']}s "
                f"(score: {h['score']}, type: {event_type}, sources: {sources})"
            )
        for k in kills:
            assist_str = f" (assist: {k['assist']})" if k.get("assist") else ""
            lines.append(f"Kill at {k['time']}s: {k['killer']} killed {k['victim']}{assist_str}")
        result = "\n".join(lines) if lines else "No highlights detected."

    if args.output:
        with open(args.output, "w") as f:
            f.write(result + "\n")
    else:
        print(result)


if __name__ == "__main__":
    main()
