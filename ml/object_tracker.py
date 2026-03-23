#!/usr/bin/env python3
"""
Hero Object Tracker for Mobile Insights.

Tracks a bounding box region through a video starting from a given timestamp
using the OpenCV CSRT (Channel and Spatial Reliability Tracking) algorithm.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2


def _create_csrt_tracker() -> Any:
    """Create a CSRT tracker, supporting both legacy and modern OpenCV APIs.

    Returns:
        An initialized CSRT tracker object.

    Raises:
        RuntimeError: If CSRT tracker is not available in this OpenCV build.
    """
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    opencv_version = getattr(cv2, "__version__", "unknown")
    raise RuntimeError(
        "OpenCV CSRT tracker is not available in this build. "
        "Install an OpenCV build with contrib modules (e.g. 'opencv-contrib-python'). "
        f"Detected OpenCV version: {opencv_version}."
    )


def track_object(
    video_path: str,
    start_time: float,
    bbox: tuple[int, int, int, int],
    duration: float = 30.0,
    fps: float = 5.0,
) -> list[dict]:
    """Track a bounding box through a video using the CSRT tracker.

    Processes every frame of the video through the tracker but only records
    positions at the requested output fps rate for compact results.

    Args:
        video_path: Path to the video file.
        start_time: Start timestamp in seconds.
        bbox: Bounding box as (x, y, w, h) in pixels relative to the video frame.
        duration: Maximum tracking duration in seconds (default: 30).
        fps: Positions to record per second (default: 5).

    Returns:
        List of dicts, each with keys 'time', 'x', 'y', 'w', 'h'.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If the video cannot be opened, the start frame cannot be read,
            or tracker initialization fails.
        ValueError: If fps or duration are not positive.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            raise RuntimeError("Video has invalid FPS")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = int(start_time * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        ret, init_frame = cap.read()
        if not ret:
            raise RuntimeError(f"Could not read frame at t={start_time:.3f}s")

        tracker = _create_csrt_tracker()
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        if not tracker.init(init_frame, (x, y, w, h)):
            raise RuntimeError(
                "Tracker initialization failed. "
                "The bounding box may be outside the frame boundaries."
            )

        positions: list[dict] = [
            {"time": round(start_time, 3), "x": x, "y": y, "w": w, "h": h}
        ]

        end_frame = start_frame + int(duration * video_fps)
        if total_frames > 0:
            end_frame = min(end_frame, total_frames - 1)

        # Interval between recorded positions (in video frames)
        record_interval = video_fps / fps
        next_record_at = start_frame + record_interval
        current_frame = start_frame + 1

        while True:
            ret, frame = cap.read()
            if not ret or current_frame > end_frame:
                break

            ok, new_bbox = tracker.update(frame)
            if not ok:
                break

            if current_frame >= next_record_at:
                bx = int(new_bbox[0])
                by = int(new_bbox[1])
                bw = int(new_bbox[2])
                bh = int(new_bbox[3])
                t = round(current_frame / video_fps, 3)
                positions.append({"time": t, "x": bx, "y": by, "w": bw, "h": bh})
                next_record_at += record_interval

            current_frame += 1

    finally:
        cap.release()

    return positions


def main() -> None:
    """CLI entry point for the object tracker.

    Usage:
        python3 object_tracker.py <video> <time> <x> <y> <w> <h>
            [--duration SECS] [--fps FPS]

    Outputs a JSON array of {time, x, y, w, h} dicts to stdout.
    """
    parser = argparse.ArgumentParser(
        description="OpenCV CSRT Object Tracker for Mobile Insights"
    )
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("time", type=float, help="Start time in seconds")
    parser.add_argument("x", type=int, help="Bounding box left edge (pixels)")
    parser.add_argument("y", type=int, help="Bounding box top edge (pixels)")
    parser.add_argument("w", type=int, help="Bounding box width (pixels)")
    parser.add_argument("h", type=int, help="Bounding box height (pixels)")
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Maximum tracking duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Positions to record per second (default: 5)",
    )
    args = parser.parse_args()

    if args.w <= 0 or args.h <= 0:
        print("Error: bounding box w and h must be positive", file=sys.stderr)
        sys.exit(1)

    if args.time < 0:
        print("Error: start time must be non-negative", file=sys.stderr)
        sys.exit(1)

    if args.fps <= 0:
        print("Error: --fps must be positive", file=sys.stderr)
        sys.exit(1)

    if args.duration <= 0:
        print("Error: --duration must be positive", file=sys.stderr)
        sys.exit(1)

    try:
        positions = track_object(
            args.video,
            args.time,
            (args.x, args.y, args.w, args.h),
            duration=args.duration,
            fps=args.fps,
        )
        print(json.dumps(positions))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
