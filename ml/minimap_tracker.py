#!/usr/bin/env python3
"""
Minimap Hero Tracker for Mobile Insights.

Detects and tracks hero positions on the MLBB minimap by isolating
the minimap region and using HSV color filtering to find hero indicator dots.
Ally dots (blue) and enemy dots (red) are detected separately and their
positions are reported normalized to the minimap coordinate space.
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

# Default minimap size as a fraction of the shorter frame dimension.
# MLBB minimap occupies roughly 18% of the screen in the bottom-left corner.
_DEFAULT_MINIMAP_FRACTION = 0.18

# HSV color ranges for hero indicator dots
_ALLY_HSV_LOWER = np.array([95, 100, 100])    # Blue (ally)
_ALLY_HSV_UPPER = np.array([135, 255, 255])
_ENEMY_HSV_LOWER1 = np.array([0, 120, 120])   # Red (enemy) — lower hue wrap
_ENEMY_HSV_UPPER1 = np.array([10, 255, 255])
_ENEMY_HSV_LOWER2 = np.array([165, 120, 120]) # Red (enemy) — upper hue wrap
_ENEMY_HSV_UPPER2 = np.array([180, 255, 255])

# Contour area thresholds (pixels on the cropped minimap)
_MIN_DOT_AREA = 5
_MAX_DOT_AREA = 1000

# Gank detection: enemy dot appears in ally territory (bottom-left quadrant).
# Normalized minimap coords: (0,0) = top-left, (1,1) = bottom-right.
# Ally starts bottom-left → ally territory: x < 0.45 AND y > 0.55.
_GANK_ENEMY_X_MAX = 0.45
_GANK_ENEMY_Y_MIN = 0.55

# Rapid-movement threshold (normalized units per second) for rotation detection
_RAPID_MOVE_THRESHOLD = 0.25

# Minimum allowed sample_fps to avoid division-by-zero in frame-step calculation
_MIN_SAMPLE_FPS = 0.1

# Known MLBB turret positions on the minimap (normalized 0-1 coords).
# (0,0) = top-left of the minimap; ally spawns bottom-left.
_TURRET_POSITIONS: list[dict] = [
    {"name": "ally_top_t1",  "team": "ally",  "x": 0.12, "y": 0.28},
    {"name": "ally_top_t2",  "team": "ally",  "x": 0.20, "y": 0.18},
    {"name": "ally_mid_t1",  "team": "ally",  "x": 0.28, "y": 0.45},
    {"name": "ally_mid_t2",  "team": "ally",  "x": 0.38, "y": 0.35},
    {"name": "ally_bot_t1",  "team": "ally",  "x": 0.30, "y": 0.68},
    {"name": "ally_bot_t2",  "team": "ally",  "x": 0.18, "y": 0.78},
    {"name": "enemy_top_t1", "team": "enemy", "x": 0.88, "y": 0.72},
    {"name": "enemy_top_t2", "team": "enemy", "x": 0.80, "y": 0.82},
    {"name": "enemy_mid_t1", "team": "enemy", "x": 0.72, "y": 0.55},
    {"name": "enemy_mid_t2", "team": "enemy", "x": 0.62, "y": 0.65},
    {"name": "enemy_bot_t1", "team": "enemy", "x": 0.70, "y": 0.32},
    {"name": "enemy_bot_t2", "team": "enemy", "x": 0.80, "y": 0.22},
]

# Lord and Turtle pit positions on the minimap (normalized coords)
_LORD_PIT: dict = {"x": 0.78, "y": 0.28}
_TURTLE_PIT: dict = {"x": 0.52, "y": 0.68}

# Turret detection parameters
_TURRET_REGION_FRACTION = 0.04   # Half-width/height of sampling region relative to minimap
_TURRET_DESTROY_RATIO = 0.50     # Brightness drops to < 50 % of baseline = destroyed
_TURRET_MIN_BASELINE = 20.0      # Skip turrets whose baseline is too dark (icon not visible)

# Objective (Lord/Turtle) detection parameters
_OBJECTIVE_RADIUS = 0.15         # Normalised radius around pit to count as "near"
_OBJECTIVE_MIN_HEROES = 2        # Minimum heroes near pit to register activity
_OBJECTIVE_COOLDOWN_SEC = 30.0   # Seconds between repeated events for the same objective


def _get_minimap_region(
    frame_width: int,
    frame_height: int,
    config: dict | None,
) -> tuple[int, int, int, int]:
    """Compute the minimap crop region (x, y, width, height) in pixels.

    Args:
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        config: Optional dict with integer keys 'x', 'y', 'width', 'height'.
                If None, the minimap is estimated from the bottom-left corner
                using the default fraction of the shorter frame dimension.

    Returns:
        Tuple of (x, y, width, height) in pixel coordinates.
    """
    if config is not None:
        size = int(min(frame_width, frame_height) * _DEFAULT_MINIMAP_FRACTION)
        x = config.get("x")
        y = config.get("y")
        w = config.get("width")
        h = config.get("height")
        return (
            int(x) if x is not None else 0,
            int(y) if y is not None else frame_height - size,
            int(w) if w is not None else size,
            int(h) if h is not None else size,
        )
    size = int(min(frame_width, frame_height) * _DEFAULT_MINIMAP_FRACTION)
    return 0, frame_height - size, size, size


def _find_dots(
    minimap_hsv: np.ndarray,
    team: str,
    min_area: int,
    max_area: int,
) -> list[tuple[float, float]]:
    """Find hero dot centers for a given team using HSV masking.

    Args:
        minimap_hsv: HSV image of the cropped minimap region.
        team: "ally" for blue dots or "enemy" for red dots.
        min_area: Minimum contour area in pixels to count as a dot.
        max_area: Maximum contour area in pixels to count as a dot.

    Returns:
        List of (x, y) center positions normalized to [0, 1] relative to the
        minimap region, where (0, 0) is the top-left corner.
    """
    h, w = minimap_hsv.shape[:2]

    if team == "ally":
        mask = cv2.inRange(minimap_hsv, _ALLY_HSV_LOWER, _ALLY_HSV_UPPER)
    else:  # enemy — red wraps around 0/180 in HSV
        mask1 = cv2.inRange(minimap_hsv, _ENEMY_HSV_LOWER1, _ENEMY_HSV_UPPER1)
        mask2 = cv2.inRange(minimap_hsv, _ENEMY_HSV_LOWER2, _ENEMY_HSV_UPPER2)
        mask = cv2.bitwise_or(mask1, mask2)

    # Small morphological opening to remove isolated noise pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    positions: list[tuple[float, float]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            M = cv2.moments(cnt)
            m00 = M["m00"]
            if m00 > 0:
                cx = round(M["m10"] / m00 / w, 3)
                cy = round(M["m01"] / m00 / h, 3)
                positions.append((cx, cy))

    return positions


def _detect_events(timeline: list[dict]) -> list[dict]:
    """Detect potential ganks and rotations from the position timeline.

    Gank: an enemy dot appears in ally territory (bottom-left quadrant).
    Rotation: any dot moves more than _RAPID_MOVE_THRESHOLD normalized
              units per second between consecutive sampled frames.

    Args:
        timeline: List of frame entries, each with 'time' and 'positions'.

    Returns:
        Sorted list of event dicts with 'time', 'type', and 'description'.
    """
    events: list[dict] = []
    reported_times: set[float] = set()

    prev_ally: list[tuple[float, float]] = []
    prev_enemy: list[tuple[float, float]] = []
    prev_time: float | None = None

    for entry in timeline:
        t = entry["time"]
        ally = [(p["x"], p["y"]) for p in entry["positions"] if p["team"] == "ally"]
        enemy = [(p["x"], p["y"]) for p in entry["positions"] if p["team"] == "enemy"]

        # --- Gank detection ---
        for ex, ey in enemy:
            if ex < _GANK_ENEMY_X_MAX and ey > _GANK_ENEMY_Y_MIN and t not in reported_times:
                events.append({
                    "time": t,
                    "type": "gank",
                    "description": (
                        f"Enemy spotted in ally territory at ({ex:.2f}, {ey:.2f})"
                    ),
                })
                reported_times.add(t)
                break

        # --- Rotation detection ---
        if prev_time is not None:
            dt = t - prev_time
            if dt > 0:
                for team_label, current_pts, prev_pts in [
                    ("Ally", ally, prev_ally),
                    ("Enemy", enemy, prev_enemy),
                ]:
                    for cx, cy in current_pts:
                        for px, py in prev_pts:
                            dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                            if dist / dt > _RAPID_MOVE_THRESHOLD and t not in reported_times:
                                events.append({
                                    "time": t,
                                    "type": "rotation",
                                    "description": (
                                        f"{team_label} hero rapid movement detected"
                                    ),
                                })
                                reported_times.add(t)
                                break

        prev_ally = ally
        prev_enemy = enemy
        prev_time = t

    events.sort(key=lambda e: e["time"])
    return events


def _sample_region_brightness(
    minimap_bgr: np.ndarray,
    norm_x: float,
    norm_y: float,
    region_fraction: float,
) -> float:
    """Return the mean pixel brightness of a small region around a minimap point.

    Args:
        minimap_bgr: BGR image of the cropped minimap region.
        norm_x: Normalised x-coordinate of the centre point (0-1).
        norm_y: Normalised y-coordinate of the centre point (0-1).
        region_fraction: Half-width/height of the sampling region as a
            fraction of the minimap dimensions.

    Returns:
        Mean grayscale brightness in [0, 255], or 0.0 for an empty region.
    """
    h, w = minimap_bgr.shape[:2]
    rx = max(1, int(w * region_fraction))
    ry = max(1, int(h * region_fraction))
    cx = int(norm_x * w)
    cy = int(norm_y * h)
    x1 = max(0, cx - rx)
    y1 = max(0, cy - ry)
    x2 = min(w, cx + rx)
    y2 = min(h, cy + ry)
    region = minimap_bgr[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def _detect_objective_events(
    timeline: list[dict],
    turret_baselines: dict[str, float],
    turret_brightness_series: dict[str, list[tuple[float, float]]],
) -> list[dict]:
    """Detect turret-destruction and Lord/Turtle objective events.

    Turret destruction is inferred when the mean brightness of a turret's
    pixel region drops to less than *_TURRET_DESTROY_RATIO* of its baseline
    (the brightness recorded in the first sampled frame).

    Lord/Turtle events are inferred when *_OBJECTIVE_MIN_HEROES* or more
    hero dots appear within *_OBJECTIVE_RADIUS* of the pit position.

    Args:
        timeline: List of frame entries from the hero-tracking pass, each
            with 'time' and 'positions'.
        turret_baselines: Mapping of turret name → baseline brightness.
        turret_brightness_series: Mapping of turret name → list of
            (timestamp, brightness) pairs recorded during tracking.

    Returns:
        Sorted list of objective event dicts:
        ``{'time': float, 'event': str, 'team': str, 'position': dict}``.
        Event type is one of ``'turret_destroyed'``, ``'lord_taken'``,
        ``'turtle_taken'``.
    """
    events: list[dict] = []

    # --- Turret destruction detection ---
    for turret in _TURRET_POSITIONS:
        name = turret["name"]
        baseline = turret_baselines.get(name)
        series = turret_brightness_series.get(name, [])
        if baseline is None or baseline < _TURRET_MIN_BASELINE:
            continue  # no reliable baseline for this turret

        destroyed_threshold = baseline * _TURRET_DESTROY_RATIO
        already_destroyed = False
        for timestamp, brightness in series:
            if not already_destroyed and brightness < destroyed_threshold:
                events.append({
                    "time": timestamp,
                    "event": "turret_destroyed",
                    "team": turret["team"],
                    "position": {"x": turret["x"], "y": turret["y"]},
                })
                already_destroyed = True  # report each turret at most once

    # --- Lord / Turtle activity detection ---
    last_lord_time: float = -_OBJECTIVE_COOLDOWN_SEC
    last_turtle_time: float = -_OBJECTIVE_COOLDOWN_SEC

    for entry in timeline:
        t = entry["time"]
        positions = entry.get("positions", [])

        for pit_name, pit, event_type in [
            ("lord", _LORD_PIT, "lord_taken"),
            ("turtle", _TURTLE_PIT, "turtle_taken"),
        ]:
            last_time = last_lord_time if pit_name == "lord" else last_turtle_time
            if t - last_time < _OBJECTIVE_COOLDOWN_SEC:
                continue

            nearby = sum(
                1
                for p in positions
                if (p["x"] - pit["x"]) ** 2 + (p["y"] - pit["y"]) ** 2
                <= _OBJECTIVE_RADIUS ** 2
            )
            if nearby >= _OBJECTIVE_MIN_HEROES:
                events.append({
                    "time": t,
                    "event": event_type,
                    "team": "unknown",
                    "position": {"x": pit["x"], "y": pit["y"]},
                })
                if pit_name == "lord":
                    last_lord_time = t
                else:
                    last_turtle_time = t

    events.sort(key=lambda e: e["time"])
    return events


def track_minimap(
    video_path: str,
    minimap_config: dict | None = None,
    sample_fps: float = 3.0,
    min_dot_area: int = _MIN_DOT_AREA,
    max_dot_area: int = _MAX_DOT_AREA,
) -> dict:
    """Detect and track hero positions on the MLBB minimap.

    Samples the video at *sample_fps* frames per second, crops the minimap
    region from the bottom-left corner, and applies HSV color filtering to
    locate ally (blue) and enemy (red) hero indicator dots via contour
    detection.  Positions are normalized to [0, 1] relative to the minimap
    crop so that (0, 0) is the top-left of the minimap.

    Args:
        video_path: Absolute or relative path to the input video file.
        minimap_config: Optional pixel-coordinate config for the minimap crop:
            {'x': int, 'y': int, 'width': int, 'height': int}.
            Defaults to auto-detection from the bottom-left corner.
        sample_fps: Frames per second to sample (recommended 2–5).
        min_dot_area: Minimum contour area in pixels to treat as a hero dot.
        max_dot_area: Maximum contour area in pixels to treat as a hero dot.

    Returns:
        Dict with:
            - 'timeline': list of {'time': float, 'positions': list of
              {'x': float, 'y': float, 'team': str}} sorted by time.
            - 'events': list of {'time': float, 'type': str,
              'description': str} for detected ganks/rotations.
            - 'objective_events': list of {'time': float, 'event': str,
              'team': str, 'position': {'x': float, 'y': float}} for
              detected turret destructions and Lord/Turtle events.
            - 'minimap_region': {'x': int, 'y': int, 'width': int,
              'height': int} pixel bounds used for analysis.

    Raises:
        FileNotFoundError: If *video_path* does not exist.
        RuntimeError: If the video cannot be opened or has an invalid FPS.
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

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mm_x, mm_y, mm_w, mm_h = _get_minimap_region(frame_width, frame_height, minimap_config)

    # Clamp region to frame boundaries
    mm_x = max(0, min(mm_x, frame_width - 1))
    mm_y = max(0, min(mm_y, frame_height - 1))
    mm_w = max(1, min(mm_w, frame_width - mm_x))
    mm_h = max(1, min(mm_h, frame_height - mm_y))

    # Number of raw frames to skip between analyzed frames
    frame_step = max(1, round(fps / max(_MIN_SAMPLE_FPS, sample_fps)))

    timeline: list[dict] = []
    frame_idx = 0

    # Turret tracking state: baseline brightness per turret + full series
    turret_baselines: dict[str, float] = {}
    turret_brightness_series: dict[str, list[tuple[float, float]]] = {
        t["name"]: [] for t in _TURRET_POSITIONS
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            timestamp = round(frame_idx / fps, 2)

            minimap = frame[mm_y : mm_y + mm_h, mm_x : mm_x + mm_w]
            if minimap.size == 0:
                frame_idx += 1
                continue

            minimap_hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)

            ally_pos = _find_dots(minimap_hsv, "ally", min_dot_area, max_dot_area)
            enemy_pos = _find_dots(minimap_hsv, "enemy", min_dot_area, max_dot_area)

            positions = (
                [{"x": x, "y": y, "team": "ally"} for x, y in ally_pos]
                + [{"x": x, "y": y, "team": "enemy"} for x, y in enemy_pos]
            )

            if positions:
                timeline.append({"time": timestamp, "positions": positions})

            # Sample turret pixel regions for objective detection
            for turret in _TURRET_POSITIONS:
                name = turret["name"]
                brightness = _sample_region_brightness(
                    minimap, turret["x"], turret["y"], _TURRET_REGION_FRACTION
                )
                if name not in turret_baselines:
                    turret_baselines[name] = brightness
                turret_brightness_series[name].append((timestamp, brightness))

        frame_idx += 1

    cap.release()

    events = _detect_events(timeline)
    objective_events = _detect_objective_events(
        timeline, turret_baselines, turret_brightness_series
    )

    return {
        "timeline": timeline,
        "events": events,
        "objective_events": objective_events,
        "minimap_region": {"x": mm_x, "y": mm_y, "width": mm_w, "height": mm_h},
    }


def main() -> None:
    """CLI entry point for the minimap tracker."""
    parser = argparse.ArgumentParser(
        description="MLBB Minimap Hero Position Tracker"
    )
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument(
        "--minimap-x", type=int, default=None,
        help="Minimap left edge in pixels (default: 0)",
    )
    parser.add_argument(
        "--minimap-y", type=int, default=None,
        help="Minimap top edge in pixels (default: auto from bottom)",
    )
    parser.add_argument(
        "--minimap-width", type=int, default=None,
        help="Minimap width in pixels (default: auto)",
    )
    parser.add_argument(
        "--minimap-height", type=int, default=None,
        help="Minimap height in pixels (default: auto)",
    )
    parser.add_argument(
        "--sample-fps", type=float, default=3.0,
        help="Frames per second to sample (default: 3.0)",
    )
    parser.add_argument(
        "--min-dot-area", type=int, default=_MIN_DOT_AREA,
        help=f"Minimum dot contour area in pixels (default: {_MIN_DOT_AREA})",
    )
    parser.add_argument(
        "--max-dot-area", type=int, default=_MAX_DOT_AREA,
        help=f"Maximum dot contour area in pixels (default: {_MAX_DOT_AREA})",
    )
    args = parser.parse_args()

    config: dict | None = None
    if any(
        v is not None
        for v in [args.minimap_x, args.minimap_y, args.minimap_width, args.minimap_height]
    ):
        config = {
            "x": args.minimap_x,
            "y": args.minimap_y,
            "width": args.minimap_width,
            "height": args.minimap_height,
        }

    try:
        result = track_minimap(
            args.video,
            minimap_config=config,
            sample_fps=args.sample_fps,
            min_dot_area=args.min_dot_area,
            max_dot_area=args.max_dot_area,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
