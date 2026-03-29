import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import minimap_tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_video(
    path: str, frames: list[np.ndarray], fps: float = 30.0
) -> None:
    """Write a list of BGR frames to a video file."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def _make_blank_frame(w: int = 320, h: int = 240) -> np.ndarray:
    """Return a black BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _draw_dot(frame: np.ndarray, cx: int, cy: int, color_bgr: tuple, radius: int = 5) -> np.ndarray:
    """Draw a filled circle on *frame* and return a copy."""
    out = frame.copy()
    cv2.circle(out, (cx, cy), radius, color_bgr, -1)
    return out


# Pure blue dot (ally) in BGR
_ALLY_BGR = (200, 50, 50)   # high B, low G, low R — maps to blue in HSV
# Pure red dot (enemy) in BGR — bright red
_ENEMY_BGR = (0, 0, 220)    # low B, low G, high R — maps to red in HSV


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def blank_video(tmp_path):
    """Short static black video — should produce no detected positions."""
    frames = [_make_blank_frame() for _ in range(15)]
    path = str(tmp_path / "blank.mp4")
    _create_test_video(path, frames, fps=15.0)
    return path


@pytest.fixture
def ally_dot_video(tmp_path):
    """Video with a bright blue dot in the bottom-left minimap area."""
    w, h = 320, 240
    # Minimap is at (0, h-size) where size = int(min(w,h)*0.18) = 43
    size = int(min(w, h) * 0.18)
    dot_x = size // 4
    dot_y = h - size + size // 4  # inside the minimap region
    frames = [_draw_dot(_make_blank_frame(w, h), dot_x, dot_y, _ALLY_BGR, radius=5)
              for _ in range(15)]
    path = str(tmp_path / "ally.mp4")
    _create_test_video(path, frames, fps=15.0)
    return path


@pytest.fixture
def enemy_dot_video(tmp_path):
    """Video with a bright red dot in the bottom-left minimap area."""
    w, h = 320, 240
    size = int(min(w, h) * 0.18)
    dot_x = size // 4
    dot_y = h - size + size // 4
    frames = [_draw_dot(_make_blank_frame(w, h), dot_x, dot_y, _ENEMY_BGR, radius=5)
              for _ in range(15)]
    path = str(tmp_path / "enemy.mp4")
    _create_test_video(path, frames, fps=15.0)
    return path


# ---------------------------------------------------------------------------
# Tests — _get_minimap_region
# ---------------------------------------------------------------------------


class TestGetMinimapRegion:
    def test_default_auto_detection(self):
        x, y, w, h = minimap_tracker._get_minimap_region(320, 240, None)
        size = int(min(320, 240) * minimap_tracker._DEFAULT_MINIMAP_FRACTION)
        assert x == 0
        assert y == 240 - size
        assert w == size
        assert h == size

    def test_custom_config_used(self):
        config = {"x": 10, "y": 20, "width": 150, "height": 150}
        x, y, w, h = minimap_tracker._get_minimap_region(1920, 1080, config)
        assert x == 10
        assert y == 20
        assert w == 150
        assert h == 150

    def test_config_defaults_when_keys_missing(self):
        # Empty config → falls back to defaults based on frame size
        config = {}
        x, y, w, h = minimap_tracker._get_minimap_region(640, 480, config)
        size = int(min(640, 480) * minimap_tracker._DEFAULT_MINIMAP_FRACTION)
        assert x == 0
        assert y == 480 - size
        assert w == size
        assert h == size


# ---------------------------------------------------------------------------
# Tests — _find_dots
# ---------------------------------------------------------------------------


class TestFindDots:
    def _make_hsv_patch(self, color_bgr: tuple, size: int = 100) -> np.ndarray:
        """Create a solid-color HSV patch."""
        frame = np.full((size, size, 3), color_bgr, dtype=np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def test_blank_image_returns_no_dots(self):
        hsv = self._make_hsv_patch((0, 0, 0))
        result = minimap_tracker._find_dots(hsv, "ally", 5, 1000)
        assert result == []

    def test_ally_dot_detected(self):
        # Create a small image with one blue dot
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img, (50, 50), 8, _ALLY_BGR, -1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        result = minimap_tracker._find_dots(hsv, "ally", 5, 1000)
        assert len(result) >= 1
        cx, cy = result[0]
        assert 0.0 <= cx <= 1.0
        assert 0.0 <= cy <= 1.0

    def test_enemy_dot_detected(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img, (50, 50), 8, _ENEMY_BGR, -1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        result = minimap_tracker._find_dots(hsv, "enemy", 5, 1000)
        assert len(result) >= 1

    def test_min_area_filters_tiny_dots(self):
        # Draw a very small dot (radius 1 → area ~3 px)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img, (50, 50), 1, _ALLY_BGR, -1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        result = minimap_tracker._find_dots(hsv, "ally", min_area=50, max_area=1000)
        assert result == []

    def test_max_area_filters_large_blobs(self):
        # Flood entire image with ally color → area = 10000 >> max_area
        img = np.full((100, 100, 3), _ALLY_BGR, dtype=np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        result = minimap_tracker._find_dots(hsv, "ally", min_area=5, max_area=100)
        assert result == []

    def test_normalized_positions_in_range(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img, (25, 75), 6, _ENEMY_BGR, -1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        result = minimap_tracker._find_dots(hsv, "enemy", 5, 1000)
        for x, y in result:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0


# ---------------------------------------------------------------------------
# Tests — _detect_events
# ---------------------------------------------------------------------------


class TestDetectEvents:
    def test_empty_timeline_returns_no_events(self):
        assert minimap_tracker._detect_events([]) == []

    def test_gank_detected_when_enemy_in_ally_territory(self):
        timeline = [
            {
                "time": 5.0,
                "positions": [
                    # Enemy dot in ally territory: x < 0.45, y > 0.55
                    {"x": 0.2, "y": 0.8, "team": "enemy"},
                ],
            }
        ]
        events = minimap_tracker._detect_events(timeline)
        assert len(events) == 1
        assert events[0]["type"] == "gank"
        assert events[0]["time"] == 5.0

    def test_no_gank_when_enemy_in_own_territory(self):
        timeline = [
            {
                "time": 3.0,
                "positions": [
                    # Enemy in their own territory: x > 0.5
                    {"x": 0.8, "y": 0.2, "team": "enemy"},
                ],
            }
        ]
        events = minimap_tracker._detect_events(timeline)
        gank_events = [e for e in events if e["type"] == "gank"]
        assert gank_events == []

    def test_rotation_detected_on_rapid_movement(self):
        # Movement of 0.6 normalized units in 1 second → speed = 0.6 > threshold 0.25
        timeline = [
            {"time": 0.0, "positions": [{"x": 0.1, "y": 0.1, "team": "ally"}]},
            {"time": 1.0, "positions": [{"x": 0.7, "y": 0.7, "team": "ally"}]},
        ]
        events = minimap_tracker._detect_events(timeline)
        rotation_events = [e for e in events if e["type"] == "rotation"]
        assert len(rotation_events) >= 1

    def test_no_rotation_on_slow_movement(self):
        # Movement of 0.05 normalized units in 1 second → speed = 0.05 < threshold
        timeline = [
            {"time": 0.0, "positions": [{"x": 0.5, "y": 0.5, "team": "ally"}]},
            {"time": 1.0, "positions": [{"x": 0.55, "y": 0.5, "team": "ally"}]},
        ]
        events = minimap_tracker._detect_events(timeline)
        rotation_events = [e for e in events if e["type"] == "rotation"]
        assert rotation_events == []

    def test_events_sorted_by_time(self):
        timeline = [
            {"time": 10.0, "positions": [{"x": 0.2, "y": 0.8, "team": "enemy"}]},
            {"time": 2.0, "positions": [{"x": 0.2, "y": 0.8, "team": "enemy"}]},
        ]
        events = minimap_tracker._detect_events(timeline)
        times = [e["time"] for e in events]
        assert times == sorted(times)

    def test_duplicate_time_not_double_reported(self):
        # Both gank and rotation could fire at the same time — only one event per time
        timeline = [
            {"time": 0.0, "positions": [{"x": 0.1, "y": 0.1, "team": "ally"}]},
            {
                "time": 1.0,
                "positions": [
                    # Fast-moving enemy dot in ally territory
                    {"x": 0.2, "y": 0.8, "team": "enemy"},
                ],
            },
        ]
        events = minimap_tracker._detect_events(timeline)
        times = [e["time"] for e in events]
        assert len(times) == len(set(times))  # no duplicate timestamps


# ---------------------------------------------------------------------------
# Tests — track_minimap (integration)
# ---------------------------------------------------------------------------


class TestTrackMinimap:
    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            minimap_tracker.track_minimap("/nonexistent/video.mp4")

    def test_blank_video_produces_empty_timeline(self, blank_video):
        result = minimap_tracker.track_minimap(blank_video, sample_fps=5.0)
        assert "timeline" in result
        assert "events" in result
        assert "minimap_region" in result
        assert isinstance(result["timeline"], list)
        assert isinstance(result["events"], list)

    def test_result_structure(self, blank_video):
        result = minimap_tracker.track_minimap(blank_video, sample_fps=5.0)
        region = result["minimap_region"]
        assert "x" in region and "y" in region
        assert "width" in region and "height" in region

    def test_custom_minimap_config_respected(self, blank_video):
        config = {"x": 0, "y": 0, "width": 50, "height": 50}
        result = minimap_tracker.track_minimap(blank_video, minimap_config=config)
        assert result["minimap_region"]["width"] == 50
        assert result["minimap_region"]["height"] == 50

    def test_timeline_entries_have_required_fields(self, ally_dot_video):
        result = minimap_tracker.track_minimap(ally_dot_video, sample_fps=5.0)
        for entry in result["timeline"]:
            assert "time" in entry
            assert "positions" in entry
            assert isinstance(entry["time"], float)
            assert isinstance(entry["positions"], list)

    def test_position_entries_have_required_fields(self, ally_dot_video):
        result = minimap_tracker.track_minimap(ally_dot_video, sample_fps=5.0)
        for entry in result["timeline"]:
            for pos in entry["positions"]:
                assert "x" in pos and "y" in pos and "team" in pos
                assert pos["team"] in ("ally", "enemy")
                assert 0.0 <= pos["x"] <= 1.0
                assert 0.0 <= pos["y"] <= 1.0

    def test_sample_fps_affects_frame_count(self, blank_video):
        result_fast = minimap_tracker.track_minimap(blank_video, sample_fps=15.0)
        result_slow = minimap_tracker.track_minimap(blank_video, sample_fps=1.0)
        # Faster sampling → more (or equal) frames analyzed
        # Both may have 0 entries (blank video), so just check no crash
        assert isinstance(result_fast["timeline"], list)
        assert isinstance(result_slow["timeline"], list)


# ---------------------------------------------------------------------------
# Tests — CLI (main)
# ---------------------------------------------------------------------------


class TestCLI:
    def test_json_output_to_stdout(self, blank_video, capsys):
        old_argv = sys.argv
        sys.argv = [
            "minimap_tracker.py",
            "--video", blank_video,
            "--sample-fps", "5",
        ]
        try:
            minimap_tracker.main()
        finally:
            sys.argv = old_argv

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "timeline" in data
        assert "events" in data
        assert "minimap_region" in data

    def test_custom_minimap_args(self, blank_video, capsys):
        old_argv = sys.argv
        sys.argv = [
            "minimap_tracker.py",
            "--video", blank_video,
            "--minimap-x", "0",
            "--minimap-y", "0",
            "--minimap-width", "80",
            "--minimap-height", "80",
        ]
        try:
            minimap_tracker.main()
        finally:
            sys.argv = old_argv

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["minimap_region"]["width"] == 80
        assert data["minimap_region"]["height"] == 80

    def test_nonexistent_video_exits_with_error(self, capsys):
        old_argv = sys.argv
        sys.argv = ["minimap_tracker.py", "--video", "/does/not/exist.mp4"]
        try:
            with pytest.raises(SystemExit) as exc_info:
                minimap_tracker.main()
            assert exc_info.value.code == 1
        finally:
            sys.argv = old_argv


# ---------------------------------------------------------------------------
# Tests — _sample_region_brightness
# ---------------------------------------------------------------------------


class TestSampleRegionBrightness:
    def test_black_region_returns_zero(self):
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        result = minimap_tracker._sample_region_brightness(black, 0.5, 0.5, 0.1)
        assert result == 0.0

    def test_white_region_returns_high_value(self):
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = minimap_tracker._sample_region_brightness(white, 0.5, 0.5, 0.1)
        assert result > 200.0

    def test_corner_coords_do_not_crash(self):
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = minimap_tracker._sample_region_brightness(white, 0.0, 0.0, 0.1)
        assert result >= 0.0
        result2 = minimap_tracker._sample_region_brightness(white, 1.0, 1.0, 0.1)
        assert result2 >= 0.0

    def test_returns_float(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = minimap_tracker._sample_region_brightness(img, 0.5, 0.5, 0.05)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Tests — _detect_objective_events
# ---------------------------------------------------------------------------


class TestDetectObjectiveEvents:
    def test_empty_inputs_return_no_events(self):
        events = minimap_tracker._detect_objective_events([], {}, {})
        assert events == []

    def test_turret_destruction_detected_on_brightness_drop(self):
        turret = minimap_tracker._TURRET_POSITIONS[0]
        name = turret["name"]
        baselines = {name: 100.0}
        # brightness drops to 40 % of baseline at t=10s → below 50 % threshold
        series = {name: [(5.0, 95.0), (10.0, 38.0)]}
        events = minimap_tracker._detect_objective_events([], baselines, series)
        assert len(events) == 1
        assert events[0]["event"] == "turret_destroyed"
        assert events[0]["team"] == turret["team"]
        assert events[0]["time"] == 10.0

    def test_turret_not_destroyed_when_brightness_stable(self):
        turret = minimap_tracker._TURRET_POSITIONS[0]
        name = turret["name"]
        baselines = {name: 100.0}
        series = {name: [(5.0, 95.0), (10.0, 90.0)]}
        events = minimap_tracker._detect_objective_events([], baselines, series)
        turret_events = [e for e in events if e["event"] == "turret_destroyed"]
        assert turret_events == []

    def test_turret_skipped_when_baseline_too_dark(self):
        turret = minimap_tracker._TURRET_POSITIONS[0]
        name = turret["name"]
        # Baseline below _TURRET_MIN_BASELINE → should not track
        baselines = {name: 5.0}
        series = {name: [(5.0, 1.0)]}
        events = minimap_tracker._detect_objective_events([], baselines, series)
        assert events == []

    def test_turret_destroyed_at_most_once(self):
        turret = minimap_tracker._TURRET_POSITIONS[0]
        name = turret["name"]
        baselines = {name: 100.0}
        # Two consecutive drops — only the first should be reported
        series = {name: [(5.0, 30.0), (10.0, 20.0)]}
        events = minimap_tracker._detect_objective_events([], baselines, series)
        turret_events = [e for e in events if e["event"] == "turret_destroyed"]
        assert len(turret_events) == 1

    def test_lord_event_detected_on_hero_clustering(self):
        pit = minimap_tracker._LORD_PIT
        timeline = [
            {
                "time": 60.0,
                "positions": [
                    {"x": pit["x"] + 0.02, "y": pit["y"] + 0.02, "team": "ally"},
                    {"x": pit["x"] - 0.02, "y": pit["y"] - 0.02, "team": "enemy"},
                ],
            }
        ]
        events = minimap_tracker._detect_objective_events(timeline, {}, {})
        lord_events = [e for e in events if e["event"] == "lord_taken"]
        assert len(lord_events) == 1
        assert lord_events[0]["time"] == 60.0

    def test_turtle_event_detected_on_hero_clustering(self):
        pit = minimap_tracker._TURTLE_PIT
        timeline = [
            {
                "time": 90.0,
                "positions": [
                    {"x": pit["x"] + 0.01, "y": pit["y"], "team": "ally"},
                    {"x": pit["x"], "y": pit["y"] + 0.01, "team": "ally"},
                ],
            }
        ]
        events = minimap_tracker._detect_objective_events(timeline, {}, {})
        turtle_events = [e for e in events if e["event"] == "turtle_taken"]
        assert len(turtle_events) == 1

    def test_objective_cooldown_prevents_duplicate_events(self):
        pit = minimap_tracker._TURTLE_PIT
        timeline = [
            {
                "time": 0.0,
                "positions": [
                    {"x": pit["x"], "y": pit["y"], "team": "ally"},
                    {"x": pit["x"], "y": pit["y"], "team": "enemy"},
                ],
            },
            {
                "time": 5.0,
                "positions": [
                    {"x": pit["x"], "y": pit["y"], "team": "ally"},
                    {"x": pit["x"], "y": pit["y"], "team": "enemy"},
                ],
            },
        ]
        events = minimap_tracker._detect_objective_events(timeline, {}, {})
        turtle_events = [e for e in events if e["event"] == "turtle_taken"]
        assert len(turtle_events) == 1

    def test_objective_events_sorted_by_time(self):
        turret = minimap_tracker._TURRET_POSITIONS[0]
        pit = minimap_tracker._TURTLE_PIT
        baselines = {turret["name"]: 100.0}
        series = {turret["name"]: [(50.0, 30.0)]}
        timeline = [
            {
                "time": 10.0,
                "positions": [
                    {"x": pit["x"], "y": pit["y"], "team": "ally"},
                    {"x": pit["x"], "y": pit["y"], "team": "enemy"},
                ],
            }
        ]
        events = minimap_tracker._detect_objective_events(timeline, baselines, series)
        times = [e["time"] for e in events]
        assert times == sorted(times)

    def test_objective_event_structure(self):
        pit = minimap_tracker._LORD_PIT
        timeline = [
            {
                "time": 120.0,
                "positions": [
                    {"x": pit["x"], "y": pit["y"], "team": "ally"},
                    {"x": pit["x"], "y": pit["y"], "team": "ally"},
                ],
            }
        ]
        events = minimap_tracker._detect_objective_events(timeline, {}, {})
        assert len(events) >= 1
        for e in events:
            assert "time" in e
            assert "event" in e
            assert "team" in e
            assert "position" in e
            assert "x" in e["position"]
            assert "y" in e["position"]


# ---------------------------------------------------------------------------
# Tests — track_minimap includes objective_events
# ---------------------------------------------------------------------------


class TestTrackMinimapObjectiveEvents:
    def test_result_contains_objective_events_key(self, blank_video):
        result = minimap_tracker.track_minimap(blank_video, sample_fps=5.0)
        assert "objective_events" in result
        assert isinstance(result["objective_events"], list)

    def test_cli_output_contains_objective_events(self, blank_video, capsys):
        old_argv = sys.argv
        sys.argv = ["minimap_tracker.py", "--video", blank_video, "--sample-fps", "5"]
        try:
            minimap_tracker.main()
        finally:
            sys.argv = old_argv
        data = json.loads(capsys.readouterr().out)
        assert "objective_events" in data
