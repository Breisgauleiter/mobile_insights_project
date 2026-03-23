import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import object_tracker


def _make_test_video(path: str, frames: list, fps: float = 10.0) -> None:
    """Write a list of BGR frames to a video file."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


@pytest.fixture
def moving_square_video(tmp_path):
    """10 fps video (50 frames = 5 s) with a white 20×20 square moving right."""
    frames = []
    for i in range(50):
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        left = i  # square shifts one pixel per frame
        frame[30:50, left : left + 20] = 255
        frames.append(frame)
    path = str(tmp_path / "moving_square.mp4")
    _make_test_video(path, frames, fps=10.0)
    return path


@pytest.fixture
def static_video(tmp_path):
    """10 fps video (30 frames = 3 s) with identical black frames."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frames = [frame.copy() for _ in range(30)]
    path = str(tmp_path / "static.mp4")
    _make_test_video(path, frames, fps=10.0)
    return path


class TestTrackObject:
    def test_returns_list_of_position_dicts(self, moving_square_video):
        positions = object_tracker.track_object(
            moving_square_video,
            start_time=0.0,
            bbox=(0, 30, 20, 20),
            duration=2.0,
            fps=2.0,
        )
        assert isinstance(positions, list)
        assert len(positions) >= 1
        for p in positions:
            assert set(p.keys()) >= {"time", "x", "y", "w", "h"}
            assert isinstance(p["time"], float)
            assert isinstance(p["x"], int)
            assert isinstance(p["y"], int)
            assert isinstance(p["w"], int)
            assert isinstance(p["h"], int)

    def test_first_position_matches_initial_bbox(self, moving_square_video):
        bbox = (5, 30, 20, 20)
        positions = object_tracker.track_object(
            moving_square_video,
            start_time=0.0,
            bbox=bbox,
            duration=2.0,
            fps=2.0,
        )
        first = positions[0]
        assert first["x"] == bbox[0]
        assert first["y"] == bbox[1]
        assert first["w"] == bbox[2]
        assert first["h"] == bbox[3]
        assert first["time"] == 0.0

    def test_first_position_time_matches_start_time(self, moving_square_video):
        positions = object_tracker.track_object(
            moving_square_video,
            start_time=1.0,
            bbox=(10, 30, 20, 20),
            duration=1.0,
            fps=2.0,
        )
        assert positions[0]["time"] == pytest.approx(1.0, abs=0.01)

    def test_duration_limits_tracking(self, moving_square_video):
        short = object_tracker.track_object(
            moving_square_video,
            start_time=0.0,
            bbox=(0, 30, 20, 20),
            duration=1.0,
            fps=5.0,
        )
        long_ = object_tracker.track_object(
            moving_square_video,
            start_time=0.0,
            bbox=(0, 30, 20, 20),
            duration=4.0,
            fps=5.0,
        )
        assert len(short) <= len(long_)
        # All recorded times should be within start + duration + small tolerance
        for p in short:
            assert p["time"] <= 1.0 + 0.2

    def test_output_fps_controls_record_count(self, moving_square_video):
        dense = object_tracker.track_object(
            moving_square_video,
            start_time=0.0,
            bbox=(0, 30, 20, 20),
            duration=2.0,
            fps=5.0,
        )
        sparse = object_tracker.track_object(
            moving_square_video,
            start_time=0.0,
            bbox=(0, 30, 20, 20),
            duration=2.0,
            fps=1.0,
        )
        assert len(dense) >= len(sparse)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            object_tracker.track_object("/nonexistent/video.mp4", 0.0, (0, 0, 10, 10))

    def test_invalid_video_raises_runtime_error(self, tmp_path):
        bad = str(tmp_path / "bad.mp4")
        with open(bad, "w") as f:
            f.write("not a video")
        with pytest.raises(RuntimeError):
            object_tracker.track_object(bad, 0.0, (0, 0, 10, 10))

    def test_bbox_dimensions_preserved_in_output(self, moving_square_video):
        positions = object_tracker.track_object(
            moving_square_video,
            start_time=0.0,
            bbox=(0, 30, 20, 20),
            duration=1.0,
            fps=2.0,
        )
        # Width and height should remain close to the initial values
        for p in positions:
            assert p["w"] > 0
            assert p["h"] > 0


class TestCreateCsrtTracker:
    def test_returns_tracker_object(self):
        tracker = object_tracker._create_csrt_tracker()
        assert tracker is not None


class TestCLI:
    def test_outputs_valid_json(self, moving_square_video):
        script = os.path.join(os.path.dirname(__file__), "..", "object_tracker.py")
        result = subprocess.run(
            [
                sys.executable,
                script,
                moving_square_video,
                "0.0",
                "0",
                "30",
                "20",
                "20",
                "--duration",
                "1",
                "--fps",
                "2",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout.strip())
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "time" in data[0]

    def test_nonexistent_video_exits_with_error(self):
        script = os.path.join(os.path.dirname(__file__), "..", "object_tracker.py")
        result = subprocess.run(
            [
                sys.executable,
                script,
                "/nonexistent/video.mp4",
                "0.0",
                "0",
                "0",
                "10",
                "10",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "Error" in result.stderr or "not found" in result.stderr.lower()

    def test_invalid_bbox_exits_with_error(self, moving_square_video):
        script = os.path.join(os.path.dirname(__file__), "..", "object_tracker.py")
        result = subprocess.run(
            [
                sys.executable,
                script,
                moving_square_video,
                "0.0",
                "0",
                "0",
                "0",   # w=0 is invalid
                "20",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
