import os

import cv2
import numpy as np
import pytest

# Add parent dir to path so we can import highlight_detector
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import highlight_detector


def _create_test_video(path: str, frames: list[np.ndarray], fps: float = 30.0) -> None:
    """Create a small test video from a list of frames."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


@pytest.fixture
def static_video(tmp_path):
    """Video with identical frames — should produce no highlights."""
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    frames = [frame.copy() for _ in range(30)]
    path = str(tmp_path / "static.mp4")
    _create_test_video(path, frames)
    return path


@pytest.fixture
def activity_video(tmp_path):
    """Video that starts black, then has a bright flash at frame 15."""
    black = np.zeros((60, 80, 3), dtype=np.uint8)
    white = np.full((60, 80, 3), 255, dtype=np.uint8)
    frames = [black.copy() for _ in range(15)]
    frames.append(white)  # frame 15: big change
    frames += [black.copy() for _ in range(14)]
    path = str(tmp_path / "activity.mp4")
    _create_test_video(path, frames, fps=30.0)
    return path


class TestDetectHighlights:
    def test_static_video_no_highlights(self, static_video):
        results = highlight_detector.detect_highlights(static_video, threshold=15.0)
        assert results == []

    def test_activity_video_detects_flash(self, activity_video):
        results = highlight_detector.detect_highlights(activity_video, threshold=10.0)
        assert len(results) >= 1
        # The flash is at frame 15, so around 0.5s at 30fps
        assert any(0.0 <= h["timestamp"] <= 1.0 for h in results)
        assert all("score" in h and "timestamp" in h for h in results)

    def test_threshold_controls_sensitivity(self, activity_video):
        low = highlight_detector.detect_highlights(activity_video, threshold=1.0)
        high = highlight_detector.detect_highlights(activity_video, threshold=200.0)
        assert len(low) >= len(high)

    def test_max_highlights_limits_output(self, activity_video):
        results = highlight_detector.detect_highlights(
            activity_video, threshold=1.0, max_highlights=1
        )
        assert len(results) <= 1

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            highlight_detector.detect_highlights("/nonexistent/video.mp4")

    def test_returns_list_of_dicts(self, activity_video):
        results = highlight_detector.detect_highlights(activity_video, threshold=10.0)
        assert isinstance(results, list)
        for h in results:
            assert isinstance(h, dict)
            assert isinstance(h["timestamp"], float)
            assert isinstance(h["score"], float)


class TestMergeCandidates:
    def test_empty_candidates(self):
        assert highlight_detector._merge_candidates([], 2.0) == []

    def test_merge_within_cooldown(self):
        candidates = [
            {"timestamp": 1.0, "score": 10.0},
            {"timestamp": 1.5, "score": 20.0},  # within cooldown, higher score
            {"timestamp": 5.0, "score": 15.0},
        ]
        merged = highlight_detector._merge_candidates(candidates, 2.0)
        assert len(merged) == 2
        assert merged[0]["score"] == 20.0  # kept the higher one
        assert merged[1]["timestamp"] == 5.0

    def test_no_merge_outside_cooldown(self):
        candidates = [
            {"timestamp": 1.0, "score": 10.0},
            {"timestamp": 5.0, "score": 15.0},
        ]
        merged = highlight_detector._merge_candidates(candidates, 2.0)
        assert len(merged) == 2


class TestCLI:
    def test_json_output(self, activity_video, tmp_path):
        import json
        output_path = str(tmp_path / "results.json")
        # Simulate CLI args
        import sys
        old_argv = sys.argv
        sys.argv = [
            "highlight_detector.py",
            "--video", activity_video,
            "--format", "json",
            "--output", output_path,
        ]
        try:
            highlight_detector.main()
        finally:
            sys.argv = old_argv

        with open(output_path) as f:
            data = json.load(f)
        assert "highlights" in data
        assert isinstance(data["highlights"], list)
