import os
import sys

import cv2
import numpy as np
import pytest

# Add parent dir to path so we can import color_detector
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import color_detector


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
    """Video with identical grey frames — no color bursts."""
    frame = np.full((60, 80, 3), 128, dtype=np.uint8)
    frames = [frame.copy() for _ in range(30)]
    path = str(tmp_path / "static.mp4")
    _create_test_video(path, frames)
    return path


@pytest.fixture
def red_flash_video(tmp_path):
    """Video that starts grey, then has a red flash at frame 15."""
    grey = np.full((60, 80, 3), 128, dtype=np.uint8)
    # Create bright red frame (BGR: blue=0, green=0, red=255)
    red = np.zeros((60, 80, 3), dtype=np.uint8)
    red[:, :, 2] = 255  # Red channel in BGR

    frames = [grey.copy() for _ in range(15)]
    frames.append(red)  # frame 15: red flash
    frames += [grey.copy() for _ in range(14)]
    path = str(tmp_path / "red_flash.mp4")
    _create_test_video(path, frames, fps=30.0)
    return path


class TestDetectColorBursts:
    def test_static_video_no_bursts(self, static_video):
        results = color_detector.detect_color_bursts(static_video)
        assert results == []

    def test_red_flash_detected(self, red_flash_video):
        results = color_detector.detect_color_bursts(
            red_flash_video, burst_threshold=1.5,
        )
        assert len(results) >= 1
        # Red flash is at frame 15 → ~0.5s at 30fps
        assert any(0.0 <= e["timestamp"] <= 1.5 for e in results)
        assert all("score" in e and "dominant_color" in e for e in results)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            color_detector.detect_color_bursts("/nonexistent/video.mp4")

    def test_max_events_limits_output(self, red_flash_video):
        results = color_detector.detect_color_bursts(
            red_flash_video, burst_threshold=1.0, max_events=1,
        )
        assert len(results) <= 1

    def test_skip_frames(self, red_flash_video):
        full = color_detector.detect_color_bursts(
            red_flash_video, burst_threshold=1.5,
        )
        skipped = color_detector.detect_color_bursts(
            red_flash_video, burst_threshold=1.5, skip_frames=3,
        )
        assert isinstance(skipped, list)

    def test_max_width(self, red_flash_video):
        results = color_detector.detect_color_bursts(
            red_flash_video, burst_threshold=1.5, max_width=40,
        )
        assert isinstance(results, list)

    def test_returns_list_of_dicts(self, red_flash_video):
        results = color_detector.detect_color_bursts(
            red_flash_video, burst_threshold=1.5,
        )
        for e in results:
            assert isinstance(e, dict)
            assert isinstance(e["timestamp"], float)
            assert isinstance(e["score"], float)
            assert isinstance(e["dominant_color"], str)


class TestMergeColorCandidates:
    def test_empty_candidates(self):
        assert color_detector._merge_color_candidates([], 2.0) == []

    def test_merge_within_cooldown(self):
        candidates = [
            {"timestamp": 1.0, "score": 10.0, "dominant_color": "red"},
            {"timestamp": 1.5, "score": 20.0, "dominant_color": "red"},
        ]
        merged = color_detector._merge_color_candidates(candidates, 2.0)
        assert len(merged) == 1
        assert merged[0]["score"] == 20.0

    def test_no_merge_outside_cooldown(self):
        candidates = [
            {"timestamp": 1.0, "score": 10.0, "dominant_color": "red"},
            {"timestamp": 5.0, "score": 15.0, "dominant_color": "gold"},
        ]
        merged = color_detector._merge_color_candidates(candidates, 2.0)
        assert len(merged) == 2
