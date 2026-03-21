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
        results = highlight_detector.detect_highlights(
            static_video, threshold=15.0, enable_audio=False, enable_color=False,
        )
        assert results == []

    def test_activity_video_detects_flash(self, activity_video):
        results = highlight_detector.detect_highlights(
            activity_video, threshold=10.0, enable_audio=False, enable_color=False,
        )
        assert len(results) >= 1
        # The flash is at frame 15, so around 0.5s at 30fps
        assert any(0.0 <= h["timestamp"] <= 1.0 for h in results)
        assert all("score" in h and "timestamp" in h for h in results)

    def test_threshold_controls_sensitivity(self, activity_video):
        low = highlight_detector.detect_highlights(
            activity_video, threshold=1.0, enable_audio=False, enable_color=False,
        )
        high = highlight_detector.detect_highlights(
            activity_video, threshold=200.0, enable_audio=False, enable_color=False,
        )
        assert len(low) >= len(high)

    def test_max_highlights_limits_output(self, activity_video):
        results = highlight_detector.detect_highlights(
            activity_video, threshold=1.0, max_highlights=1,
            enable_audio=False, enable_color=False,
        )
        assert len(results) <= 1

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            highlight_detector.detect_highlights(
                "/nonexistent/video.mp4", enable_audio=False, enable_color=False,
            )

    def test_returns_list_of_dicts(self, activity_video):
        results = highlight_detector.detect_highlights(
            activity_video, threshold=10.0, enable_audio=False, enable_color=False,
        )
        assert isinstance(results, list)
        for h in results:
            assert isinstance(h, dict)
            assert isinstance(h["timestamp"], float)
            assert isinstance(h["score"], float)
            assert "type" in h
            assert "sources" in h

    def test_skip_frames_reduces_candidates(self, activity_video):
        full = highlight_detector.detect_highlights(
            activity_video, threshold=1.0, enable_audio=False, enable_color=False,
        )
        skipped = highlight_detector.detect_highlights(
            activity_video, threshold=1.0, skip_frames=5,
            enable_audio=False, enable_color=False,
        )
        # With skip_frames, we process fewer frames so may find fewer or equal candidates
        assert isinstance(skipped, list)
        assert len(skipped) <= len(full)

    def test_max_width_still_detects_flash(self, activity_video):
        results = highlight_detector.detect_highlights(
            activity_video, threshold=10.0, max_width=40,
            enable_audio=False, enable_color=False,
        )
        assert len(results) >= 1
        assert any(0.0 <= h["timestamp"] <= 1.0 for h in results)

    def test_skip_frames_zero_is_noop(self, activity_video):
        normal = highlight_detector.detect_highlights(
            activity_video, threshold=10.0, enable_audio=False, enable_color=False,
        )
        explicit = highlight_detector.detect_highlights(
            activity_video, threshold=10.0, skip_frames=0,
            enable_audio=False, enable_color=False,
        )
        assert len(normal) == len(explicit)


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
            "--no-audio",
            "--no-color",
        ]
        try:
            highlight_detector.main()
        finally:
            sys.argv = old_argv

        with open(output_path) as f:
            data = json.load(f)
        assert "highlights" in data
        assert isinstance(data["highlights"], list)


class TestCombineEvents:
    def test_empty_events(self):
        assert highlight_detector._combine_events([], 2.0) == []

    def test_merges_nearby_events(self):
        events = [
            {"timestamp": 1.0, "score": 10.0, "type": "action", "sources": ["frame_diff"]},
            {"timestamp": 1.5, "score": 20.0, "type": "kill", "sources": ["audio"]},
        ]
        combined = highlight_detector._combine_events(events, 2.0)
        assert len(combined) == 1
        assert combined[0]["type"] == "kill"  # audio type takes priority
        assert "frame_diff" in combined[0]["sources"]
        assert "audio" in combined[0]["sources"]

    def test_keeps_separate_events(self):
        events = [
            {"timestamp": 1.0, "score": 10.0, "type": "action", "sources": ["frame_diff"]},
            {"timestamp": 10.0, "score": 15.0, "type": "kill", "sources": ["audio"]},
        ]
        combined = highlight_detector._combine_events(events, 2.0)
        assert len(combined) == 2

    def test_score_boosted_on_merge(self):
        events = [
            {"timestamp": 1.0, "score": 50.0, "type": "action", "sources": ["frame_diff"]},
            {"timestamp": 1.2, "score": 60.0, "type": "effect", "sources": ["color"]},
        ]
        combined = highlight_detector._combine_events(events, 2.0)
        assert len(combined) == 1
        # Score should be boosted: max(50, 60) * 1.2 = 72.0
        assert combined[0]["score"] == 72.0

    def test_no_boost_single_source(self):
        events = [
            {"timestamp": 1.0, "score": 50.0, "type": "action", "sources": ["frame_diff"]},
            {"timestamp": 1.2, "score": 60.0, "type": "action", "sources": ["frame_diff"]},
        ]
        combined = highlight_detector._combine_events(events, 2.0)
        assert len(combined) == 1
        # Same source, no boost
        assert combined[0]["score"] == 60.0

    def test_score_capped_at_100(self):
        events = [
            {"timestamp": 1.0, "score": 90.0, "type": "action", "sources": ["frame_diff"]},
            {"timestamp": 1.1, "score": 95.0, "type": "kill", "sources": ["audio"]},
        ]
        combined = highlight_detector._combine_events(events, 2.0)
        assert combined[0]["score"] <= 100.0


class TestMultiSignalIntegration:
    """Integration tests for the combined pipeline with mocked detectors."""

    def test_combined_pipeline_merges_sources(self, activity_video, monkeypatch):
        """Mock audio/color detectors and verify the combined pipeline."""

        def mock_audio(video_path, whisper_model="tiny", ffmpeg_timeout=300):
            return [
                {"timestamp": 0.5, "score": 80.0, "type": "kill", "text": "slain"},
            ]

        def mock_color(video_path, **kwargs):
            return [
                {"timestamp": 0.6, "score": 40.0, "dominant_color": "red"},
            ]

        monkeypatch.setattr(highlight_detector, "detect_audio_events", mock_audio)
        monkeypatch.setattr(highlight_detector, "detect_color_bursts", mock_color)

        results = highlight_detector.detect_highlights(
            activity_video, threshold=10.0,
            enable_audio=True, enable_color=True,
        )
        assert len(results) >= 1
        # Should have merged sources from frame_diff, audio, color
        top = results[0]
        assert "type" in top
        assert "sources" in top
        assert isinstance(top["sources"], list)

    def test_audio_failure_does_not_break_pipeline(self, activity_video, monkeypatch):
        """If audio raises RuntimeError, pipeline still returns frame-diff results."""

        def mock_audio_fail(video_path, whisper_model="tiny", ffmpeg_timeout=300):
            raise RuntimeError("Whisper model not found")

        monkeypatch.setattr(highlight_detector, "detect_audio_events", mock_audio_fail)

        results = highlight_detector.detect_highlights(
            activity_video, threshold=10.0,
            enable_audio=True, enable_color=False,
        )
        assert len(results) >= 1
        assert all("frame_diff" in h.get("sources", []) for h in results)
