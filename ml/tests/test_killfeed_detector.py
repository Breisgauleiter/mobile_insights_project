import os
import sys
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

# Add parent dir to path so we can import killfeed_detector
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import killfeed_detector


def _create_test_video(path: str, frames: list[np.ndarray], fps: float = 30.0) -> None:
    """Create a small test video from a list of frames."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


@pytest.fixture
def simple_video(tmp_path):
    """A short static video with a white kill-feed region in the top-right."""
    # 640×360 is a typical mobile resolution
    base = np.zeros((360, 640, 3), dtype=np.uint8)
    # Fill top-right region with white (simulating kill-feed text area)
    base[0:72, 384:640] = 255
    frames = [base.copy() for _ in range(60)]  # 2 seconds at 30 FPS
    path = str(tmp_path / "simple.mp4")
    _create_test_video(path, frames, fps=30.0)
    return path


class TestCropKillfeed:
    def test_crop_dimensions_standard_frame(self):
        """Crop should be 40% wide and 20% tall of the original frame."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        crop = killfeed_detector._crop_killfeed(frame)
        expected_h = int(360 * (killfeed_detector._KILLFEED_Y_END - killfeed_detector._KILLFEED_Y_START))
        expected_w = int(640 * (killfeed_detector._KILLFEED_X_END - killfeed_detector._KILLFEED_X_START))
        assert crop.shape[0] == expected_h
        assert crop.shape[1] == expected_w

    def test_crop_is_bgr(self):
        """Crop retains all three color channels."""
        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        crop = killfeed_detector._crop_killfeed(frame)
        assert crop.ndim == 3
        assert crop.shape[2] == 3

    def test_crop_top_right_content(self):
        """Pixels from the top-right corner appear in the crop."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        # Mark the top-right corner red
        frame[0, 639] = [0, 0, 255]
        crop = killfeed_detector._crop_killfeed(frame)
        # The top-right corner pixel of the crop should be red
        assert list(crop[0, -1]) == [0, 0, 255]


class TestPreprocessForOcr:
    def test_output_is_grayscale(self):
        """Preprocessed image should be single-channel."""
        region = np.zeros((72, 256, 3), dtype=np.uint8)
        processed = killfeed_detector._preprocess_for_ocr(region)
        assert processed.ndim == 2

    def test_output_is_2x_upscaled(self):
        """Preprocessed image should be 2× the input dimensions."""
        region = np.zeros((72, 256, 3), dtype=np.uint8)
        processed = killfeed_detector._preprocess_for_ocr(region)
        assert processed.shape == (144, 512)

    def test_white_pixels_pass_threshold(self):
        """White pixels (255) in input should remain white after threshold."""
        region = np.full((10, 10, 3), 255, dtype=np.uint8)
        processed = killfeed_detector._preprocess_for_ocr(region)
        assert np.all(processed == 255)

    def test_dark_pixels_become_black(self):
        """Dark pixels (value < 160) should become black after threshold."""
        region = np.full((10, 10, 3), 100, dtype=np.uint8)
        processed = killfeed_detector._preprocess_for_ocr(region)
        assert np.all(processed == 0)


class TestParseKillText:
    def test_killed_verb(self):
        result = killfeed_detector._parse_kill_text("Layla killed Tigreal")
        assert result is not None
        killer, victim, assist = result
        assert killer == "Layla"
        assert victim == "Tigreal"
        assert assist is None

    def test_arrow_notation(self):
        result = killfeed_detector._parse_kill_text("Alucard -> Franco")
        assert result is not None
        killer, victim, _ = result
        assert killer == "Alucard"
        assert victim == "Franco"

    def test_gt_notation(self):
        result = killfeed_detector._parse_kill_text("Fanny > Eudora")
        assert result is not None
        killer, victim, _ = result
        assert killer == "Fanny"
        assert victim == "Eudora"

    def test_slain_verb(self):
        result = killfeed_detector._parse_kill_text("Chou slain Zilong")
        assert result is not None
        assert result[0] == "Chou"
        assert result[1] == "Zilong"

    def test_empty_string_returns_none(self):
        assert killfeed_detector._parse_kill_text("") is None

    def test_short_text_returns_none(self):
        assert killfeed_detector._parse_kill_text("Hi") is None

    def test_no_pattern_returns_none(self):
        assert killfeed_detector._parse_kill_text("Random garbage text here") is None

    def test_non_printable_chars_stripped(self):
        # Inject a non-ASCII character; should still parse correctly
        text = "Layla\x00 killed Tigreal"
        result = killfeed_detector._parse_kill_text(text)
        assert result is not None
        assert result[0] == "Layla"

    def test_case_insensitive(self):
        result = killfeed_detector._parse_kill_text("Gusion KILLED Hanabi")
        assert result is not None
        assert result[0] == "Gusion"
        assert result[1] == "Hanabi"

    def test_names_too_short_returns_none(self):
        # Single-char names should not match (minimum length 2)
        result = killfeed_detector._parse_kill_text("A killed B")
        assert result is None


class TestDetectKillfeed:
    def _inject_pytesseract(self, monkeypatch, return_text: str) -> MagicMock:
        """Inject a mocked pytesseract module into sys.modules."""
        mock_tess = MagicMock()
        mock_tess.image_to_string.return_value = return_text
        monkeypatch.setitem(sys.modules, "pytesseract", mock_tess)
        return mock_tess

    def test_detects_single_kill(self, simple_video, monkeypatch):
        self._inject_pytesseract(monkeypatch, "Layla killed Tigreal\n")
        events = killfeed_detector.detect_killfeed(simple_video, sample_fps=30.0)
        assert len(events) >= 1
        assert events[0]["killer"] == "Layla"
        assert events[0]["victim"] == "Tigreal"
        assert "time" in events[0]
        assert isinstance(events[0]["time"], float)

    def test_deduplication_suppresses_repeat_kills(self, simple_video, monkeypatch):
        """Same (killer, victim) pair within dedup_window should only appear once."""
        self._inject_pytesseract(monkeypatch, "Layla killed Tigreal\n")
        # sample_fps=30 means every frame gets OCR → many frames, but dedup_window=60 suppresses repeats
        events = killfeed_detector.detect_killfeed(
            simple_video, sample_fps=30.0, dedup_window=60.0
        )
        kill_events = [e for e in events if e["killer"] == "Layla" and e["victim"] == "Tigreal"]
        assert len(kill_events) == 1

    def test_no_ocr_text_returns_empty(self, simple_video, monkeypatch):
        """Empty OCR output should produce no kill events."""
        self._inject_pytesseract(monkeypatch, "")
        events = killfeed_detector.detect_killfeed(simple_video)
        assert events == []

    def test_garbled_text_returns_empty(self, simple_video, monkeypatch):
        """Non-kill OCR text should be ignored."""
        self._inject_pytesseract(monkeypatch, "||||| @@@@ ####\n???\n")
        events = killfeed_detector.detect_killfeed(simple_video)
        assert events == []

    def test_file_not_found_raises(self, monkeypatch):
        mock_tess = MagicMock()
        monkeypatch.setitem(sys.modules, "pytesseract", mock_tess)
        with pytest.raises(FileNotFoundError):
            killfeed_detector.detect_killfeed("/nonexistent/video.mp4")

    def test_import_error_when_pytesseract_missing(self, simple_video, monkeypatch):
        """ImportError should be raised when pytesseract is not available."""
        # Remove pytesseract from sys.modules so the import inside the function fails
        monkeypatch.delitem(sys.modules, "pytesseract", raising=False)
        # Also patch builtins.__import__ is complex; instead we block the import via sys.modules sentinel
        monkeypatch.setitem(sys.modules, "pytesseract", None)  # type: ignore[arg-type]
        with pytest.raises((ImportError, TypeError)):
            killfeed_detector.detect_killfeed(simple_video)

    def test_returns_list_of_dicts(self, simple_video, monkeypatch):
        self._inject_pytesseract(monkeypatch, "Gusion killed Hanabi\n")
        events = killfeed_detector.detect_killfeed(simple_video, sample_fps=30.0)
        assert isinstance(events, list)
        for e in events:
            assert isinstance(e, dict)
            assert "time" in e
            assert "killer" in e
            assert "victim" in e

    def test_max_events_limit(self, simple_video, monkeypatch):
        """max_events parameter should cap the returned list."""
        self._inject_pytesseract(monkeypatch, "Hero killed Victim\n")
        events = killfeed_detector.detect_killfeed(
            simple_video, sample_fps=30.0, dedup_window=0.0, max_events=2
        )
        assert len(events) <= 2

    def test_events_ordered_by_time(self, simple_video, monkeypatch):
        """Kill events should be ordered ascending by time."""
        self._inject_pytesseract(monkeypatch, "Layla killed Tigreal\n")
        events = killfeed_detector.detect_killfeed(
            simple_video, sample_fps=30.0, dedup_window=0.0
        )
        times = [e["time"] for e in events]
        assert times == sorted(times)

    def test_zero_sample_fps_raises_value_error(self, simple_video, monkeypatch):
        """sample_fps=0 must raise ValueError before processing begins."""
        self._inject_pytesseract(monkeypatch, "")
        with pytest.raises(ValueError, match="sample_fps"):
            killfeed_detector.detect_killfeed(simple_video, sample_fps=0)

    def test_negative_sample_fps_raises_value_error(self, simple_video, monkeypatch):
        """Negative sample_fps must raise ValueError."""
        self._inject_pytesseract(monkeypatch, "")
        with pytest.raises(ValueError, match="sample_fps"):
            killfeed_detector.detect_killfeed(simple_video, sample_fps=-1.0)

    def test_zero_max_events_raises_value_error(self, simple_video, monkeypatch):
        """max_events=0 must raise ValueError."""
        self._inject_pytesseract(monkeypatch, "")
        with pytest.raises(ValueError, match="max_events"):
            killfeed_detector.detect_killfeed(simple_video, max_events=0)

    def test_negative_max_events_raises_value_error(self, simple_video, monkeypatch):
        """Negative max_events must raise ValueError."""
        self._inject_pytesseract(monkeypatch, "")
        with pytest.raises(ValueError, match="max_events"):
            killfeed_detector.detect_killfeed(simple_video, max_events=-5)

    def test_missing_tesseract_binary_raises_runtime_error(self, simple_video, monkeypatch):
        """TesseractNotFoundError from pytesseract is re-raised as RuntimeError."""
        mock_tess = MagicMock()
        # TesseractNotFoundError must be a real exception class
        class TesseractNotFoundError(Exception):
            pass

        mock_tess.TesseractNotFoundError = TesseractNotFoundError
        mock_tess.image_to_string.side_effect = TesseractNotFoundError("tesseract not found")
        monkeypatch.setitem(sys.modules, "pytesseract", mock_tess)

        with pytest.raises(RuntimeError, match="Tesseract binary not found"):
            killfeed_detector.detect_killfeed(simple_video, sample_fps=30.0)
