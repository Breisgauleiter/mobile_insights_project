import os
import sys
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

# Add parent dir to path so we can import stats_extractor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import stats_extractor


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
    """A short static black video (640×360, 60 frames, 30 FPS)."""
    base = np.zeros((360, 640, 3), dtype=np.uint8)
    frames = [base.copy() for _ in range(60)]  # 2 seconds at 30 FPS
    path = str(tmp_path / "simple.mp4")
    _create_test_video(path, frames, fps=30.0)
    return path


class TestCropTimerRegion:
    def test_crop_dimensions_standard_frame(self):
        """Crop should cover the correct percentage of the frame."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        crop = stats_extractor._crop_timer_region(frame)
        expected_h = int(360 * (stats_extractor._TIMER_Y_END - stats_extractor._TIMER_Y_START))
        expected_w = int(640 * (stats_extractor._TIMER_X_END - stats_extractor._TIMER_X_START))
        assert crop.shape[0] == expected_h
        assert crop.shape[1] == expected_w

    def test_crop_is_bgr(self):
        """Crop retains all three color channels."""
        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        crop = stats_extractor._crop_timer_region(frame)
        assert crop.ndim == 3
        assert crop.shape[2] == 3

    def test_crop_center_content(self):
        """A pixel placed inside the timer region appears in the crop."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        # Place a red pixel at top-center (y=0, x=50%)
        cx = int(640 * 0.50)
        frame[0, cx] = [0, 0, 255]
        crop = stats_extractor._crop_timer_region(frame)
        # The pixel should be present somewhere in the crop
        assert np.any(np.all(crop == [0, 0, 255], axis=2))


class TestPreprocessForOcr:
    def test_output_is_grayscale(self):
        """Preprocessed image should be single-channel."""
        region = np.zeros((29, 154, 3), dtype=np.uint8)
        processed = stats_extractor._preprocess_for_ocr(region)
        assert processed.ndim == 2

    def test_output_is_2x_upscaled(self):
        """Preprocessed image should be 2× the input dimensions."""
        region = np.zeros((29, 154, 3), dtype=np.uint8)
        processed = stats_extractor._preprocess_for_ocr(region)
        assert processed.shape == (58, 308)

    def test_white_pixels_pass_threshold(self):
        """White pixels (255) in input should remain white after threshold."""
        region = np.full((10, 10, 3), 255, dtype=np.uint8)
        processed = stats_extractor._preprocess_for_ocr(region)
        assert np.all(processed == 255)

    def test_dark_pixels_become_black(self):
        """Dark pixels (value < 160) should become black after threshold."""
        region = np.full((10, 10, 3), 100, dtype=np.uint8)
        processed = stats_extractor._preprocess_for_ocr(region)
        assert np.all(processed == 0)


class TestParseTimerText:
    def test_standard_format(self):
        assert stats_extractor._parse_timer_text("02:47") == 167

    def test_single_digit_minutes(self):
        assert stats_extractor._parse_timer_text("2:47") == 167

    def test_zero_zero(self):
        assert stats_extractor._parse_timer_text("00:00") == 0

    def test_embedded_in_text(self):
        assert stats_extractor._parse_timer_text("Time: 05:30 elapsed") == 330

    def test_invalid_seconds_above_59(self):
        assert stats_extractor._parse_timer_text("01:61") is None

    def test_empty_string(self):
        assert stats_extractor._parse_timer_text("") is None

    def test_no_timer_in_text(self):
        assert stats_extractor._parse_timer_text("Hello World") is None

    def test_non_printable_chars_around_timer(self):
        """Non-ASCII chars surrounding a valid timer should not prevent parsing."""
        assert stats_extractor._parse_timer_text("\x00 03:15 \x01") == 195


class TestSecondsToMmss:
    def test_zero(self):
        assert stats_extractor._seconds_to_mmss(0) == "00:00"

    def test_one_minute(self):
        assert stats_extractor._seconds_to_mmss(60) == "01:00"

    def test_mixed(self):
        assert stats_extractor._seconds_to_mmss(147) == "02:27"

    def test_large_value(self):
        assert stats_extractor._seconds_to_mmss(3600) == "60:00"


class TestGamePhase:
    def test_early_game_start(self):
        assert stats_extractor._game_phase(0) == "early"

    def test_early_game_boundary(self):
        assert stats_extractor._game_phase(300) == "early"

    def test_mid_game(self):
        assert stats_extractor._game_phase(301) == "mid"

    def test_mid_game_boundary(self):
        assert stats_extractor._game_phase(900) == "mid"

    def test_late_game(self):
        assert stats_extractor._game_phase(901) == "late"


class TestGetGameTimeAt:
    def _make_timeline(self) -> list[dict]:
        return [
            {"video_time": 10.0, "game_time": "00:00", "game_seconds": 0, "phase": "early"},
            {"video_time": 20.0, "game_time": "00:10", "game_seconds": 10, "phase": "early"},
            {"video_time": 30.0, "game_time": "00:20", "game_seconds": 20, "phase": "early"},
        ]

    def test_empty_timeline(self):
        assert stats_extractor.get_game_time_at([], 15.0) is None

    def test_before_range(self):
        tl = self._make_timeline()
        assert stats_extractor.get_game_time_at(tl, 5.0) is None

    def test_after_range(self):
        tl = self._make_timeline()
        assert stats_extractor.get_game_time_at(tl, 35.0) is None

    def test_exact_first_entry(self):
        tl = self._make_timeline()
        assert stats_extractor.get_game_time_at(tl, 10.0) == 0

    def test_exact_last_entry(self):
        tl = self._make_timeline()
        assert stats_extractor.get_game_time_at(tl, 30.0) == 20

    def test_interpolation_midpoint(self):
        tl = self._make_timeline()
        # Midpoint between video_time 10 and 20 → game_seconds midpoint 0 and 10 = 5
        result = stats_extractor.get_game_time_at(tl, 15.0)
        assert result == 5


class TestExtractTimerMapping:
    def _inject_pytesseract(self, monkeypatch, return_text: str) -> MagicMock:
        """Inject a mocked pytesseract module into sys.modules."""
        mock_tess = MagicMock()
        mock_tess.image_to_string.return_value = return_text
        monkeypatch.setitem(sys.modules, "pytesseract", mock_tess)
        return mock_tess

    def test_detects_timer(self, simple_video, monkeypatch):
        self._inject_pytesseract(monkeypatch, "02:47\n")
        result = stats_extractor.extract_timer_mapping(simple_video, sample_interval=1.0 / 30)
        assert len(result["timeline"]) >= 1
        assert result["timeline"][0]["game_seconds"] == 167
        assert result["timeline"][0]["game_time"] == "02:47"

    def test_no_timer_returns_empty_timeline(self, simple_video, monkeypatch):
        self._inject_pytesseract(monkeypatch, "")
        result = stats_extractor.extract_timer_mapping(simple_video)
        assert result["timeline"] == []
        assert result["game_start_video_time"] is None
        assert result["game_end_video_time"] is None

    def test_garbled_text_returns_empty(self, simple_video, monkeypatch):
        self._inject_pytesseract(monkeypatch, "||||| @@@@ ####\n???\n")
        result = stats_extractor.extract_timer_mapping(simple_video)
        assert result["timeline"] == []

    def test_game_start_detected(self, simple_video, monkeypatch):
        self._inject_pytesseract(monkeypatch, "00:05\n")
        result = stats_extractor.extract_timer_mapping(simple_video, sample_interval=1.0 / 30)
        assert result["game_start_video_time"] is not None
        assert isinstance(result["game_start_video_time"], float)

    def test_game_end_detected(self, simple_video, monkeypatch):
        self._inject_pytesseract(monkeypatch, "10:30\n")
        result = stats_extractor.extract_timer_mapping(simple_video, sample_interval=1.0 / 30)
        assert result["game_end_video_time"] is not None
        assert isinstance(result["game_end_video_time"], float)

    def test_phase_in_timeline(self, simple_video, monkeypatch):
        self._inject_pytesseract(monkeypatch, "01:00\n")
        result = stats_extractor.extract_timer_mapping(simple_video, sample_interval=1.0 / 30)
        assert len(result["timeline"]) >= 1
        assert result["timeline"][0]["phase"] == "early"

    def test_returns_dict_structure(self, simple_video, monkeypatch):
        self._inject_pytesseract(monkeypatch, "")
        result = stats_extractor.extract_timer_mapping(simple_video)
        assert isinstance(result, dict)
        assert "timeline" in result
        assert "game_start_video_time" in result
        assert "game_end_video_time" in result
        assert isinstance(result["timeline"], list)

    def test_file_not_found_raises(self, monkeypatch):
        mock_tess = MagicMock()
        monkeypatch.setitem(sys.modules, "pytesseract", mock_tess)
        with pytest.raises(FileNotFoundError):
            stats_extractor.extract_timer_mapping("/nonexistent/video.mp4")

    def test_import_error_when_pytesseract_missing(self, simple_video, monkeypatch):
        """ImportError should be raised when pytesseract is not available."""
        monkeypatch.delitem(sys.modules, "pytesseract", raising=False)
        monkeypatch.setitem(sys.modules, "pytesseract", None)  # type: ignore[arg-type]
        with pytest.raises((ImportError, TypeError)):
            stats_extractor.extract_timer_mapping(simple_video)

    def test_zero_sample_interval_raises(self, simple_video, monkeypatch):
        self._inject_pytesseract(monkeypatch, "")
        with pytest.raises(ValueError, match="sample_interval"):
            stats_extractor.extract_timer_mapping(simple_video, sample_interval=0)

    def test_negative_sample_interval_raises(self, simple_video, monkeypatch):
        self._inject_pytesseract(monkeypatch, "")
        with pytest.raises(ValueError, match="sample_interval"):
            stats_extractor.extract_timer_mapping(simple_video, sample_interval=-1.0)

    def test_missing_tesseract_binary_raises_runtime_error(self, simple_video, monkeypatch):
        """TesseractNotFoundError from pytesseract is re-raised as RuntimeError."""
        mock_tess = MagicMock()

        class MockTesseractNotFoundError(Exception):
            pass

        mock_tess.TesseractNotFoundError = MockTesseractNotFoundError
        mock_tess.image_to_string.side_effect = MockTesseractNotFoundError("tesseract not found")
        monkeypatch.setitem(sys.modules, "pytesseract", mock_tess)

        with pytest.raises(RuntimeError, match="Tesseract binary not found"):
            stats_extractor.extract_timer_mapping(simple_video, sample_interval=1.0 / 30)
