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
    """A short static video with white gold regions at the top of the frame."""
    # 640×360 is a typical mobile resolution
    base = np.zeros((360, 640, 3), dtype=np.uint8)
    # Fill top 8% of left area with white (simulating team1 gold area)
    base[0:29, 0:256] = 255
    # Fill top 8% of right area with white (simulating team2 gold area)
    base[0:29, 384:640] = 255
    frames = [base.copy() for _ in range(60)]  # 2 seconds at 30 FPS
    path = str(tmp_path / "simple.mp4")
    _create_test_video(path, frames, fps=30.0)
    return path


class TestCropGoldRegion:
    def test_crop_dimensions_standard_frame(self):
        """Crop dimensions should match the fractional bounds."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        crop = stats_extractor._crop_gold_region(
            frame,
            stats_extractor._TEAM1_X_START,
            stats_extractor._TEAM1_X_END,
            stats_extractor._GOLD_Y_START,
            stats_extractor._GOLD_Y_END,
        )
        expected_h = int(
            360 * (stats_extractor._GOLD_Y_END - stats_extractor._GOLD_Y_START)
        )
        expected_w = int(
            640 * (stats_extractor._TEAM1_X_END - stats_extractor._TEAM1_X_START)
        )
        assert crop.shape[0] == expected_h
        assert crop.shape[1] == expected_w

    def test_crop_is_bgr(self):
        """Crop retains all three colour channels."""
        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        crop = stats_extractor._crop_gold_region(frame, 0.15, 0.40, 0.00, 0.08)
        assert crop.ndim == 3
        assert crop.shape[2] == 3

    def test_crop_returns_correct_region(self):
        """A red pixel at a known location should appear in the team1 crop."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        # x=200, y=5 is inside team1 gold region (x: 96..256, y: 0..28)
        frame[5, 200] = [0, 0, 255]
        crop = stats_extractor._crop_gold_region(
            frame,
            stats_extractor._TEAM1_X_START,
            stats_extractor._TEAM1_X_END,
            stats_extractor._GOLD_Y_START,
            stats_extractor._GOLD_Y_END,
        )
        # Crop x origin = int(640 * 0.15) = 96  →  crop col = 200 - 96 = 104
        assert list(crop[5, 104]) == [0, 0, 255]


class TestPreprocessForOcr:
    def test_output_is_grayscale(self):
        """Preprocessed image should be single-channel."""
        region = np.zeros((72, 256, 3), dtype=np.uint8)
        processed = stats_extractor._preprocess_for_ocr(region)
        assert processed.ndim == 2

    def test_output_is_2x_upscaled(self):
        """Preprocessed image should be 2× the input dimensions."""
        region = np.zeros((72, 256, 3), dtype=np.uint8)
        processed = stats_extractor._preprocess_for_ocr(region)
        assert processed.shape == (144, 512)

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


class TestParseGoldValue:
    def test_simple_number(self):
        assert stats_extractor._parse_gold_value("1234") == 1234

    def test_number_with_comma(self):
        assert stats_extractor._parse_gold_value("1,234") == 1234

    def test_number_with_dot(self):
        assert stats_extractor._parse_gold_value("1.234") == 1234

    def test_empty_returns_none(self):
        assert stats_extractor._parse_gold_value("") is None

    def test_non_numeric_returns_none(self):
        assert stats_extractor._parse_gold_value("abc") is None

    def test_value_too_large_returns_none(self):
        assert stats_extractor._parse_gold_value("200000") is None

    def test_extracts_first_match(self):
        assert stats_extractor._parse_gold_value("gold: 5678 pts") == 5678

    def test_non_printable_stripped(self):
        # \x00 (NUL) is removed entirely, joining "12" and "34" into "1234"
        assert stats_extractor._parse_gold_value("12\x0034") == 1234


class TestExtractGoldTimeline:
    def _inject_pytesseract(self, monkeypatch, texts: list) -> MagicMock:
        """Inject a mocked pytesseract module into sys.modules."""
        mock_tess = MagicMock()
        mock_tess.image_to_string.side_effect = texts
        monkeypatch.setitem(sys.modules, "pytesseract", mock_tess)
        return mock_tess

    def test_extracts_gold_values(self, simple_video, monkeypatch):
        """Valid OCR text should produce entries with correct gold values."""
        # 60 frames @ 30 FPS, sample_fps=1.0 → frame_step=30 → 2 sampled frames
        # Each frame makes 2 image_to_string calls (team1, team2) → 4 total
        self._inject_pytesseract(monkeypatch, ["1234", "5678"] * 10)
        result = stats_extractor.extract_gold_timeline(simple_video)
        assert len(result) >= 1
        first = result[0]
        assert first["team1_gold"] == 1234
        assert first["team2_gold"] == 5678
        assert first["gold_diff"] == -4444

    def test_empty_ocr_returns_empty(self, simple_video, monkeypatch):
        """Empty OCR output for both teams should produce no timeline entries."""
        self._inject_pytesseract(monkeypatch, ["", ""] * 10)
        result = stats_extractor.extract_gold_timeline(simple_video)
        assert result == []

    def test_file_not_found_raises(self, monkeypatch):
        mock_tess = MagicMock()
        monkeypatch.setitem(sys.modules, "pytesseract", mock_tess)
        with pytest.raises(FileNotFoundError):
            stats_extractor.extract_gold_timeline("/nonexistent/video.mp4")

    def test_invalid_sample_fps_raises(self, simple_video, monkeypatch):
        mock_tess = MagicMock()
        monkeypatch.setitem(sys.modules, "pytesseract", mock_tess)
        with pytest.raises(ValueError, match="sample_fps"):
            stats_extractor.extract_gold_timeline(simple_video, sample_fps=0)

    def test_invalid_max_samples_raises(self, simple_video, monkeypatch):
        mock_tess = MagicMock()
        monkeypatch.setitem(sys.modules, "pytesseract", mock_tess)
        with pytest.raises(ValueError, match="max_samples"):
            stats_extractor.extract_gold_timeline(simple_video, max_samples=0)

    def test_import_error_when_pytesseract_missing(self, simple_video, monkeypatch):
        """ImportError (or TypeError) raised when pytesseract sentinel is None."""
        monkeypatch.setitem(sys.modules, "pytesseract", None)  # type: ignore[arg-type]
        with pytest.raises((ImportError, TypeError)):
            stats_extractor.extract_gold_timeline(simple_video)

    def test_missing_tesseract_binary_raises_runtime_error(
        self, simple_video, monkeypatch
    ):
        """TesseractNotFoundError from pytesseract is re-raised as RuntimeError."""
        mock_tess = MagicMock()

        class TesseractNotFoundError(Exception):
            pass

        mock_tess.TesseractNotFoundError = TesseractNotFoundError
        mock_tess.image_to_string.side_effect = TesseractNotFoundError("not found")
        monkeypatch.setitem(sys.modules, "pytesseract", mock_tess)

        with pytest.raises(RuntimeError, match="Tesseract binary not found"):
            stats_extractor.extract_gold_timeline(simple_video, sample_fps=30.0)

    def test_result_has_correct_keys(self, simple_video, monkeypatch):
        """Each timeline entry must contain all expected keys."""
        self._inject_pytesseract(monkeypatch, ["1000", "2000"] * 10)
        result = stats_extractor.extract_gold_timeline(simple_video)
        assert len(result) >= 1
        assert {"time", "team1_gold", "team2_gold", "gold_diff"}.issubset(
            result[0].keys()
        )

    def test_gold_diff_is_correct(self, simple_video, monkeypatch):
        """gold_diff should equal team1_gold minus team2_gold."""
        self._inject_pytesseract(monkeypatch, ["3000", "1000"] * 10)
        result = stats_extractor.extract_gold_timeline(simple_video)
        assert len(result) >= 1
        assert result[0]["gold_diff"] == 2000
