import os
import sys

import pytest

# Add parent dir to path so we can import audio_detector
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import audio_detector


class TestClassifyText:
    def test_savage_detected(self):
        result = audio_detector._classify_text("Savage!")
        assert result is not None
        assert result[0] == "savage"
        assert result[1] == 100.0

    def test_maniac_detected(self):
        result = audio_detector._classify_text("Maniac")
        assert result is not None
        assert result[0] == "maniac"

    def test_triple_kill(self):
        result = audio_detector._classify_text("Triple Kill!")
        assert result is not None
        assert result[0] == "triple_kill"

    def test_double_kill(self):
        result = audio_detector._classify_text("Double kill achieved")
        assert result is not None
        assert result[0] == "double_kill"

    def test_first_blood(self):
        result = audio_detector._classify_text("First Blood!")
        assert result is not None
        assert result[0] == "first_blood"

    def test_slain_is_kill(self):
        result = audio_detector._classify_text("An enemy has been slain")
        assert result is not None
        assert result[0] == "kill"

    def test_lord_with_slain(self):
        result = audio_detector._classify_text("The lord has been slain")
        assert result is not None
        assert result[0] == "lord"
        assert result[1] == 85.0

    def test_turtle_with_slain(self):
        result = audio_detector._classify_text("The turtle has been slain")
        assert result is not None
        assert result[0] == "turtle"

    def test_turret_destroyed(self):
        result = audio_detector._classify_text("Your turret has been destroyed")
        assert result is not None
        assert result[0] == "turret"

    def test_victory_is_game_end(self):
        result = audio_detector._classify_text("Victory!")
        assert result is not None
        assert result[0] == "game_end"

    def test_defeat_is_game_end(self):
        result = audio_detector._classify_text("Defeat")
        assert result is not None
        assert result[0] == "game_end"

    def test_no_match_returns_none(self):
        assert audio_detector._classify_text("random noise text") is None

    def test_empty_text_returns_none(self):
        assert audio_detector._classify_text("") is None

    def test_case_insensitive(self):
        result = audio_detector._classify_text("SAVAGE")
        assert result is not None
        assert result[0] == "savage"

    def test_priority_order(self):
        """Savage should match before kill even if text contains both."""
        result = audio_detector._classify_text("savage slain")
        assert result is not None
        assert result[0] == "savage"  # higher priority


class TestDetectAudioEvents:
    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            audio_detector.detect_audio_events("/nonexistent/video.mp4")

    def test_returns_empty_without_whisper(self, tmp_path):
        """When Whisper is not available, should return empty list."""
        if audio_detector.WHISPER_AVAILABLE:
            pytest.skip("Whisper is installed — cannot test fallback")
        # Create a dummy file so FileNotFoundError is not raised
        dummy = tmp_path / "dummy.mp4"
        dummy.touch()
        result = audio_detector.detect_audio_events(str(dummy))
        assert result == []
