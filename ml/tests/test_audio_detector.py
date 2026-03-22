import os
import sys
import wave

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

    def test_returns_empty_without_whisper(self, tmp_path, monkeypatch):
        """When Whisper is not available, should return empty list."""
        monkeypatch.setattr(audio_detector, "WHISPER_AVAILABLE", False)
        dummy = tmp_path / "dummy.mp4"
        dummy.touch()
        result = audio_detector.detect_audio_events(str(dummy))
        assert result == []


class TestMockWhisperOutput:
    """Integration tests using mocked Whisper transcription segments."""

    def test_mock_transcription_produces_events(self, tmp_path, monkeypatch):
        """detect_audio_events should classify events from mocked Whisper output."""
        dummy = tmp_path / "game.mp4"
        dummy.touch()

        mock_segments = [
            {"start": 1.5, "text": "SAVAGE!"},
            {"start": 5.0, "text": "An enemy has been slain"},
            {"start": 10.0, "text": "Triple Kill!"},
        ]

        monkeypatch.setattr(audio_detector, "WHISPER_AVAILABLE", True)
        monkeypatch.setattr(
            audio_detector,
            "_extract_audio",
            lambda path, timeout=300: str(dummy),
        )
        monkeypatch.setattr(
            audio_detector,
            "_transcribe_audio",
            lambda audio_path, model_name: mock_segments,
        )

        events = audio_detector.detect_audio_events(str(dummy))
        assert len(events) == 3
        types = [e["type"] for e in events]
        assert "savage" in types
        assert "kill" in types
        assert "triple_kill" in types

    def test_mock_transcription_timestamps(self, tmp_path, monkeypatch):
        """Timestamps from Whisper segments are preserved in events."""
        dummy = tmp_path / "game.mp4"
        dummy.touch()

        mock_segments = [{"start": 42.5, "text": "First Blood!"}]

        monkeypatch.setattr(audio_detector, "WHISPER_AVAILABLE", True)
        monkeypatch.setattr(
            audio_detector,
            "_extract_audio",
            lambda path, timeout=300: str(dummy),
        )
        monkeypatch.setattr(
            audio_detector,
            "_transcribe_audio",
            lambda audio_path, model_name: mock_segments,
        )

        events = audio_detector.detect_audio_events(str(dummy))
        assert len(events) == 1
        assert events[0]["timestamp"] == 42.5
        assert events[0]["type"] == "first_blood"

    def test_mock_transcription_no_match_skipped(self, tmp_path, monkeypatch):
        """Segments without known phrases produce no events."""
        dummy = tmp_path / "game.mp4"
        dummy.touch()

        mock_segments = [
            {"start": 1.0, "text": "some background chatter"},
            {"start": 3.0, "text": "unintelligible audio"},
        ]

        monkeypatch.setattr(audio_detector, "WHISPER_AVAILABLE", True)
        monkeypatch.setattr(
            audio_detector,
            "_extract_audio",
            lambda path, timeout=300: str(dummy),
        )
        monkeypatch.setattr(
            audio_detector,
            "_transcribe_audio",
            lambda audio_path, model_name: mock_segments,
        )

        events = audio_detector.detect_audio_events(str(dummy))
        assert events == []

    def test_debug_flag_prints_segments(self, tmp_path, monkeypatch, capsys):
        """With debug=True, Whisper segments are printed to stderr."""
        dummy = tmp_path / "game.mp4"
        dummy.touch()

        mock_segments = [{"start": 2.0, "text": "Savage!"}]

        monkeypatch.setattr(audio_detector, "WHISPER_AVAILABLE", True)
        monkeypatch.setattr(
            audio_detector,
            "_extract_audio",
            lambda path, timeout=300: str(dummy),
        )
        monkeypatch.setattr(
            audio_detector,
            "_transcribe_audio",
            lambda audio_path, model_name: mock_segments,
        )

        audio_detector.detect_audio_events(str(dummy), debug=True)
        captured = capsys.readouterr()
        assert "[debug]" in captured.err
        assert "Savage!" in captured.err

    def test_debug_flag_off_no_stderr(self, tmp_path, monkeypatch, capsys):
        """With debug=False (default), no debug lines are printed to stderr."""
        dummy = tmp_path / "game.mp4"
        dummy.touch()

        mock_segments = [{"start": 1.0, "text": "Maniac"}]

        monkeypatch.setattr(audio_detector, "WHISPER_AVAILABLE", True)
        monkeypatch.setattr(
            audio_detector,
            "_extract_audio",
            lambda path, timeout=300: str(dummy),
        )
        monkeypatch.setattr(
            audio_detector,
            "_transcribe_audio",
            lambda audio_path, model_name: mock_segments,
        )

        audio_detector.detect_audio_events(str(dummy), debug=False)
        captured = capsys.readouterr()
        assert "[debug]" not in captured.err


class TestDetectVolumeEvents:
    """Tests for the RMS energy-based volume detector."""

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            audio_detector.detect_volume_events("/nonexistent/video.mp4")

    def test_returns_empty_on_extraction_failure(self, tmp_path, monkeypatch):
        dummy = tmp_path / "dummy.mp4"
        dummy.touch()
        monkeypatch.setattr(
            audio_detector, "_extract_audio", lambda path, timeout=300: None
        )
        result = audio_detector.detect_volume_events(str(dummy))
        assert result == []

    def _make_wav(self, path: str, samples: list[int], sample_rate: int = 16000) -> None:
        """Write a 16-bit mono WAV file."""
        import struct

        raw = struct.pack(f"<{len(samples)}h", *samples)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(raw)

    def test_loud_moment_detected(self, tmp_path, monkeypatch):
        """A WAV with one loud burst should produce at least one volume event."""
        sample_rate = 16000
        # Silence for first second, loud burst in second second
        samples = [0] * sample_rate + [32000] * sample_rate
        wav_path = str(tmp_path / "test.wav")
        self._make_wav(wav_path, samples, sample_rate)

        monkeypatch.setattr(
            audio_detector,
            "_extract_audio",
            lambda path, timeout=300: wav_path,
        )

        dummy = tmp_path / "dummy.mp4"
        dummy.touch()
        events = audio_detector.detect_volume_events(
            str(dummy), rms_threshold=0.5, cooldown_sec=0.5
        )
        assert len(events) >= 1
        assert events[0]["type"] == "loud_moment"
        assert "rms" in events[0]
        assert "score" in events[0]

    def test_silent_audio_no_events(self, tmp_path, monkeypatch):
        """Completely silent WAV should produce no volume events."""
        sample_rate = 16000
        samples = [0] * (sample_rate * 2)
        wav_path = str(tmp_path / "silent.wav")
        self._make_wav(wav_path, samples, sample_rate)

        monkeypatch.setattr(
            audio_detector,
            "_extract_audio",
            lambda path, timeout=300: wav_path,
        )

        dummy = tmp_path / "dummy.mp4"
        dummy.touch()
        events = audio_detector.detect_volume_events(
            str(dummy), rms_threshold=0.01, cooldown_sec=0.5
        )
        assert events == []

    def test_cooldown_limits_events(self, tmp_path, monkeypatch):
        """Cooldown period prevents consecutive events within the window."""
        sample_rate = 16000
        # 4 seconds of loud audio
        samples = [30000] * (sample_rate * 4)
        wav_path = str(tmp_path / "loud.wav")
        self._make_wav(wav_path, samples, sample_rate)

        monkeypatch.setattr(
            audio_detector,
            "_extract_audio",
            lambda path, timeout=300: wav_path,
        )

        dummy = tmp_path / "dummy.mp4"
        dummy.touch()
        # With 2-second cooldown on 4 seconds of audio, expect at most 2 events
        events = audio_detector.detect_volume_events(
            str(dummy), rms_threshold=0.5, cooldown_sec=2.0, hop_sec=0.5
        )
        assert len(events) <= 2
