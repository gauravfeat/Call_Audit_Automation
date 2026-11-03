"""
Unit Tests for M4 Speaker Diarization Module.

Tests cover:
- DiarizedSegment validation (start/end constraints, duration, rounding)
- SpeakerStats schema and default role="UNKNOWN"
- DiarizedTranscript schema, properties (speaker_ids, total_words, labeled_transcript), JSON I/O
- SarvamDiarizer helper functions (_normalize_speaker_id, calculate_cost, _compute_speaker_stats)
- SarvamDiarizer parsing of Sarvam Batch API responses
- DiarizationProcessor batch orchestration, caching, error isolation, and summary reporting
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.diarization.diarization_schema import (
    DiarizedSegment,
    DiarizedTranscript,
    SpeakerStats,
)
from app.diarization.sarvam_diarizer import (
    SarvamDiarizer,
    _normalize_speaker_id,
    _get_audio_duration,
)
from app.diarization.processor import DiarizationProcessor


# ======================== DiarizedSegment Tests ========================

class TestDiarizedSegment:
    def test_valid_segment(self):
        seg = DiarizedSegment(
            segment_id=0,
            speaker_id="SPEAKER_00",
            start=0.5,
            end=3.25,
            text="Namaste, main HDFC Securities se bol raha hoon.",
        )
        assert seg.segment_id == 0
        assert seg.speaker_id == "SPEAKER_00"
        assert seg.start == 0.5
        assert seg.end == 3.25
        assert seg.duration == 2.75
        assert "HDFC Securities" in seg.text

    def test_segment_end_before_start_raises_error(self):
        with pytest.raises(ValueError, match="cannot be before start"):
            DiarizedSegment(
                segment_id=0,
                speaker_id="SPEAKER_00",
                start=5.0,
                end=2.0,
                text="Invalid timestamps",
            )

    def test_timestamp_rounding(self):
        seg = DiarizedSegment(
            segment_id=1,
            speaker_id="SPEAKER_01",
            start=1.23456,
            end=4.56789,
            text="Rounding test",
        )
        assert seg.start == 1.235
        assert seg.end == 4.568


# ======================== SpeakerStats Tests ========================

class TestSpeakerStats:
    def test_speaker_stats_creation(self):
        stats = SpeakerStats(
            speaker_id="SPEAKER_00",
            total_segments=10,
            total_words=120,
            total_talk_time_seconds=45.5,
            talk_time_percentage=55.0,
            first_segment_start=0.5,
            last_segment_end=58.2,
        )
        assert stats.speaker_id == "SPEAKER_00"
        assert stats.role == "UNKNOWN"  # Must remain UNKNOWN until M5
        assert stats.total_segments == 10
        assert stats.total_words == 120
        assert stats.total_talk_time_seconds == 45.5
        assert stats.talk_time_percentage == 55.0


# ======================== DiarizedTranscript Tests ========================

class TestDiarizedTranscript:
    def _create_sample_transcript(self) -> DiarizedTranscript:
        seg1 = DiarizedSegment(
            segment_id=0,
            speaker_id="SPEAKER_00",
            start=0.0,
            end=3.0,
            text="Hello, how can I help you?",
        )
        seg2 = DiarizedSegment(
            segment_id=1,
            speaker_id="SPEAKER_01",
            start=3.5,
            end=7.0,
            text="I want to buy some shares.",
        )
        seg3 = DiarizedSegment(
            segment_id=2,
            speaker_id="SPEAKER_00",
            start=7.5,
            end=10.0,
            text="Sure, please share your Demat account details.",
        )
        stats1 = SpeakerStats(
            speaker_id="SPEAKER_00",
            role="UNKNOWN",
            total_segments=2,
            total_words=13,
            total_talk_time_seconds=5.5,
            talk_time_percentage=55.0,
            first_segment_start=0.0,
            last_segment_end=10.0,
        )
        stats2 = SpeakerStats(
            speaker_id="SPEAKER_01",
            role="UNKNOWN",
            total_segments=1,
            total_words=6,
            total_talk_time_seconds=3.5,
            talk_time_percentage=35.0,
            first_segment_start=3.5,
            last_segment_end=7.0,
        )
        return DiarizedTranscript(
            recording_id="rec_001",
            source_file="rec_001.wav",
            duration=10.0,
            num_speakers_detected=2,
            num_speakers_requested=2,
            model="sarvam-saaras-v4:saaras:v4",
            mode="codemix",
            processing_time_seconds=2.5,
            api_cost_inr=0.025,
            language="hi-IN",
            segments=[seg1, seg2, seg3],
            speaker_stats=[stats1, stats2],
            full_transcript="Hello, how can I help you? I want to buy some shares. Sure, please share your Demat account details.",
            status="SUCCESS",
        )

    def test_speaker_ids_property(self):
        dt = self._create_sample_transcript()
        assert dt.speaker_ids == ["SPEAKER_00", "SPEAKER_01"]

    def test_total_words_property(self):
        dt = self._create_sample_transcript()
        assert dt.total_words == 19

    def test_labeled_transcript_property(self):
        dt = self._create_sample_transcript()
        labeled = dt.labeled_transcript
        assert "[SPEAKER_00]: Hello, how can I help you?" in labeled
        assert "[SPEAKER_01]: I want to buy some shares." in labeled
        assert "[SPEAKER_00]: Sure, please share your Demat account details." in labeled

    def test_json_roundtrip(self, tmp_path):
        dt = self._create_sample_transcript()
        out_file = tmp_path / "rec_001.json"
        dt.save_json(out_file)

        assert out_file.exists()
        loaded = DiarizedTranscript.from_json(out_file)
        assert loaded.recording_id == dt.recording_id
        assert loaded.num_speakers_detected == 2
        assert len(loaded.segments) == 3
        assert len(loaded.speaker_stats) == 2
        assert loaded.speaker_stats[0].role == "UNKNOWN"
        assert loaded.segments[1].speaker_id == "SPEAKER_01"


# ======================== SarvamDiarizer Tests ========================

class TestSarvamDiarizerHelpers:
    def test_normalize_speaker_id(self):
        assert _normalize_speaker_id("0") == "SPEAKER_00"
        assert _normalize_speaker_id("1") == "SPEAKER_01"
        assert _normalize_speaker_id("12") == "SPEAKER_12"
        assert _normalize_speaker_id("SPEAKER_00") == "SPEAKER_00"
        assert _normalize_speaker_id("speaker_01") == "SPEAKER_01"
        assert _normalize_speaker_id("rm") == "SPEAKER_RM"

    def test_calculate_cost(self):
        with patch.dict(os.environ, {"SARVAM_API_KEY": "dummy_key"}):
            diarizer = SarvamDiarizer(api_key="dummy_key")
            # 60s = 1 min -> 0.15 INR
            assert diarizer.calculate_cost(60.0) == 0.15
            # 120s = 2 min -> 0.30 INR
            assert diarizer.calculate_cost(120.0) == 0.30

    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Sarvam API key is required"):
                SarvamDiarizer(api_key=None)

    def test_compute_speaker_stats(self):
        with patch.dict(os.environ, {"SARVAM_API_KEY": "dummy_key"}):
            diarizer = SarvamDiarizer(api_key="dummy_key")
            segs = [
                DiarizedSegment(
                    segment_id=0, speaker_id="SPEAKER_00", start=0.0, end=4.0, text="Hello world"
                ),
                DiarizedSegment(
                    segment_id=1, speaker_id="SPEAKER_01", start=4.5, end=9.5, text="Hi there good morning"
                ),
                DiarizedSegment(
                    segment_id=2, speaker_id="SPEAKER_00", start=10.0, end=14.0, text="Welcome to HDFC"
                ),
            ]
            stats = diarizer._compute_speaker_stats(segs, total_duration=20.0)
            assert len(stats) == 2
            s0 = stats[0]
            assert s0.speaker_id == "SPEAKER_00"
            assert s0.role == "UNKNOWN"
            assert s0.total_segments == 2
            assert s0.total_words == 5
            assert s0.total_talk_time_seconds == 8.0
            assert s0.talk_time_percentage == 40.0  # 8 / 20 * 100

            s1 = stats[1]
            assert s1.speaker_id == "SPEAKER_01"
            assert s1.total_segments == 1
            assert s1.total_words == 4
            assert s1.total_talk_time_seconds == 5.0
            assert s1.talk_time_percentage == 25.0  # 5 / 20 * 100

    def test_parse_diarized_output(self, tmp_path):
        with patch.dict(os.environ, {"SARVAM_API_KEY": "dummy_key"}):
            diarizer = SarvamDiarizer(api_key="dummy_key")

            # Create mock downloaded JSON
            mock_data = {
                "transcript": "Hello from full transcript",
                "diarized_transcript": {
                    "entries": [
                        {
                            "transcript": "Namaste HDFC Securities",
                            "start_time_seconds": 0.5,
                            "end_time_seconds": 3.0,
                            "speaker_id": "0",
                        },
                        {
                            "transcript": "Mujhe portfolio check karna hai",
                            "start_time_seconds": 3.2,
                            "end_time_seconds": 6.8,
                            "speaker_id": "1",
                        },
                    ]
                },
                "language_code": "hi-IN",
            }
            json_file = tmp_path / "test_audio.wav.json"
            json_file.write_text(json.dumps(mock_data), encoding="utf-8")

            segs, full_text, lang = diarizer._parse_diarized_output(
                tmp_path, "test_audio.wav"
            )

            assert len(segs) == 2
            assert segs[0].speaker_id == "SPEAKER_00"
            assert segs[0].text == "Namaste HDFC Securities"
            assert segs[1].speaker_id == "SPEAKER_01"
            assert segs[1].text == "Mujhe portfolio check karna hai"
            assert lang == "hi-IN"
            assert "Namaste HDFC Securities" in full_text

    def test_diarize_missing_file(self):
        with patch.dict(os.environ, {"SARVAM_API_KEY": "dummy_key"}):
            diarizer = SarvamDiarizer(api_key="dummy_key")
            result = diarizer.diarize(
                audio_path=Path("non_existent_file.wav"),
                recording_id="missing_rec",
            )
            assert result.status == "FAILED"
            assert "not found" in result.error_details


# ======================== DiarizationProcessor Tests ========================

class TestDiarizationProcessor:
    def test_empty_input_dir(self, tmp_path):
        with patch.dict(os.environ, {"SARVAM_API_KEY": "dummy_key"}):
            proc = DiarizationProcessor(mode="codemix", num_speakers=2)
            summary = proc.process_batch(
                input_dir=tmp_path / "empty_dir",
                output_dir=tmp_path / "out",
            )
            assert summary["total_recordings"] == 0
            assert summary["successful_diarizations"] == 0

    def test_caching_reuses_existing_valid_transcript(self, tmp_path):
        with patch.dict(os.environ, {"SARVAM_API_KEY": "dummy_key"}):
            proc = DiarizationProcessor(mode="codemix", num_speakers=2)

            in_dir = tmp_path / "audio"
            in_dir.mkdir()
            out_dir = tmp_path / "out"
            out_dir.mkdir()

            # Create dummy wav file
            wav_file = in_dir / "rec_cached.wav"
            wav_file.write_bytes(b"RIFF....WAVEfmt ....data....")

            # Create cached diarized JSON
            cached = DiarizedTranscript(
                recording_id="rec_cached",
                source_file="rec_cached.wav",
                duration=12.0,
                num_speakers_detected=2,
                model="sarvam-saaras-v4:saaras:v4",
                mode="codemix",
                processing_time_seconds=1.0,
                segments=[
                    DiarizedSegment(
                        segment_id=0,
                        speaker_id="SPEAKER_00",
                        start=0.0,
                        end=5.0,
                        text="Cached test segment",
                    )
                ],
                speaker_stats=[
                    SpeakerStats(
                        speaker_id="SPEAKER_00",
                        role="UNKNOWN",
                        total_segments=1,
                        total_words=3,
                        total_talk_time_seconds=5.0,
                        talk_time_percentage=100.0,
                        first_segment_start=0.0,
                        last_segment_end=5.0,
                    )
                ],
                full_transcript="Cached test segment",
                status="SUCCESS",
            )
            cached.save_json(out_dir / "rec_cached.json")

            # Mock diarizer so that if it is called, it would fail
            proc.diarizer.diarize = MagicMock(side_effect=RuntimeError("Should not be called!"))

            summary = proc.process_batch(
                input_dir=in_dir,
                output_dir=out_dir,
                reuse_existing=True,
            )

            assert summary["total_recordings"] == 1
            assert summary["cached_results"] == 1
            assert summary["successful_diarizations"] == 1
            proc.diarizer.diarize.assert_not_called()

    def test_error_isolation_continues_batch(self, tmp_path):
        with patch.dict(os.environ, {"SARVAM_API_KEY": "dummy_key"}):
            proc = DiarizationProcessor(mode="codemix", num_speakers=2)

            in_dir = tmp_path / "audio"
            in_dir.mkdir()
            out_dir = tmp_path / "out"
            out_dir.mkdir()

            (in_dir / "rec_01.wav").write_bytes(b"RIFF....WAVEfmt ....data....")
            (in_dir / "rec_02.wav").write_bytes(b"RIFF....WAVEfmt ....data....")

            # First call fails, second succeeds
            mock_fail = DiarizedTranscript(
                recording_id="rec_01",
                source_file="rec_01.wav",
                duration=5.0,
                num_speakers_detected=0,
                model="sarvam-saaras-v4:saaras:v4",
                mode="codemix",
                processing_time_seconds=0.5,
                status="FAILED",
                error_details="Simulated failure",
            )
            mock_ok = DiarizedTranscript(
                recording_id="rec_02",
                source_file="rec_02.wav",
                duration=8.0,
                num_speakers_detected=2,
                model="sarvam-saaras-v4:saaras:v4",
                mode="codemix",
                processing_time_seconds=1.2,
                segments=[
                    DiarizedSegment(
                        segment_id=0,
                        speaker_id="SPEAKER_00",
                        start=0.0,
                        end=4.0,
                        text="Hello",
                    )
                ],
                status="SUCCESS",
            )

            proc.diarizer.diarize = MagicMock(side_effect=[mock_fail, mock_ok])

            summary = proc.process_batch(
                input_dir=in_dir,
                output_dir=out_dir,
                reuse_existing=False,
            )

            assert summary["total_recordings"] == 2
            assert summary["failed_diarizations"] == 1
            assert summary["successful_diarizations"] == 1
            assert (out_dir / "diarization_summary.json").exists()
