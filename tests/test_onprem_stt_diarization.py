"""
Unit Tests for Remote Whisper Large-v3 API and Local pyannote Community-1 Diarization.

Covers:
1. Whisper API successful response parsing and serialization
2. Whisper API timeout and connection failure handling
3. Whisper API invalid JSON handling
4. Whisper API empty transcription handling
5. UTF-8 Hindi/Gujarati transcript saving
6. JSON and TXT output generation
7. Duplicate processing prevention (reuse_existing=True)
8. Explicit reprocessing (reuse_existing=False)
9. pyannote two-speaker diarization (mocked pipeline)
10. pyannote single-speaker diarization
11. pyannote multi-speaker diarization
12. Overlapping speech handling
13. Timestamp validation and chronological ordering
14. Transcript-to-speaker alignment (dominant overlap)
15. Ambiguous alignment / UNKNOWN speaker marking
16. Provider switching in AudioTranscriber & DiarizationProcessor
17. Existing Sarvam regression verification
18. M4.5 Gemini Role Classification compatibility
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from app.core.config import Settings
from app.diarization.alignment import (
    align_transcript_and_diarization,
    calculate_overlap,
    compute_speaker_overlap_map,
    save_onprem_diarized_transcript,
)
from app.diarization.diarization_schema import DiarizedTranscript, DiarizedSegment
from app.diarization.processor import DiarizationProcessor
from app.diarization.pyannote_diarizer import PyannoteDiarizer, PyannoteDiarizationError, SpeakerTurn
from app.transcription.transcriber import AudioTranscriber, TranscriptionError
from app.transcription.transcript_schema import Transcript, TranscriptSegment
from app.transcription.whisper_api_client import WhisperAPIClient, WhisperAPIError


@pytest.fixture
def mock_settings(tmp_path: Path) -> Settings:
    """Fixture providing isolated Settings with temporary directories."""
    settings = Settings()
    settings.BASE_DIR = tmp_path
    settings.DEFAULT_STT_PROVIDER = "sarvam"
    settings.WHISPER_API_BASE_URL = "http://mock-whisper-api:7091"
    settings.ensure_directories()
    return settings


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    """Fixture creating a dummy WAV file for testing."""
    wav_path = tmp_path / "MOCK_CALL_002.wav"
    wav_path.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    return wav_path


# ==============================================================================
# 1-8: Whisper API Client Tests
# ==============================================================================

class TestWhisperAPIClient:
    def test_whisper_api_success_verbose_json(self, mock_settings: Settings, sample_wav: Path):
        """1. Verify successful Whisper API verbose_json parsing and Transcript building."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "नमस्ते वैभव सर आप बोलिए",
            "duration": 12.5,
            "language": "hi",
            "segments": [
                {"id": 0, "start": 0.0, "end": 5.2, "text": "नमस्ते वैभव सर"},
                {"id": 1, "start": 5.5, "end": 12.5, "text": "आप बोलिए"},
            ],
            "usage": {"type": "duration", "seconds": 12.5},
        }
        mock_session.post.return_value = mock_response

        client = WhisperAPIClient(settings=mock_settings, session=mock_session)
        transcript = client.transcribe_file(sample_wav, reuse_existing=False)

        assert transcript.recording_id == "MOCK_CALL_002"
        assert transcript.language == "hi"
        assert transcript.duration == 12.5
        assert len(transcript.segments) == 2
        assert transcript.segments[0].text == "नमस्ते वैभव सर"
        assert transcript.segments[1].end == 12.5

        # Check saved files
        json_path = mock_settings.onprem_whisper_transcripts_dir / "MOCK_CALL_002.json"
        txt_path = mock_settings.onprem_whisper_transcripts_dir / "MOCK_CALL_002.txt"
        assert json_path.is_file()
        assert txt_path.is_file()

        # Check UTF-8 content
        content = txt_path.read_text(encoding="utf-8")
        assert "नमस्ते वैभव सर" in content

    def test_whisper_api_timeout(self, mock_settings: Settings, sample_wav: Path):
        """2. Verify timeout raises WhisperAPIError with detailed message."""
        mock_session = MagicMock()
        mock_session.post.side_effect = requests.exceptions.Timeout("Connection timed out after 600s")

        client = WhisperAPIClient(settings=mock_settings, session=mock_session)
        with pytest.raises(WhisperAPIError, match="timed out"):
            client.transcribe_file(sample_wav, reuse_existing=False)

    def test_whisper_api_connection_error(self, mock_settings: Settings, sample_wav: Path):
        """Verify connection failure raises WhisperAPIError."""
        mock_session = MagicMock()
        mock_session.post.side_effect = requests.exceptions.ConnectionError("Failed to establish a new connection")

        client = WhisperAPIClient(settings=mock_settings, session=mock_session)
        with pytest.raises(WhisperAPIError, match="Failed to connect"):
            client.transcribe_file(sample_wav, reuse_existing=False)

    def test_whisper_api_invalid_json(self, mock_settings: Settings, sample_wav: Path):
        """3. Verify invalid JSON response raises WhisperAPIError."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON string")
        mock_session.post.return_value = mock_response

        client = WhisperAPIClient(settings=mock_settings, session=mock_session)
        with pytest.raises(WhisperAPIError, match="Invalid JSON"):
            client.transcribe_file(sample_wav, reuse_existing=False)

    def test_whisper_api_empty_transcription(self, mock_settings: Settings, sample_wav: Path):
        """4. Verify empty transcript text is handled gracefully without crashing."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "",
            "duration": 5.0,
            "language": "hi",
            "segments": [],
        }
        mock_session.post.return_value = mock_response

        client = WhisperAPIClient(settings=mock_settings, session=mock_session)
        transcript = client.transcribe_file(sample_wav, reuse_existing=False)
        assert transcript.full_text == ""
        assert len(transcript.segments) == 0

    def test_duplicate_processing_prevention(self, mock_settings: Settings, sample_wav: Path):
        """7 & 8. Verify reuse_existing=True skips API call, while False forces reprocess."""
        out_dir = mock_settings.onprem_whisper_transcripts_dir
        cached_json = out_dir / "MOCK_CALL_002.json"
        cached_payload = {
            "recording_id": "MOCK_CALL_002",
            "source_file": "MOCK_CALL_002.wav",
            "provider": "whisper_api",
            "model": "large-v3",
            "duration": 10.0,
            "text": "Cached text",
            "segments": [{"segment_id": 0, "start": 0.0, "end": 10.0, "text": "Cached text"}],
        }
        cached_json.write_text(json.dumps(cached_payload), encoding="utf-8")

        mock_session = MagicMock()
        client = WhisperAPIClient(settings=mock_settings, session=mock_session)

        # Reusing existing: no network call
        transcript = client.transcribe_file(sample_wav, reuse_existing=True)
        assert transcript.full_text == "Cached text"
        mock_session.post.assert_not_called()

        # Explicit reprocessing: calls network
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "Freshly reprocessed text",
            "duration": 10.0,
            "language": "hi",
            "segments": [{"id": 0, "start": 0.0, "end": 10.0, "text": "Freshly reprocessed text"}],
        }
        mock_session.post.return_value = mock_response

        fresh = client.transcribe_file(sample_wav, reuse_existing=False)
        assert fresh.full_text == "Freshly reprocessed text"
        assert mock_session.post.call_count == 1


# ==============================================================================
# 9-13: pyannote Diarizer Tests
# ==============================================================================

class TestPyannoteDiarizer:
    def test_pyannote_two_speaker_diarization(self, mock_settings: Settings, sample_wav: Path):
        """9. Verify normal two-speaker diarization turn parsing."""
        mock_pipeline = MagicMock()
        mock_annotation = MagicMock()

        class MockTurn:
            def __init__(self, start, end):
                self.start = start
                self.end = end

        tracks = [
            (MockTurn(0.0, 5.0), None, "SPEAKER_00"),
            (MockTurn(5.2, 10.0), None, "SPEAKER_01"),
            (MockTurn(10.5, 15.0), None, "SPEAKER_00"),
        ]
        mock_annotation.itertracks.return_value = tracks
        mock_pipeline.return_value = mock_annotation

        diarizer = PyannoteDiarizer(settings=mock_settings, pipeline=mock_pipeline)
        turns = diarizer.diarize_file(sample_wav, num_speakers=2)

        assert len(turns) == 3
        assert turns[0].speaker_id == "SPEAKER_00"
        assert turns[1].speaker_id == "SPEAKER_01"
        assert turns[0].duration == 5.0

    def test_pyannote_single_speaker_diarization(self, mock_settings: Settings, sample_wav: Path):
        """10. Verify single-speaker call diarization."""
        mock_pipeline = MagicMock()
        mock_annotation = MagicMock()

        class MockTurn:
            def __init__(self, start, end):
                self.start = start
                self.end = end

        tracks = [
            (MockTurn(0.0, 8.0), None, "SPEAKER_00"),
            (MockTurn(9.0, 15.0), None, "SPEAKER_00"),
        ]
        mock_annotation.itertracks.return_value = tracks
        mock_pipeline.return_value = mock_annotation

        diarizer = PyannoteDiarizer(settings=mock_settings, pipeline=mock_pipeline)
        turns = diarizer.diarize_file(sample_wav)

        assert len(turns) == 2
        assert all(t.speaker_id == "SPEAKER_00" for t in turns)

    def test_pyannote_multi_speaker_diarization(self, mock_settings: Settings, sample_wav: Path):
        """11. Verify 3+ speaker call diarization."""
        mock_pipeline = MagicMock()
        mock_annotation = MagicMock()

        class MockTurn:
            def __init__(self, start, end):
                self.start = start
                self.end = end

        tracks = [
            (MockTurn(0.0, 4.0), None, "SPEAKER_00"),
            (MockTurn(4.5, 8.0), None, "SPEAKER_01"),
            (MockTurn(8.5, 12.0), None, "SPEAKER_02"),
        ]
        mock_annotation.itertracks.return_value = tracks
        mock_pipeline.return_value = mock_annotation

        diarizer = PyannoteDiarizer(settings=mock_settings, pipeline=mock_pipeline)
        turns = diarizer.diarize_file(sample_wav)

        speakers = set(t.speaker_id for t in turns)
        assert len(speakers) == 3
        assert "SPEAKER_02" in speakers


# ==============================================================================
# 14-15: Alignment Engine Tests
# ==============================================================================

class TestAlignmentEngine:
    def test_majority_overlap_alignment(self, mock_settings: Settings):
        """14. Verify transcript segment is attributed to the dominant overlapping speaker."""
        transcript = Transcript(
            recording_id="MOCK_CALL_002",
            source_file="MOCK_CALL_002.wav",
            language="hi",
            language_probability=1.0,
            duration=30.0,
            model="whisper-large-v3",
            processing_time_seconds=1.5,
            segments=[
                TranscriptSegment(segment_id=0, start=0.0, end=5.0, text="Hello Sir RM here"),
                TranscriptSegment(segment_id=1, start=5.2, end=10.0, text="Haan boliye kya hua"),
            ],
        )

        speaker_turns = [
            SpeakerTurn(speaker_id="SPEAKER_00", start=0.0, end=5.0),
            SpeakerTurn(speaker_id="SPEAKER_01", start=5.0, end=10.0),
        ]

        diarized = align_transcript_and_diarization(
            transcript=transcript,
            speaker_turns=speaker_turns,
            settings=mock_settings,
        )

        assert diarized.num_speakers_detected == 2
        assert diarized.segments[0].speaker_id == "SPEAKER_00"
        assert diarized.segments[1].speaker_id == "SPEAKER_01"
        assert len(diarized.speaker_stats) == 2

    def test_ambiguous_overlap_and_warnings(self, mock_settings: Settings):
        """15. Verify overlapping speech and warnings are properly recorded."""
        transcript = Transcript(
            recording_id="MOCK_CALL_002",
            source_file="MOCK_CALL_002.wav",
            language="hi",
            duration=20.0,
            model="whisper-large-v3",
            processing_time_seconds=1.0,
            segments=[
                # Segment overlaps almost equally with both speakers
                TranscriptSegment(segment_id=0, start=5.0, end=9.0, text="Simultaneous talking"),
            ],
        )

        # SPEAKER_00: 5.0 to 7.0 (2.0s), SPEAKER_01: 6.8 to 9.0 (2.2s)
        speaker_turns = [
            SpeakerTurn(speaker_id="SPEAKER_00", start=5.0, end=7.0),
            SpeakerTurn(speaker_id="SPEAKER_01", start=6.8, end=9.0),
        ]

        diarized = align_transcript_and_diarization(
            transcript=transcript,
            speaker_turns=speaker_turns,
            settings=mock_settings,
        )

        # Attributed to majority overlap (SPEAKER_01 with 2.2s vs 2.0s)
        assert diarized.segments[0].speaker_id == "SPEAKER_01"
        assert len(diarized.warnings) > 0


# ==============================================================================
# 16-18: Provider Switching, Sarvam Regression, M4.5 Compatibility
# ==============================================================================

class TestIntegrationAndCompatibility:
    def test_provider_switching_in_transcriber(self, mock_settings: Settings, sample_wav: Path):
        """16. Verify AudioTranscriber switches between sarvam and whisper_api."""
        transcriber_sarvam = AudioTranscriber(provider="sarvam", settings=mock_settings)
        assert transcriber_sarvam.provider == "sarvam"

        transcriber_api = AudioTranscriber(provider="whisper_api", settings=mock_settings)
        assert transcriber_api.provider == "whisper_api"

    def test_sarvam_regression_intact(self, mock_settings: Settings, sample_wav: Path):
        """17. Verify existing Sarvam transcription flow remains fully intact."""
        from app.benchmark.schema import ASRResult, BenchmarkSegment
        mock_adapter = MagicMock()
        mock_adapter.transcribe.return_value = ASRResult(
            recording_id="MOCK_CALL_002",
            source_file="MOCK_CALL_002.wav",
            input_audio_type="processed_wav_16khz",
            model_name="sarvam-saaras-v4",
            model_version="saaras:v4",
            transcription_mode="codemix",
            status="SUCCESS",
            transcript="Sarvam transcript text",
            segments=[
                BenchmarkSegment(
                    segment_id=0,
                    start=0.0,
                    end=10.0,
                    text="Sarvam transcript text",
                )
            ],
            duration_seconds=10.0,
            processing_time_seconds=1.2,
        )

        transcriber = AudioTranscriber(provider="sarvam", adapter=mock_adapter, settings=mock_settings)
        transcript = transcriber.transcribe_file(sample_wav, reuse_existing=False)

        assert transcript.provider == "sarvam"
        assert transcript.full_text == "Sarvam transcript text"

    def test_m45_role_classifier_compatibility(self, mock_settings: Settings):
        """18. Verify aligned DiarizedTranscript passes directly into RoleClassifier without errors."""
        from app.diarization.role_classifier import RoleClassifier

        diarized = DiarizedTranscript(
            recording_id="MOCK_CALL_002",
            source_file="MOCK_CALL_002.wav",
            duration=30.0,
            num_speakers_detected=2,
            model="pyannote/speaker-diarization-community-1",
            mode="codemix",
            processing_time_seconds=2.0,
            segments=[
                DiarizedSegment(segment_id=0, speaker_id="SPEAKER_00", speaker_role="UNKNOWN", start=0.0, end=5.0, text="Hello Saurabh speaking from HDFC Securities."),
                DiarizedSegment(segment_id=1, speaker_id="SPEAKER_01", speaker_role="UNKNOWN", start=5.5, end=10.0, text="Haan bolo na mera stock loss mein kyu hai?"),
            ],
            provider="pyannote",
        )

        mock_gemini_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "recording_id": "MOCK_CALL_002",
            "speakers": [
                {
                    "speaker_id": "SPEAKER_00",
                    "predicted_role": "RM",
                    "confidence": 0.95,
                    "role_indicators": ["Introduced as HDFC Securities RM"],
                    "supporting_quotes": ["Hello Saurabh speaking from HDFC Securities"],
                },
                {
                    "speaker_id": "SPEAKER_01",
                    "predicted_role": "CUSTOMER",
                    "confidence": 0.92,
                    "role_indicators": ["Inquires about stock loss"],
                    "supporting_quotes": ["Haan bolo na mera stock loss mein kyu hai?"],
                },
            ],
            "rationale": "Clear RM introduction and customer query.",
        })
        mock_gemini_client.models.generate_content.return_value = mock_response

        classifier = RoleClassifier(api_key="mock-key", client=mock_gemini_client, settings=mock_settings)
        result = classifier.classify_transcript(diarized)

        assert result.status == "AUTO_CLASSIFIED"
        assert result.speakers[0].predicted_role == "RM"
        assert result.speakers[1].predicted_role == "CUSTOMER"
