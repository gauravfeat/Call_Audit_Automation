"""
Unit Tests for Speech-to-Text Transcription Module (Milestone 3).

All tests mock the underlying Whisper model to avoid downloading heavy weights
during automated test execution.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from app.transcription.model import ModelLoadError, WhisperModelManager
from app.transcription.transcriber import AudioTranscriber, TranscriptionError
from app.transcription.transcript_schema import Transcript, TranscriptSegment, WordTimestamp


# -----------------------------------------------------------------------------
# 1. Schema Validation Tests
# -----------------------------------------------------------------------------

def test_transcript_schema_valid():
    """Verify standard valid Transcript construction and helper properties."""
    words = [
        WordTimestamp(word="Hello", start=0.5, end=0.9, probability=0.98),
        WordTimestamp(word="sir", start=1.0, end=1.3, probability=0.95),
    ]
    segment = TranscriptSegment(
        segment_id=0,
        start=0.5,
        end=1.3,
        text="Hello sir",
        words=words,
    )
    transcript = Transcript(
        recording_id="call_test_01",
        source_file="call_test_01.wav",
        language="en",
        language_probability=0.965,
        duration=15.0,
        model="small",
        processing_time_seconds=1.25,
        segments=[segment],
    )

    assert transcript.recording_id == "call_test_01"
    assert transcript.language == "en"
    assert transcript.full_text == "Hello sir"
    assert transcript.total_words == 2
    assert len(transcript.segments[0].words) == 2


def test_transcript_timestamp_validation():
    """Ensure validation error when segment end timestamp precedes start timestamp."""
    with pytest.raises(ValidationError, match="end timestamp .* cannot be before start"):
        TranscriptSegment(
            segment_id=0,
            start=5.0,
            end=3.0,  # Invalid: end < start
            text="Invalid segment",
        )

    with pytest.raises(ValidationError, match="end timestamp .* cannot be before start"):
        WordTimestamp(
            word="test",
            start=2.0,
            end=1.0,  # Invalid: end < start
        )


def test_language_probability_bounds():
    """Ensure validation error when language probability is out of [0.0, 1.0]."""
    with pytest.raises(ValidationError, match="Language probability must be between 0.0 and 1.0"):
        Transcript(
            recording_id="test",
            source_file="test.wav",
            language="hi",
            language_probability=1.5,  # Invalid: > 1.0
            duration=10.0,
            model="small",
            processing_time_seconds=0.5,
            segments=[],
        )


# -----------------------------------------------------------------------------
# 2. File and Error Handling Tests
# -----------------------------------------------------------------------------

def test_missing_audio_file(tmp_path: Path):
    """Verify FileNotFoundError when target audio file does not exist."""
    transcriber = AudioTranscriber(model=MagicMock())
    missing = tmp_path / "non_existent.wav"
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        transcriber.transcribe_file(missing)


def test_empty_audio_file(tmp_path: Path):
    """Verify TranscriptionError when input audio is 0 bytes."""
    empty_wav = tmp_path / "empty.wav"
    empty_wav.write_bytes(b"")  # 0-byte file

    transcriber = AudioTranscriber(model=MagicMock())
    with pytest.raises(TranscriptionError, match="empty or zero-byte"):
        transcriber.transcribe_file(empty_wav)


def test_transcription_runtime_failure_handling(tmp_path: Path):
    """Verify that model execution exceptions are caught and wrapped in TranscriptionError."""
    sample_wav = tmp_path / "sample.wav"
    sample_wav.write_bytes(b"dummy wav data")

    mock_model = MagicMock()
    mock_model.transcribe.side_effect = RuntimeError("CUDA out of memory")

    transcriber = AudioTranscriber(model=mock_model)
    with pytest.raises(TranscriptionError, match="CUDA out of memory"):
        transcriber.transcribe_file(sample_wav)


# -----------------------------------------------------------------------------
# 3. Mocked Model Inference & Output Generation Tests
# -----------------------------------------------------------------------------

def test_transcription_with_mocked_model(tmp_path: Path):
    """
    Test end-to-end transcription flow using a mocked Faster-Whisper model.
    Verifies Hinglish token preservation, word timestamps, and schema output.
    """
    sample_wav = tmp_path / "call_sample.wav"
    sample_wav.write_bytes(b"dummy wav content header and pcm samples")

    # Construct mock segment with word timestamps
    mock_word_1 = MagicMock()
    mock_word_1.word = "Namaste"
    mock_word_1.start = 0.5
    mock_word_1.end = 1.0
    mock_word_1.probability = 0.95

    mock_word_2 = MagicMock()
    mock_word_2.word = "Sir"
    mock_word_2.start = 1.1
    mock_word_2.end = 1.4
    mock_word_2.probability = 0.98

    mock_segment = MagicMock()
    mock_segment.start = 0.5
    mock_segment.end = 1.5
    mock_segment.text = "Namaste Sir, HDFC Securities se baat kar raha hoon."
    mock_segment.words = [mock_word_1, mock_word_2]

    # Construct mock info
    mock_info = MagicMock()
    mock_info.language = "hi"
    mock_info.language_probability = 0.92
    mock_info.duration = 45.0

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    transcriber = AudioTranscriber(model=mock_model)
    transcript = transcriber.transcribe_file(sample_wav)

    assert transcript.recording_id == "call_sample"
    assert transcript.source_file == "call_sample.wav"
    assert transcript.language == "hi"
    assert transcript.language_probability == 0.92
    assert transcript.duration == 45.0
    assert len(transcript.segments) == 1
    assert "HDFC Securities" in transcript.segments[0].text
    assert transcript.segments[0].words is not None
    assert len(transcript.segments[0].words) == 2
    assert transcript.segments[0].words[0].word == "Namaste"
    assert transcript.segments[0].words[1].word == "Sir"


def test_json_output_serialization(tmp_path: Path):
    """Test JSON saving and round-trip reloading of Transcript object."""
    segment = TranscriptSegment(
        segment_id=0,
        start=0.0,
        end=2.0,
        text="Hello, thank you for calling.",
    )
    transcript = Transcript(
        recording_id="test_rec",
        source_file="test_rec.wav",
        language="en",
        language_probability=0.99,
        duration=2.0,
        model="small",
        processing_time_seconds=0.45,
        segments=[segment],
    )

    out_file = tmp_path / "test_rec.json"
    transcript.save_json(out_file)

    assert out_file.is_file()
    loaded = Transcript.from_json(out_file)
    assert loaded.recording_id == "test_rec"
    assert loaded.full_text == "Hello, thank you for calling."
    assert loaded.segments[0].text == "Hello, thank you for calling."


# -----------------------------------------------------------------------------
# 4. Sarvam Saaras v4 Primary STT Integration Tests
# -----------------------------------------------------------------------------

def test_audio_transcriber_default_provider_is_sarvam():
    """Verify AudioTranscriber defaults to Sarvam Saaras v4 with codemix mode."""
    transcriber = AudioTranscriber()
    assert transcriber.provider == "sarvam"
    assert transcriber.mode == "codemix"


def test_audio_transcriber_with_mocked_sarvam_adapter(tmp_path: Path):
    """Verify transcription flow when using Sarvam as primary provider."""
    from app.benchmark.schema import ASRResult, BenchmarkSegment

    sample_wav = tmp_path / "sarvam_call.wav"
    sample_wav.write_bytes(b"dummy audio pcm data")

    mock_asr_result = ASRResult(
        recording_id="sarvam_call",
        source_file="sarvam_call.wav",
        input_audio_type="processed_wav_16khz",
        model_name="sarvam-saaras-v4",
        model_version="saaras:v4",
        transcription_mode="codemix",
        detected_language="hi-IN",
        language_probability=0.985,
        transcript="Hello Saurabh बोल रहा हूँ Securities से।",
        segments=[
            BenchmarkSegment(
                segment_id=0,
                start=0.5,
                end=3.5,
                text="Hello Saurabh बोल रहा हूँ Securities से।",
            )
        ],
        processing_time_seconds=2.45,
        api_cost_inr=0.0375,
        status="SUCCESS",
    )

    mock_adapter = MagicMock()
    mock_adapter.transcribe.return_value = mock_asr_result

    transcriber = AudioTranscriber(provider="sarvam", mode="codemix", adapter=mock_adapter)
    transcript = transcriber.transcribe_file(sample_wav, reuse_existing=False)

    assert transcript.recording_id == "sarvam_call"
    assert transcript.provider == "sarvam"
    assert transcript.model == "sarvam-saaras-v4:saaras:v4"
    assert transcript.transcription_mode == "codemix"
    assert transcript.language == "hi-IN"
    assert transcript.language_probability == 0.985
    assert transcript.api_cost_inr == 0.0375
    assert len(transcript.segments) == 1
    assert "Saurabh बोल रहा हूँ" in transcript.segments[0].text


def test_audio_transcriber_sarvam_runtime_failure(tmp_path: Path):
    """Verify TranscriptionError when Sarvam adapter returns status FAILED."""
    from app.benchmark.schema import ASRResult

    sample_wav = tmp_path / "failed_call.wav"
    sample_wav.write_bytes(b"dummy audio data")

    mock_failed_result = ASRResult(
        recording_id="failed_call",
        source_file="failed_call.wav",
        input_audio_type="processed_wav_16khz",
        model_name="sarvam-saaras-v4",
        model_version="saaras:v4",
        transcription_mode="codemix",
        transcript="",
        processing_time_seconds=0.0,
        status="FAILED",
        error_details="Sarvam API Rate Limit Exceeded",
    )

    mock_adapter = MagicMock()
    mock_adapter.transcribe.return_value = mock_failed_result

    transcriber = AudioTranscriber(provider="sarvam", adapter=mock_adapter)
    with pytest.raises(TranscriptionError, match="Rate Limit Exceeded"):
        transcriber.transcribe_file(sample_wav, reuse_existing=False)


def test_transcript_from_asr_result_conversion():
    """Verify conversion of benchmark ASRResult into canonical Transcript."""
    from app.benchmark.schema import ASRResult, BenchmarkSegment

    asr_res = ASRResult(
        recording_id="rec_conversion_test",
        source_file="rec_conversion_test.wav",
        input_audio_type="processed_wav_16khz",
        model_name="sarvam-saaras-v4",
        model_version="saaras:v4",
        transcription_mode="codemix",
        detected_language="hi-IN",
        language_probability=0.99,
        transcript="Tata Motors margin shortfall",
        segments=[
            BenchmarkSegment(segment_id=0, start=1.0, end=4.5, text="Tata Motors margin shortfall")
        ],
        processing_time_seconds=3.1,
        api_cost_inr=0.05,
        status="SUCCESS",
        provider_metadata={"job_id": "test_job_123"},
    )

    transcript = Transcript.from_asr_result(asr_res, duration=15.0)
    assert transcript.recording_id == "rec_conversion_test"
    assert transcript.provider == "sarvam"
    assert transcript.model == "sarvam-saaras-v4:saaras:v4"
    assert transcript.transcription_mode == "codemix"
    assert transcript.duration == 15.0
    assert transcript.total_words == 4
    assert transcript.provider_metadata == {"job_id": "test_job_123"}

