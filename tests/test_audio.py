"""
Unit and Integration Tests for Audio Preprocessing and Quality Pipeline (M2).
"""

import math
import struct
import wave
from pathlib import Path

import pytest

from app.audio.exceptions import (
    AudioPreprocessingError,
    FFmpegNotFoundError,
    UnsupportedAudioFormatError,
)
from app.audio.ffmpeg_utils import get_ffmpeg_path, get_ffprobe_path, verify_ffmpeg_installation
from app.audio.metadata import extract_metadata
from app.audio.preprocess import TARGET_CHANNELS, TARGET_SAMPLE_RATE, preprocess_audio
from app.audio.processor import AudioProcessor
from app.audio.quality import analyze_audio_quality


def create_synthetic_wav(
    file_path: Path,
    duration_seconds: float = 1.0,
    sample_rate: int = 8000,
    channels: int = 1,
    frequency_hz: float = 440.0,
    amplitude: float = 0.5,
) -> Path:
    """
    Generate a pure synthetic sine-wave WAV file using Python's standard wave module.
    """
    num_frames = int(duration_seconds * sample_rate)
    with wave.open(str(file_path), "w") as wav_out:
        wav_out.setnchannels(channels)
        wav_out.setsampwidth(2)  # 16-bit
        wav_out.setframerate(sample_rate)

        frames = bytearray()
        for i in range(num_frames):
            t = float(i) / sample_rate
            val = int(amplitude * 32767.0 * math.sin(2.0 * math.pi * frequency_hz * t))
            for _ in range(channels):
                frames.extend(struct.pack("<h", val))

        wav_out.writeframes(frames)

    return file_path


def create_silent_wav(
    file_path: Path,
    duration_seconds: float = 2.0,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """
    Generate a completely silent WAV file for silence detection testing.
    """
    num_frames = int(duration_seconds * sample_rate)
    with wave.open(str(file_path), "w") as wav_out:
        wav_out.setnchannels(channels)
        wav_out.setsampwidth(2)
        wav_out.setframerate(sample_rate)
        wav_out.writeframes(b"\x00" * (num_frames * channels * 2))

    return file_path


# -----------------------------------------------------------------------------
# 1. FFmpeg Binary Verification Tests
# -----------------------------------------------------------------------------

def test_ffmpeg_binary_availability():
    """Verify that FFmpeg and FFprobe binaries are found and executable."""
    results = verify_ffmpeg_installation()
    assert "ffmpeg" in results
    assert "ffprobe" in results
    assert Path(get_ffmpeg_path()).is_file()
    assert Path(get_ffprobe_path()).is_file()


# -----------------------------------------------------------------------------
# 2. Metadata Extraction Tests
# -----------------------------------------------------------------------------

def test_metadata_extraction_mono_wav(tmp_path: Path):
    """Verify extraction of audio metadata for a synthetic mono WAV file."""
    wav_path = tmp_path / "test_mono.wav"
    create_synthetic_wav(wav_path, duration_seconds=1.5, sample_rate=8000, channels=1)

    meta = extract_metadata(wav_path)
    assert meta.filename == "test_mono.wav"
    assert meta.sample_rate == 8000
    assert meta.channels == 1
    assert meta.channel_layout == "mono"
    assert meta.bit_depth == 16
    assert meta.duration_seconds >= 1.4 and meta.duration_seconds <= 1.6
    assert meta.file_size_bytes > 0


def test_metadata_extraction_stereo_wav(tmp_path: Path):
    """Verify metadata extraction handles multi-channel audio correctly."""
    wav_path = tmp_path / "test_stereo.wav"
    create_synthetic_wav(wav_path, duration_seconds=1.0, sample_rate=44100, channels=2)

    meta = extract_metadata(wav_path)
    assert meta.channels == 2
    assert meta.sample_rate == 44100
    assert meta.channel_layout == "stereo"


def test_missing_audio_file(tmp_path: Path):
    """Verify FileNotFoundError when querying a non-existent file."""
    non_existent = tmp_path / "does_not_exist.mp3"
    with pytest.raises(FileNotFoundError):
        extract_metadata(non_existent)


def test_unsupported_audio_format(tmp_path: Path):
    """Verify UnsupportedAudioFormatError when file extension is not supported."""
    txt_file = tmp_path / "report.txt"
    txt_file.write_text("not an audio file", encoding="utf-8")
    with pytest.raises(UnsupportedAudioFormatError):
        extract_metadata(txt_file)


# -----------------------------------------------------------------------------
# 3. Audio Preprocessing & Standardization Tests
# -----------------------------------------------------------------------------

def test_audio_preprocessing_standardization(tmp_path: Path):
    """
    Verify conversion to 16,000 Hz, mono channel, 16-bit PCM WAV.
    """
    raw_audio = tmp_path / "raw_telephony.wav"
    out_dir = tmp_path / "processed"
    # Create an 8000Hz stereo input
    create_synthetic_wav(raw_audio, duration_seconds=2.0, sample_rate=8000, channels=2)

    converted_path = preprocess_audio(raw_audio, output_dir=out_dir)

    assert converted_path.is_file()
    assert converted_path.suffix == ".wav"
    assert converted_path.name == "raw_telephony.wav"

    # Verify converted properties with metadata extraction
    processed_meta = extract_metadata(converted_path)
    assert processed_meta.sample_rate == TARGET_SAMPLE_RATE  # 16000 Hz
    assert processed_meta.channels == TARGET_CHANNELS        # 1 (mono)
    assert processed_meta.bit_depth == 16                   # 16-bit PCM
    assert processed_meta.codec_name == "pcm_s16le"

    # Verify original file was preserved untouched
    assert raw_audio.is_file()
    orig_meta = extract_metadata(raw_audio)
    assert orig_meta.sample_rate == 8000
    assert orig_meta.channels == 2


def test_preprocessing_fails_on_same_source_dest(tmp_path: Path):
    """Verify safety check prevents overwriting original file with output."""
    raw_audio = tmp_path / "audio.wav"
    create_synthetic_wav(raw_audio, duration_seconds=1.0)

    with pytest.raises(AudioPreprocessingError, match="cannot be identical"):
        preprocess_audio(raw_audio, output_dir=tmp_path, output_filename="audio.wav")


# -----------------------------------------------------------------------------
# 4. Audio Quality Analysis Tests
# -----------------------------------------------------------------------------

def test_audio_quality_normal_signal(tmp_path: Path):
    """Verify quality analysis metrics for a normal synthetic tone."""
    tone_file = tmp_path / "normal_tone.wav"
    create_synthetic_wav(tone_file, duration_seconds=2.0, sample_rate=16000, amplitude=0.5)

    metrics = analyze_audio_quality(tone_file, known_duration=2.0)
    assert metrics.mean_volume_db < 0.0
    assert metrics.peak_amplitude_db <= 0.0
    assert not metrics.is_clipped
    assert metrics.clipping_samples_count == 0


def test_audio_quality_silence_detection(tmp_path: Path):
    """Verify silence detection and silence percentage calculation on silent audio."""
    silent_file = tmp_path / "silent_test.wav"
    create_silent_wav(silent_file, duration_seconds=3.0, sample_rate=16000)

    metrics = analyze_audio_quality(silent_file, known_duration=3.0)
    assert metrics.silence_percentage > 80.0
    assert metrics.total_silence_seconds > 2.0
    # Should contain silence warning
    assert any("silence" in w.lower() for w in metrics.warnings)


# -----------------------------------------------------------------------------
# 5. Full Pipeline Orchestration Tests
# -----------------------------------------------------------------------------

def test_audio_processor_pipeline_e2e(tmp_path: Path):
    """
    Test end-to-end AudioProcessor workflow on synthetic recordings:
    discovery -> metadata -> quality -> conversion -> JSON/CSV reports.
    """
    input_dir = tmp_path / "input_recordings"
    output_dir = tmp_path / "output_processed"
    input_dir.mkdir()
    output_dir.mkdir()

    # Create 2 mock test audio files
    create_synthetic_wav(input_dir / "sample_1.wav", duration_seconds=1.5, sample_rate=8000)
    create_synthetic_wav(input_dir / "sample_2.wav", duration_seconds=2.0, sample_rate=16000)

    processor = AudioProcessor()
    summary = processor.process_all(input_dir=input_dir, output_dir=output_dir)

    assert summary.total_recordings_discovered == 2
    assert summary.successfully_processed == 2
    assert summary.failed_recordings == 0
    assert len(summary.reports) == 2

    # Check generated files
    assert Path(summary.report_path_json).is_file()
    assert Path(summary.report_path_csv).is_file()

    # Check processed WAV outputs exist
    for r in summary.reports:
        assert r.processing_status in ("SUCCESS", "WARNING")
        assert r.processed_filename is not None
        proc_file = output_dir / r.processed_filename
        assert proc_file.is_file()
        assert r.processed_metadata.sample_rate == 16000
        assert r.processed_metadata.channels == 1
