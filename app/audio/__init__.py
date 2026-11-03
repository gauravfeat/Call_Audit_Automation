"""
Audio Processing Package for Call Audit Automation.

Provides audio discovery, metadata extraction, quality assessment,
and standardized 16kHz mono WAV preprocessing.
"""

from app.audio.exceptions import (
    AudioMetadataError,
    AudioPreprocessingError,
    AudioProcessingError,
    AudioQualityAnalysisError,
    FFmpegNotFoundError,
    UnsupportedAudioFormatError,
)
from app.audio.ffmpeg_utils import get_ffmpeg_path, get_ffprobe_path, verify_ffmpeg_installation
from app.audio.metadata import AudioMetadata, extract_metadata
from app.audio.preprocess import preprocess_audio
from app.audio.processor import AudioFileReport, AudioPipelineSummary, AudioProcessor
from app.audio.quality import AudioQualityMetrics, analyze_audio_quality

__all__ = [
    "AudioMetadata",
    "extract_metadata",
    "AudioQualityMetrics",
    "analyze_audio_quality",
    "preprocess_audio",
    "AudioProcessor",
    "AudioFileReport",
    "AudioPipelineSummary",
    "AudioProcessingError",
    "FFmpegNotFoundError",
    "UnsupportedAudioFormatError",
    "AudioMetadataError",
    "AudioQualityAnalysisError",
    "AudioPreprocessingError",
    "verify_ffmpeg_installation",
    "get_ffmpeg_path",
    "get_ffprobe_path",
]
