"""
Transcription Package for Call Audit Automation.

Provides Faster-Whisper integration, model management, structured transcript
schemas, and batch transcription workflows.
"""

from app.transcription.model import ModelLoadError, WhisperModelManager, get_whisper_model
from app.transcription.transcriber import AudioTranscriber, TranscriptionError
from app.transcription.transcript_schema import Transcript, TranscriptSegment, WordTimestamp

__all__ = [
    "WordTimestamp",
    "TranscriptSegment",
    "Transcript",
    "WhisperModelManager",
    "get_whisper_model",
    "ModelLoadError",
    "AudioTranscriber",
    "TranscriptionError",
]
