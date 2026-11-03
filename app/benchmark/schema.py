"""
Common ASR Benchmark Result Schemas.

Defines standardized Pydantic models for cross-model Speech-to-Text benchmarking
across local (Whisper) and cloud API (Sarvam) providers.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BenchmarkSegment(BaseModel):
    """
    Normalized audio transcription segment.
    """

    segment_id: int = Field(description="Sequential segment index")
    start: Optional[float] = Field(default=None, description="Start timestamp in seconds")
    end: Optional[float] = Field(default=None, description="End timestamp in seconds")
    text: str = Field(description="Segment transcript text")
    speaker: Optional[str] = Field(default=None, description="Speaker identifier if diarized")
    confidence: Optional[float] = Field(default=None, description="Segment decoding confidence (0.0 to 1.0)")


class ASRResult(BaseModel):
    """
    Standardized, provider-agnostic ASR evaluation result.
    """

    recording_id: str = Field(description="Audio sample identifier or stem")
    source_file: str = Field(description="Audio filename used for inference")
    input_audio_type: str = Field(description="Input audio type: 'processed_wav_16khz' or 'original_mp3_8khz'")
    model_name: str = Field(description="Unique model name (e.g. 'sarvam-saaras-v4', 'whisper-large-v3')")
    model_version: str = Field(description="Model version string or checkpoint")
    transcription_mode: str = Field(description="Transcription mode: 'default', 'codemix', 'verbatim'")
    detected_language: Optional[str] = Field(default=None, description="Detected language code (e.g. 'hi', 'en')")
    language_probability: Optional[float] = Field(default=None, description="Confidence of language detection")
    transcript: str = Field(description="Complete concatenated transcription text")
    segments: List[BenchmarkSegment] = Field(default_factory=list, description="Ordered transcription segments")
    processing_time_seconds: float = Field(description="Wall-clock inference/API processing time in seconds")
    api_cost_inr: Optional[float] = Field(default=None, description="Estimated API invocation cost in INR")
    status: str = Field(description="Outcome status: SUCCESS, WARNING, or FAILED")
    warnings: List[str] = Field(default_factory=list, description="Descriptive execution warnings")
    error_details: Optional[str] = Field(default=None, description="Error message if execution failed")
    provider_metadata: Dict[str, Any] = Field(default_factory=dict, description="Raw provider metadata / request IDs")

    @property
    def total_words(self) -> int:
        """Total number of words in the recognized transcript."""
        return len(self.transcript.split())

    def save_json(self, destination: Path) -> Path:
        """Serialize benchmark result to JSON."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return destination

    @classmethod
    def from_json(cls, file_path: Path) -> "ASRResult":
        """Load benchmark result from JSON."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)
