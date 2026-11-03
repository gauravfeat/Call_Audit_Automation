"""
Pydantic Schemas for Structured Speech-to-Text Transcripts.

Provides schema validation for segments, word-level timestamps,
and complete audio call transcripts.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class WordTimestamp(BaseModel):
    """
    Word-level timing and model confidence.
    """

    word: str = Field(description="Recognized word token")
    start: float = Field(description="Start time of the word in seconds")
    end: float = Field(description="End time of the word in seconds")
    probability: Optional[float] = Field(
        default=None,
        description="Decoder confidence probability for this token (0.0 to 1.0)",
    )

    @field_validator("end")
    @classmethod
    def validate_word_timestamps(cls, v: float, info) -> float:
        """Ensure word end timestamp is greater than or equal to start timestamp."""
        start = info.data.get("start")
        if start is not None and v < start:
            raise ValueError(f"Word end timestamp ({v}) cannot be before start timestamp ({start})")
        return round(v, 3)

    @field_validator("start")
    @classmethod
    def round_start(cls, v: float) -> float:
        return round(v, 3)


class TranscriptSegment(BaseModel):
    """
    Sequential transcription segment corresponding to a speech utterance.
    """

    segment_id: int = Field(description="Sequential zero-indexed identifier for the segment")
    start: float = Field(default=0.0, description="Start time of segment in seconds")
    end: float = Field(default=0.0, description="End time of segment in seconds")
    text: str = Field(description="Transcribed text content for this segment")
    words: Optional[List[WordTimestamp]] = Field(
        default=None,
        description="Optional word-level timestamps if enabled during decoding",
    )
    speaker: Optional[str] = Field(
        default=None,
        description="Speaker label for future M4 diarization integration (e.g. 'RM', 'Customer')",
    )

    @field_validator("end")
    @classmethod
    def validate_segment_timestamps(cls, v: float, info) -> float:
        """Ensure segment end timestamp is greater than or equal to start timestamp."""
        start = info.data.get("start")
        if start is not None and v < start:
            raise ValueError(f"Segment end timestamp ({v}) cannot be before start timestamp ({start})")
        return round(v, 3)

    @field_validator("start")
    @classmethod
    def round_start(cls, v: float) -> float:
        return round(v, 3)


class Transcript(BaseModel):
    """
    Complete structured transcript for an audio recording.
    Supports both Sarvam Saaras v4 (primary) and Faster-Whisper engines.
    """

    recording_id: str = Field(description="Unique identifier or file stem for the audio recording")
    source_file: str = Field(description="Standardized source audio filename (e.g. call_001.wav)")
    language: str = Field(description="Detected or configured BCP-47 / ISO-639-1 language code (e.g. 'hi', 'en', 'hi-IN')")
    language_probability: float = Field(default=1.0, description="Model confidence probability for detected language (0.0 to 1.0)")
    duration: float = Field(description="Total recording duration in seconds")
    model: str = Field(description="Model identifier used (e.g., 'sarvam-saaras-v4:saaras:v4' or 'small')")
    processing_time_seconds: float = Field(description="Wall-clock inference time in seconds")
    segments: List[TranscriptSegment] = Field(
        default_factory=list,
        description="Chronologically ordered transcript segments",
    )
    provider: str = Field(default="sarvam", description="STT Provider name (e.g. 'sarvam', 'faster-whisper')")
    transcription_mode: Optional[str] = Field(default="codemix", description="Transcription mode (e.g. 'codemix', 'verbatim')")
    api_cost_inr: Optional[float] = Field(default=None, description="API cost in INR if cloud provider")
    provider_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Provider-specific execution metadata")

    @field_validator("language_probability")
    @classmethod
    def validate_probability(cls, v: float) -> float:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"Language probability must be between 0.0 and 1.0, got {v}")
        return round(v, 4) if v is not None else 1.0

    @property
    def full_text(self) -> str:
        """Full concatenated text of all segments."""
        return " ".join(seg.text.strip() for seg in self.segments if seg.text.strip())

    @property
    def total_words(self) -> int:
        """Total count of words in the transcript."""
        return sum(len(seg.text.split()) for seg in self.segments)

    def save_json(self, output_path: Path) -> Path:
        """
        Save the transcript to disk formatted as standard indented JSON.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return output_path

    @classmethod
    def from_json(cls, file_path: Path) -> "Transcript":
        """
        Load and validate a transcript from a JSON file.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    @classmethod
    def from_asr_result(cls, asr_result: Any, duration: Optional[float] = None) -> "Transcript":
        """
        Convert an ASRResult (from app.benchmark) into a canonical Transcript model.
        """
        segments: List[TranscriptSegment] = []
        for s in getattr(asr_result, "segments", []):
            start = getattr(s, "start", 0.0)
            end = getattr(s, "end", 0.0)
            text = getattr(s, "text", "")
            speaker = getattr(s, "speaker", None)
            segments.append(
                TranscriptSegment(
                    segment_id=getattr(s, "segment_id", len(segments)),
                    start=round(float(start), 3) if start is not None else 0.0,
                    end=round(float(end), 3) if end is not None else (round(float(start), 3) if start is not None else 0.0),
                    text=str(text).strip(),
                    words=None,
                    speaker=speaker,
                )
            )

        calc_duration = duration
        if calc_duration is None:
            if segments and segments[-1].end > 0:
                calc_duration = segments[-1].end
            else:
                calc_duration = 0.0

        model_name = getattr(asr_result, "model_name", "sarvam-saaras-v4")
        model_version = getattr(asr_result, "model_version", "saaras:v4")
        mode = getattr(asr_result, "transcription_mode", "codemix")
        lang = getattr(asr_result, "detected_language", None) or "hi-IN"
        lang_prob = getattr(asr_result, "language_probability", None)

        provider = "sarvam" if "sarvam" in model_name.lower() else "faster-whisper"

        return cls(
            recording_id=getattr(asr_result, "recording_id", "unknown"),
            source_file=getattr(asr_result, "source_file", "unknown.wav"),
            language=lang,
            language_probability=float(lang_prob) if lang_prob is not None else 1.0,
            duration=round(float(calc_duration), 2),
            model=f"{model_name}:{model_version}",
            processing_time_seconds=round(float(getattr(asr_result, "processing_time_seconds", 0.0)), 2),
            segments=segments,
            provider=provider,
            transcription_mode=mode,
            api_cost_inr=getattr(asr_result, "api_cost_inr", None),
            provider_metadata=getattr(asr_result, "provider_metadata", None),
        )
