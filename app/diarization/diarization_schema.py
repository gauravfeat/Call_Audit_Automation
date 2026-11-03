"""
Pydantic Schemas for Speaker Diarization Output.

Defines structured models for diarized transcripts including:
- Per-segment speaker identification with timestamps.
- Recording-level diarization summary and metadata.
- Speaker statistics (talk-time, segment counts).

Speaker IDs are generic (SPEAKER_00, SPEAKER_01) - business role mapping
(RM vs Customer) is intentionally deferred to a later stage.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class DiarizedSegment(BaseModel):
    """
    A single diarized transcript segment with speaker attribution.
    """

    segment_id: int = Field(description="Sequential zero-indexed segment identifier")
    speaker_id: str = Field(description="Speaker label (e.g. 'SPEAKER_00', 'SPEAKER_01')")
    speaker_role: str = Field(
        default="UNKNOWN",
        description="Assigned business role: 'RM', 'CUSTOMER', or 'UNKNOWN'",
    )
    start: float = Field(description="Segment start time in seconds")
    end: float = Field(description="Segment end time in seconds")
    text: str = Field(description="Transcribed text for this segment")

    @field_validator("end")
    @classmethod
    def validate_segment_timestamps(cls, v: float, info) -> float:
        """Ensure segment end timestamp is >= start timestamp."""
        start = info.data.get("start")
        if start is not None and v < start:
            raise ValueError(
                f"Segment end timestamp ({v}) cannot be before start timestamp ({start})"
            )
        return round(v, 3)

    @field_validator("start")
    @classmethod
    def round_start(cls, v: float) -> float:
        return round(v, 3)

    @property
    def duration(self) -> float:
        """Duration of this segment in seconds."""
        return round(self.end - self.start, 3)


class SpeakerStats(BaseModel):
    """
    Aggregated statistics for a single identified speaker.
    """

    speaker_id: str = Field(description="Speaker label (e.g. 'SPEAKER_00')")
    role: str = Field(default="UNKNOWN", description="Business role: UNKNOWN until mapped")
    total_segments: int = Field(description="Number of segments attributed to this speaker")
    total_words: int = Field(description="Total word count for this speaker")
    total_talk_time_seconds: float = Field(description="Total speaking duration in seconds")
    talk_time_percentage: float = Field(
        description="Percentage of total recording duration this speaker talks"
    )
    first_segment_start: float = Field(description="Start time of the speaker's first segment")
    last_segment_end: float = Field(description="End time of the speaker's last segment")


class DiarizedTranscript(BaseModel):
    """
    Complete diarized transcript for an audio recording.
    """

    recording_id: str = Field(description="Unique identifier or file stem for the audio recording")
    source_file: str = Field(description="Source audio filename (e.g. 'call_001.wav')")
    duration: float = Field(description="Total recording duration in seconds")
    num_speakers_detected: int = Field(description="Number of distinct speakers detected")
    num_speakers_requested: Optional[int] = Field(
        default=None,
        description="Number of speakers requested in API call (if specified)",
    )
    model: str = Field(description="Model identifier (e.g. 'sarvam-saaras-v4:saaras:v4')")
    mode: str = Field(default="codemix", description="Transcription mode used")
    processing_time_seconds: float = Field(description="Wall-clock processing time in seconds")
    api_cost_inr: Optional[float] = Field(default=None, description="API cost in INR")
    language: Optional[str] = Field(default=None, description="Detected language code")
    segments: List[DiarizedSegment] = Field(
        default_factory=list,
        description="Chronologically ordered diarized segments",
    )
    speaker_stats: List[SpeakerStats] = Field(
        default_factory=list,
        description="Per-speaker aggregated statistics",
    )
    full_transcript: str = Field(
        default="",
        description="Full concatenated transcript text (without speaker labels)",
    )
    provider: str = Field(default="sarvam", description="STT provider name")
    provider_metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Provider-specific metadata (job IDs, etc.)"
    )
    status: str = Field(default="SUCCESS", description="Diarization status: SUCCESS or FAILED")
    error_details: Optional[str] = Field(default=None, description="Error message if failed")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of creation",
    )

    @property
    def speaker_ids(self) -> List[str]:
        """List of unique speaker IDs in order of first appearance."""
        seen = []
        for seg in self.segments:
            if seg.speaker_id not in seen:
                seen.append(seg.speaker_id)
        return seen

    @property
    def total_words(self) -> int:
        """Total word count across all segments."""
        return sum(len(seg.text.split()) for seg in self.segments if seg.text.strip())

    @property
    def labeled_transcript(self) -> str:
        """Full transcript with speaker labels, e.g. '[SPEAKER_00]: Hello ...'"""
        lines = []
        for seg in self.segments:
            lines.append(f"[{seg.speaker_id}]: {seg.text}")
        return "\n".join(lines)

    @property
    def role_labeled_transcript(self) -> str:
        """Full transcript with speaker labels and roles, e.g. '[SPEAKER_00 (RM)]: Hello ...'"""
        lines = []
        for seg in self.segments:
            lines.append(f"[{seg.speaker_id} ({seg.speaker_role})]: {seg.text}")
        return "\n".join(lines)

    def save_json(self, output_path: Path) -> Path:
        """Save the diarized transcript to disk as indented JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return output_path

    @classmethod
    def from_json(cls, file_path: Path) -> "DiarizedTranscript":
        """Load and validate a diarized transcript from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)
