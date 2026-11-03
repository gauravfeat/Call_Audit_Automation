"""
Whisper Small (Baseline) ASR Adapter.

Reads cached M3 transcripts from output/transcripts/ or re-runs the
existing Faster-Whisper Small model if cached results are missing.
Normalizes output into the common ASRResult schema.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from app.benchmark.base import BaseASRAdapter
from app.benchmark.schema import ASRResult, BenchmarkSegment
from app.core.config import get_settings

logger = logging.getLogger("call_audit.benchmark.whisper_small_adapter")


class WhisperSmallAdapter(BaseASRAdapter):
    """
    Baseline Faster-Whisper Small adapter.
    Reads existing M3 transcript JSON files and normalizes to ASRResult.
    """

    def __init__(self):
        super().__init__(model_name="whisper-small", model_version="small")

    def calculate_cost(self, duration_seconds: float) -> Optional[float]:
        """Local model — no API cost."""
        return None

    def transcribe(
        self,
        audio_path: Path,
        recording_id: str,
        input_audio_type: str = "processed_wav_16khz",
        **kwargs,
    ) -> ASRResult:
        """
        Load cached M3 transcript or run fresh transcription if absent.
        """
        settings = get_settings()
        audio_path = Path(audio_path).resolve()

        # Look for cached transcript in output/transcripts/
        cached_json = settings.transcripts_dir / f"{recording_id}.json"

        if cached_json.exists():
            logger.info("Loading cached baseline transcript: %s", cached_json.name)
            return self._load_cached_transcript(cached_json, recording_id, input_audio_type)

        # No cached transcript — run fresh Small inference
        logger.info("No cached transcript found for %s. Running fresh Small inference.", recording_id)
        return self._run_fresh_transcription(audio_path, recording_id, input_audio_type)

    def _load_cached_transcript(
        self,
        json_path: Path,
        recording_id: str,
        input_audio_type: str,
    ) -> ASRResult:
        """Parse cached M3 JSON transcript into ASRResult."""
        warnings: List[str] = []
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract fields from the M3 Transcript schema
            segments_data = data.get("segments", [])
            segments = []
            full_text_parts = []

            for seg in segments_data:
                text = seg.get("text", "").strip()
                if not text:
                    continue
                full_text_parts.append(text)
                segments.append(BenchmarkSegment(
                    segment_id=seg.get("segment_id", len(segments)),
                    start=seg.get("start"),
                    end=seg.get("end"),
                    text=text,
                ))

            return ASRResult(
                recording_id=recording_id,
                source_file=data.get("source_file", json_path.stem),
                input_audio_type=input_audio_type,
                model_name=self.model_name,
                model_version=data.get("model", "small"),
                transcription_mode="default",
                detected_language=data.get("language"),
                language_probability=data.get("language_probability"),
                transcript=" ".join(full_text_parts),
                segments=segments,
                processing_time_seconds=data.get("processing_time_seconds", 0.0),
                api_cost_inr=None,
                status="SUCCESS",
                warnings=warnings,
                provider_metadata={
                    "source": "cached_m3_transcript",
                    "cached_file": str(json_path),
                    "audio_duration": data.get("duration"),
                },
            )

        except Exception as exc:
            logger.error("Failed to load cached transcript %s: %s", json_path, exc)
            return ASRResult(
                recording_id=recording_id,
                source_file=json_path.stem,
                input_audio_type=input_audio_type,
                model_name=self.model_name,
                model_version="small",
                transcription_mode="default",
                transcript="",
                processing_time_seconds=0.0,
                status="FAILED",
                error_details=f"Failed to parse cached transcript: {exc}",
            )

    def _run_fresh_transcription(
        self,
        audio_path: Path,
        recording_id: str,
        input_audio_type: str,
    ) -> ASRResult:
        """
        Run fresh transcription using the existing AudioTranscriber (Whisper Small).
        Does NOT save to output/transcripts/ to avoid overwriting M3 data.
        """
        start_time = time.perf_counter()
        try:
            from app.transcription.transcriber import AudioTranscriber
            transcriber = AudioTranscriber()
            transcript = transcriber.transcribe_file(audio_path, recording_id=recording_id)

            elapsed = round(time.perf_counter() - start_time, 2)

            segments = []
            full_text_parts = []
            for seg in transcript.segments:
                text = seg.text.strip()
                if not text:
                    continue
                full_text_parts.append(text)
                segments.append(BenchmarkSegment(
                    segment_id=seg.segment_id,
                    start=seg.start,
                    end=seg.end,
                    text=text,
                ))

            return ASRResult(
                recording_id=recording_id,
                source_file=audio_path.name,
                input_audio_type=input_audio_type,
                model_name=self.model_name,
                model_version=transcript.model,
                transcription_mode="default",
                detected_language=transcript.language,
                language_probability=transcript.language_probability,
                transcript=" ".join(full_text_parts),
                segments=segments,
                processing_time_seconds=elapsed,
                api_cost_inr=None,
                status="SUCCESS",
                warnings=[],
                provider_metadata={
                    "source": "fresh_inference",
                    "audio_duration": transcript.duration,
                },
            )

        except Exception as exc:
            elapsed = round(time.perf_counter() - start_time, 2)
            return ASRResult(
                recording_id=recording_id,
                source_file=audio_path.name,
                input_audio_type=input_audio_type,
                model_name=self.model_name,
                model_version="small",
                transcription_mode="default",
                transcript="",
                processing_time_seconds=elapsed,
                status="FAILED",
                error_details=str(exc),
            )
