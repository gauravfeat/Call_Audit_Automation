"""
Sarvam Saaras v4 Speaker Diarization Adapter.

Uses the Sarvam Batch API with with_diarization=True to perform
speaker diarization on audio recordings.

Diarization is only available via the Batch API (not sync REST or WebSocket).
The API returns diarized_transcript.entries with speaker IDs and timestamps.
"""

import json
import logging
import os
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.diarization.diarization_schema import (
    DiarizedSegment,
    DiarizedTranscript,
    SpeakerStats,
)

logger = logging.getLogger("call_audit.diarization.sarvam_diarizer")

# Sarvam pricing: approx INR 0.15 per minute of audio
SARVAM_COST_PER_MINUTE_INR = 0.15

# Batch API polling / timeout defaults
BATCH_POLL_INTERVAL_SECONDS = 5
BATCH_TIMEOUT_SECONDS = 600


def _get_audio_duration(audio_path: Path) -> float:
    """
    Get audio duration in seconds from WAV header.
    Falls back to file-size heuristic for non-WAV files.
    """
    if audio_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(audio_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return round(frames / rate, 2)
        except Exception:
            pass

    # Fallback: conservative estimate
    size_bytes = audio_path.stat().st_size
    return max(round(size_bytes / 32000.0, 2), 0.0)


def _load_keyterms(keyterms_path: Optional[Path] = None) -> List[str]:
    """
    Load financial domain keyterms from JSON configuration.
    """
    if keyterms_path is None:
        from app.core.config import get_settings
        keyterms_path = get_settings().financial_keyterms_path

    if not keyterms_path.exists():
        logger.warning("Financial keyterms config not found at %s", keyterms_path)
        return []

    try:
        with open(keyterms_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        terms = data.get("keyterms", [])
        logger.debug("Loaded %d financial keyterms for diarization", len(terms))
        return terms
    except Exception as exc:
        logger.warning("Failed to load keyterms from %s: %s", keyterms_path, exc)
        return []


def _normalize_speaker_id(raw_speaker_id: str) -> str:
    """
    Normalize Sarvam speaker IDs (e.g. '0', '1') to standard format
    (SPEAKER_00, SPEAKER_01).
    """
    raw = str(raw_speaker_id).strip()

    # Already in SPEAKER_XX format
    if raw.upper().startswith("SPEAKER_"):
        return raw.upper()

    # Numeric ID -> SPEAKER_XX
    try:
        num = int(raw)
        return f"SPEAKER_{num:02d}"
    except ValueError:
        return f"SPEAKER_{raw.upper()}"


class SarvamDiarizer:
    """
    Speaker Diarization adapter using Sarvam Saaras v4 Batch API.

    This adapter always uses the Batch API since diarization is only
    supported in batch mode by Sarvam.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "saaras:v4",
        mode: str = "codemix",
        num_speakers: int = 2,
        keyterms_path: Optional[Path] = None,
    ):
        self._api_key = api_key or os.environ.get("SARVAM_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Sarvam API key is required. Set SARVAM_API_KEY environment "
                "variable or pass api_key parameter."
            )
        self._model = model
        self._mode = mode
        self._num_speakers = num_speakers
        self._keyterms = _load_keyterms(keyterms_path)
        self._client = None

    @property
    def client(self):
        """Lazy-initialize the Sarvam AI client."""
        if self._client is None:
            try:
                import sarvamai
                self._client = sarvamai.SarvamAI(
                    api_subscription_key=self._api_key
                )
                logger.info("Sarvam AI client initialized for diarization.")
            except ImportError:
                raise ImportError(
                    "sarvamai package is not installed. Run: pip install sarvamai"
                )
        return self._client

    def calculate_cost(self, duration_seconds: float) -> float:
        """Estimate API cost in INR based on audio duration."""
        minutes = duration_seconds / 60.0
        return round(minutes * SARVAM_COST_PER_MINUTE_INR, 4)

    def diarize(
        self,
        audio_path: Path,
        recording_id: str,
        num_speakers: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> DiarizedTranscript:
        """
        Run speaker diarization on an audio file using the Sarvam Batch API.

        Args:
            audio_path: Path to the processed WAV audio file.
            recording_id: Unique identifier for the recording.
            num_speakers: Override for expected speaker count (default: self._num_speakers).
            mode: Override transcription mode (default: self._mode).

        Returns:
            DiarizedTranscript with speaker-attributed segments.
        """
        audio_path = Path(audio_path).resolve()
        effective_mode = mode or self._mode
        effective_num_speakers = num_speakers or self._num_speakers

        if not audio_path.is_file():
            return DiarizedTranscript(
                recording_id=recording_id,
                source_file=audio_path.name,
                duration=0.0,
                num_speakers_detected=0,
                num_speakers_requested=effective_num_speakers,
                model=f"sarvam-saaras-v4:{self._model}",
                mode=effective_mode,
                processing_time_seconds=0.0,
                status="FAILED",
                error_details=f"Audio file not found: {audio_path}",
            )

        duration = _get_audio_duration(audio_path)
        start_time = time.perf_counter()
        warnings: List[str] = []
        metadata: Dict[str, Any] = {
            "api_method": "batch",
            "keyterms_count": len(self._keyterms),
            "num_speakers_requested": effective_num_speakers,
        }

        try:
            return self._diarize_batch(
                audio_path=audio_path,
                recording_id=recording_id,
                mode=effective_mode,
                num_speakers=effective_num_speakers,
                duration=duration,
                start_time=start_time,
                warnings=warnings,
                metadata=metadata,
            )
        except Exception as exc:
            elapsed = round(time.perf_counter() - start_time, 2)
            err_msg = f"Diarization failed for '{recording_id}': {exc}"
            logger.error(err_msg)
            return DiarizedTranscript(
                recording_id=recording_id,
                source_file=audio_path.name,
                duration=duration,
                num_speakers_detected=0,
                num_speakers_requested=effective_num_speakers,
                model=f"sarvam-saaras-v4:{self._model}",
                mode=effective_mode,
                processing_time_seconds=elapsed,
                api_cost_inr=self.calculate_cost(duration),
                status="FAILED",
                error_details=err_msg,
                warnings=warnings,
                provider_metadata=metadata,
            )

    def _diarize_batch(
        self,
        audio_path: Path,
        recording_id: str,
        mode: str,
        num_speakers: int,
        duration: float,
        start_time: float,
        warnings: List[str],
        metadata: Dict[str, Any],
    ) -> DiarizedTranscript:
        """
        Execute diarization via the Sarvam Batch STT API with with_diarization=True.
        """
        logger.info(
            "Creating Sarvam batch diarization job: recording=%s, mode=%s, "
            "num_speakers=%d, duration=%.1fs",
            recording_id, mode, num_speakers, duration,
        )

        # 1. Create batch job with diarization enabled
        job = self.client.speech_to_text_job.create_job(
            model=self._model,
            mode=mode,
            language_code="unknown",
            with_timestamps=True,
            with_diarization=True,
            num_speakers=num_speakers,
            keyterms=self._keyterms if self._keyterms else None,
        )
        job_id = job.job_id
        metadata["job_id"] = job_id
        logger.info("Batch diarization job created: %s", job_id)

        # 2. Upload audio file
        logger.info("Uploading audio file for diarization job %s...", job_id)
        upload_ok = job.upload_files(file_paths=[str(audio_path)])
        if not upload_ok:
            raise RuntimeError(
                f"Failed to upload audio file to Sarvam diarization job {job_id}"
            )
        logger.info("File uploaded successfully for diarization job %s.", job_id)

        # 3. Start job
        logger.info("Starting diarization job %s...", job_id)
        job.start()
        logger.info("Diarization job %s started. Polling for completion...", job_id)

        # 4. Wait for completion
        status_response = job.wait_until_complete(
            poll_interval=BATCH_POLL_INTERVAL_SECONDS,
            timeout=BATCH_TIMEOUT_SECONDS,
        )
        metadata["final_status"] = str(getattr(status_response, "status", "unknown"))

        # 5. Check results
        file_results = job.get_file_results()
        successful = file_results.get("successful", [])
        failed = file_results.get("failed", [])

        if failed:
            err_msgs = [f.get("error_message", "Unknown error") for f in failed]
            warnings.append(f"Batch job reported failures: {err_msgs}")

        if not successful:
            elapsed = round(time.perf_counter() - start_time, 2)
            return DiarizedTranscript(
                recording_id=recording_id,
                source_file=audio_path.name,
                duration=duration,
                num_speakers_detected=0,
                num_speakers_requested=num_speakers,
                model=f"sarvam-saaras-v4:{self._model}",
                mode=mode,
                processing_time_seconds=elapsed,
                api_cost_inr=self.calculate_cost(duration),
                status="FAILED",
                error_details="No successful files in batch diarization results.",
                warnings=warnings,
                provider_metadata=metadata,
            )

        # 6. Download outputs
        from app.core.config import get_settings
        settings = get_settings()
        download_dir = (
            settings.diarized_transcripts_dir
            / "raw_outputs"
            / recording_id
        )
        download_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Downloading diarization outputs for job %s to %s...",
            job_id, download_dir,
        )
        job.download_outputs(output_dir=str(download_dir))

        # 7. Parse diarized output
        segments, full_text, detected_lang = self._parse_diarized_output(
            download_dir, audio_path.name
        )

        elapsed = round(time.perf_counter() - start_time, 2)

        # 8. Compute speaker statistics
        unique_speakers = list(dict.fromkeys(seg.speaker_id for seg in segments))
        speaker_stats = self._compute_speaker_stats(segments, duration)

        return DiarizedTranscript(
            recording_id=recording_id,
            source_file=audio_path.name,
            duration=duration,
            num_speakers_detected=len(unique_speakers),
            num_speakers_requested=num_speakers,
            model=f"sarvam-saaras-v4:{self._model}",
            mode=mode,
            processing_time_seconds=elapsed,
            api_cost_inr=self.calculate_cost(duration),
            language=detected_lang,
            segments=segments,
            speaker_stats=speaker_stats,
            full_transcript=full_text,
            provider="sarvam",
            provider_metadata=metadata,
            status="SUCCESS",
            warnings=warnings,
        )

    def _parse_diarized_output(
        self,
        download_dir: Path,
        source_filename: str,
    ) -> Tuple[List[DiarizedSegment], str, Optional[str]]:
        """
        Parse downloaded Sarvam batch diarization output JSON files.

        Expected structure:
        {
            "transcript": "...",
            "diarized_transcript": {
                "entries": [
                    {
                        "transcript": "segment text",
                        "start_time_seconds": 0.01,
                        "end_time_seconds": 2.5,
                        "speaker_id": "0"
                    },
                    ...
                ]
            },
            "language_code": "hi-IN"
        }

        Returns: (segments, full_transcript_text, detected_language)
        """
        segments: List[DiarizedSegment] = []
        transcript_parts: List[str] = []
        detected_lang: Optional[str] = None

        # Locate output JSON files
        target_json = download_dir / f"{source_filename}.json"
        if target_json.exists():
            json_files = [target_json]
        else:
            stem_match = list(
                download_dir.glob(f"*{Path(source_filename).stem}*.json")
            )
            if stem_match:
                json_files = stem_match
            else:
                json_files = sorted(download_dir.glob("*.json"))

        if not json_files:
            logger.warning(
                "No JSON output files found in diarization download dir: %s",
                download_dir,
            )
            return ([], "", None)

        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Handle both single-object and list formats
                items = data if isinstance(data, list) else [data]
                for item in items:
                    # Capture language code
                    if detected_lang is None:
                        detected_lang = item.get("language_code")

                    # Capture fallback transcript
                    fallback_text = item.get("transcript", "")
                    if fallback_text:
                        transcript_parts.append(fallback_text)

                    # Parse diarized_transcript.entries
                    diarized = item.get("diarized_transcript")
                    if diarized is None:
                        # Try attribute access for SDK response objects
                        diarized = getattr(item, "diarized_transcript", None)

                    if diarized is not None:
                        entries = None
                        if isinstance(diarized, dict):
                            entries = diarized.get("entries", [])
                        else:
                            entries = getattr(diarized, "entries", [])

                        if entries:
                            for entry in entries:
                                if isinstance(entry, dict):
                                    text = entry.get("transcript", "")
                                    start = entry.get("start_time_seconds", 0.0)
                                    end = entry.get("end_time_seconds", 0.0)
                                    speaker = entry.get("speaker_id", "0")
                                else:
                                    text = getattr(entry, "transcript", "")
                                    start = getattr(entry, "start_time_seconds", 0.0)
                                    end = getattr(entry, "end_time_seconds", 0.0)
                                    speaker = getattr(entry, "speaker_id", "0")

                                if not text or not str(text).strip():
                                    continue

                                segments.append(
                                    DiarizedSegment(
                                        segment_id=len(segments),
                                        speaker_id=_normalize_speaker_id(speaker),
                                        start=round(float(start), 3),
                                        end=round(float(end), 3),
                                        text=str(text).strip(),
                                    )
                                )

            except Exception as exc:
                logger.warning(
                    "Failed to parse diarization output file %s: %s", jf, exc
                )

        # Build full transcript text from diarized segments if available,
        # otherwise fall back to the plain transcript
        if segments:
            full_text = " ".join(seg.text for seg in segments)
        else:
            full_text = " ".join(transcript_parts).strip()

        return (segments, full_text, detected_lang)

    def _compute_speaker_stats(
        self,
        segments: List[DiarizedSegment],
        total_duration: float,
    ) -> List[SpeakerStats]:
        """
        Compute per-speaker aggregated statistics from diarized segments.
        """
        if not segments:
            return []

        # Group segments by speaker
        speaker_data: Dict[str, Dict[str, Any]] = {}
        for seg in segments:
            sid = seg.speaker_id
            if sid not in speaker_data:
                speaker_data[sid] = {
                    "segments": 0,
                    "words": 0,
                    "talk_time": 0.0,
                    "first_start": seg.start,
                    "last_end": seg.end,
                }
            sd = speaker_data[sid]
            sd["segments"] += 1
            sd["words"] += len(seg.text.split())
            sd["talk_time"] += seg.duration
            sd["first_start"] = min(sd["first_start"], seg.start)
            sd["last_end"] = max(sd["last_end"], seg.end)

        stats: List[SpeakerStats] = []
        for sid, sd in speaker_data.items():
            talk_pct = (
                round((sd["talk_time"] / total_duration) * 100, 2)
                if total_duration > 0
                else 0.0
            )
            stats.append(
                SpeakerStats(
                    speaker_id=sid,
                    role="UNKNOWN",
                    total_segments=sd["segments"],
                    total_words=sd["words"],
                    total_talk_time_seconds=round(sd["talk_time"], 3),
                    talk_time_percentage=talk_pct,
                    first_segment_start=round(sd["first_start"], 3),
                    last_segment_end=round(sd["last_end"], 3),
                )
            )

        # Sort by first appearance
        stats.sort(key=lambda s: s.first_segment_start)
        return stats
