"""
Sarvam Saaras v4 ASR Adapter.

Integrates Sarvam AI's speech-to-text API using the official sarvamai SDK.
Supports both synchronous (short audio) and batch (long audio) modes.
Handles codemix and verbatim transcription modes with financial keyterm priming.
"""

import json
import logging
import os
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.benchmark.base import BaseASRAdapter
from app.benchmark.schema import ASRResult, BenchmarkSegment

logger = logging.getLogger("call_audit.benchmark.sarvam_adapter")

# Sarvam pricing: approx ₹0.15 per minute of audio
SARVAM_COST_PER_MINUTE_INR = 0.15

# Maximum audio duration (seconds) for synchronous transcription
# Recordings longer than this threshold use the Batch API
SYNC_DURATION_LIMIT_SECONDS = 30.0

# Batch API polling / timeout defaults
BATCH_POLL_INTERVAL_SECONDS = 5
BATCH_TIMEOUT_SECONDS = 600


def _get_audio_duration(audio_path: Path) -> float:
    """
    Get audio duration in seconds.
    Supports WAV natively; for other formats, falls back to a conservative estimate.
    """
    if audio_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(audio_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return frames / rate
        except Exception:
            pass

    # Fallback: use file size heuristic for MP3 (~1 byte/ms at 8kbps mono)
    # Conservative estimate to prefer Batch API over sync for long files
    size_bytes = audio_path.stat().st_size
    # Approximate: MP3 at 32kbps → 4000 bytes/sec
    return max(size_bytes / 4000.0, 31.0)


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
        logger.debug("Loaded %d financial keyterms from %s", len(terms), keyterms_path)
        return terms
    except Exception as exc:
        logger.warning("Failed to load keyterms from %s: %s", keyterms_path, exc)
        return []


def _detect_audio_codec(audio_path: Path) -> Optional[str]:
    """
    Map file extension to Sarvam's expected input_audio_codec value.
    """
    ext = audio_path.suffix.lower()
    mapping = {
        ".wav": "wav",
        ".mp3": "mp3",
        ".aac": "aac",
        ".flac": "flac",
        ".ogg": "ogg",
        ".m4a": "x-m4a",
        ".mp4": "mp4",
        ".webm": "webm",
    }
    return mapping.get(ext)


class SarvamAdapter(BaseASRAdapter):
    """
    Sarvam Saaras v4 Speech-to-Text adapter.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "saaras:v4",
        mode: str = "codemix",
        keyterms_path: Optional[Path] = None,
    ):
        super().__init__(model_name="sarvam-saaras-v4", model_version=model)
        self._api_key = api_key or os.environ.get("SARVAM_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Sarvam API key is required. Set SARVAM_API_KEY environment variable or pass api_key."
            )
        self._model = model
        self._mode = mode
        self._keyterms = _load_keyterms(keyterms_path)
        self._client = None

    @property
    def client(self):
        """Lazy-initialize the Sarvam client."""
        if self._client is None:
            try:
                import sarvamai
                self._client = sarvamai.SarvamAI(api_subscription_key=self._api_key)
                logger.info("Sarvam AI client initialized successfully.")
            except ImportError:
                raise ImportError("sarvamai package is not installed. Run: pip install sarvamai")
        return self._client

    def calculate_cost(self, duration_seconds: float) -> Optional[float]:
        """Estimate API cost in INR based on audio duration."""
        minutes = duration_seconds / 60.0
        return round(minutes * SARVAM_COST_PER_MINUTE_INR, 4)

    def transcribe(
        self,
        audio_path: Path,
        recording_id: str,
        input_audio_type: str = "processed_wav_16khz",
        **kwargs,
    ) -> ASRResult:
        """
        Transcribe audio using Sarvam Saaras v4 API.
        Automatically selects sync or batch mode based on audio duration.
        """
        mode = kwargs.get("mode", self._mode)
        audio_path = Path(audio_path).resolve()

        if not audio_path.is_file():
            return ASRResult(
                recording_id=recording_id,
                source_file=audio_path.name,
                input_audio_type=input_audio_type,
                model_name=self.model_name,
                model_version=self.model_version,
                transcription_mode=mode,
                transcript="",
                processing_time_seconds=0.0,
                status="FAILED",
                error_details=f"Audio file not found: {audio_path}",
            )

        duration = _get_audio_duration(audio_path)
        logger.info(
            "Sarvam transcription: recording=%s, mode=%s, duration=%.1fs, method=%s",
            recording_id, mode, duration,
            "batch" if duration > SYNC_DURATION_LIMIT_SECONDS else "sync",
        )

        try:
            if duration <= SYNC_DURATION_LIMIT_SECONDS:
                return self._transcribe_sync(audio_path, recording_id, input_audio_type, mode)
            else:
                return self._transcribe_batch(audio_path, recording_id, input_audio_type, mode, duration)
        except Exception as exc:
            logger.error("Sarvam transcription failed for %s: %s", recording_id, exc)
            return ASRResult(
                recording_id=recording_id,
                source_file=audio_path.name,
                input_audio_type=input_audio_type,
                model_name=self.model_name,
                model_version=self.model_version,
                transcription_mode=mode,
                transcript="",
                processing_time_seconds=0.0,
                status="FAILED",
                error_details=str(exc),
            )

    def _transcribe_sync(
        self,
        audio_path: Path,
        recording_id: str,
        input_audio_type: str,
        mode: str,
    ) -> ASRResult:
        """
        Transcribe short audio (≤30s) using the synchronous REST endpoint.
        """
        start_time = time.perf_counter()
        warnings: List[str] = []
        codec = _detect_audio_codec(audio_path)

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        response = self.client.speech_to_text.transcribe(
            file=audio_bytes,
            model=self._model,
            mode=mode,
            language_code="unknown",
            with_timestamps=True,
            input_audio_codec=codec,
            keyterms=self._keyterms if self._keyterms else None,
        )

        elapsed = round(time.perf_counter() - start_time, 2)
        transcript = getattr(response, "transcript", "") or ""
        request_id = getattr(response, "request_id", None)
        lang_code = getattr(response, "language_code", None)
        lang_prob = getattr(response, "language_probability", None)
        timestamps = getattr(response, "timestamps", None)

        segments = self._parse_timestamps_to_segments(timestamps)

        # Estimate duration from WAV for cost
        try:
            duration = _get_audio_duration(audio_path)
        except Exception:
            duration = 0.0

        return ASRResult(
            recording_id=recording_id,
            source_file=audio_path.name,
            input_audio_type=input_audio_type,
            model_name=self.model_name,
            model_version=self.model_version,
            transcription_mode=mode,
            detected_language=lang_code,
            language_probability=float(lang_prob) if lang_prob is not None else None,
            transcript=transcript,
            segments=segments,
            processing_time_seconds=elapsed,
            api_cost_inr=self.calculate_cost(duration),
            status="SUCCESS",
            warnings=warnings,
            provider_metadata={
                "request_id": request_id,
                "api_method": "sync",
                "keyterms_count": len(self._keyterms),
            },
        )

    def _transcribe_batch(
        self,
        audio_path: Path,
        recording_id: str,
        input_audio_type: str,
        mode: str,
        duration: float,
    ) -> ASRResult:
        """
        Transcribe long audio (>30s) using the Sarvam Batch STT API.
        """
        start_time = time.perf_counter()
        warnings: List[str] = []
        metadata: Dict[str, Any] = {"api_method": "batch", "keyterms_count": len(self._keyterms)}

        # 1. Create batch job
        logger.info("Creating Sarvam batch STT job for %s (mode=%s)...", recording_id, mode)
        job = self.client.speech_to_text_job.create_job(
            model=self._model,
            mode=mode,
            language_code="unknown",
            with_timestamps=True,
            keyterms=self._keyterms if self._keyterms else None,
        )
        job_id = job.job_id
        metadata["job_id"] = job_id
        logger.info("Batch job created: %s", job_id)

        # 2. Upload file
        logger.info("Uploading audio file for job %s...", job_id)
        upload_ok = job.upload_files(file_paths=[str(audio_path)])
        if not upload_ok:
            raise RuntimeError(f"Failed to upload audio file to Sarvam batch job {job_id}")
        logger.info("File uploaded successfully for job %s.", job_id)

        # 3. Start job
        logger.info("Starting batch job %s...", job_id)
        job.start()
        logger.info("Batch job %s started. Polling for completion...", job_id)

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
            return ASRResult(
                recording_id=recording_id,
                source_file=audio_path.name,
                input_audio_type=input_audio_type,
                model_name=self.model_name,
                model_version=self.model_version,
                transcription_mode=mode,
                transcript="",
                processing_time_seconds=elapsed,
                api_cost_inr=self.calculate_cost(duration),
                status="FAILED",
                warnings=warnings,
                error_details="No successful files in batch results.",
                provider_metadata=metadata,
            )

        # 6. Download outputs
        from app.core.config import get_settings
        settings = get_settings()
        download_dir = settings.stt_benchmark_dir / f"sarvam_v4_{mode}" / "raw_outputs" / recording_id
        download_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading batch outputs for job %s to %s...", job_id, download_dir)
        job.download_outputs(output_dir=str(download_dir))

        # 7. Parse downloaded transcript
        transcript_text, segments, detected_lang, lang_prob = self._parse_batch_output(
            download_dir, audio_path.name
        )

        elapsed = round(time.perf_counter() - start_time, 2)

        return ASRResult(
            recording_id=recording_id,
            source_file=audio_path.name,
            input_audio_type=input_audio_type,
            model_name=self.model_name,
            model_version=self.model_version,
            transcription_mode=mode,
            detected_language=detected_lang,
            language_probability=lang_prob,
            transcript=transcript_text,
            segments=segments,
            processing_time_seconds=elapsed,
            api_cost_inr=self.calculate_cost(duration),
            status="SUCCESS",
            warnings=warnings,
            provider_metadata=metadata,
        )

    def _parse_batch_output(
        self, download_dir: Path, source_filename: str
    ) -> tuple:
        """
        Parse downloaded Sarvam batch job output JSON files.

        Returns: (transcript_text, segments, detected_language, language_probability)
        """
        transcript_parts = []
        all_segments: List[BenchmarkSegment] = []
        detected_lang = None
        lang_prob = None

        # Look for target JSON output file first
        target_json = download_dir / f"{source_filename}.json"
        if target_json.exists():
            json_files = [target_json]
        else:
            stem_match = list(download_dir.glob(f"*{Path(source_filename).stem}*.json"))
            if stem_match:
                json_files = stem_match
            else:
                json_files = sorted(download_dir.glob("*.json"))

        if not json_files:
            logger.warning("No JSON output files found in %s", download_dir)
            return ("", [], None, None)

        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Handle both single-object and list-of-objects formats
                items = data if isinstance(data, list) else [data]
                for item in items:
                    text = item.get("transcript", "")
                    if text:
                        transcript_parts.append(text)

                    if detected_lang is None:
                        detected_lang = item.get("language_code")
                    if lang_prob is None:
                        lp = item.get("language_probability")
                        if lp is not None:
                            lang_prob = float(lp)

                    timestamps = item.get("timestamps")
                    if timestamps:
                        parsed_segs = self._parse_timestamps_to_segments(timestamps)
                        for seg in parsed_segs:
                            seg.segment_id = len(all_segments)
                            all_segments.append(seg)

            except Exception as exc:
                logger.warning("Failed to parse batch output file %s: %s", jf, exc)

        full_text = " ".join(transcript_parts).strip()
        return (full_text, all_segments, detected_lang, lang_prob)

    def _parse_timestamps_to_segments(self, timestamps: Any) -> List[BenchmarkSegment]:
        """
        Parse Sarvam timestamp response objects into BenchmarkSegment list.
        Supports:
          1. Sarvam TimestampsModel or dict with parallel lists:
             - words: List[str]
             - start_time_seconds: List[float]
             - end_time_seconds: List[float]
          2. List of chunk objects/dicts:
             - start / start_time_seconds
             - end / end_time_seconds
             - text / word
        """
        if not timestamps:
            return []

        segments: List[BenchmarkSegment] = []

        # Check 1: Parallel arrays (Sarvam TimestampsModel or dict)
        words = getattr(timestamps, "words", None)
        starts = getattr(timestamps, "start_time_seconds", None)
        ends = getattr(timestamps, "end_time_seconds", None)

        if words is None and isinstance(timestamps, dict):
            words = timestamps.get("words")
            starts = timestamps.get("start_time_seconds")
            ends = timestamps.get("end_time_seconds")

        if words is not None and isinstance(words, (list, tuple)):
            starts = starts if isinstance(starts, (list, tuple)) else []
            ends = ends if isinstance(ends, (list, tuple)) else []
            for i, w in enumerate(words):
                s = starts[i] if i < len(starts) else None
                e = ends[i] if i < len(ends) else None
                segments.append(
                    BenchmarkSegment(
                        segment_id=i,
                        start=round(float(s), 3) if s is not None else None,
                        end=round(float(e), 3) if e is not None else None,
                        text=str(w).strip(),
                    )
                )
            return segments

        # Check 2: List of chunk objects/dicts
        if isinstance(timestamps, (list, tuple)):
            for i, ts in enumerate(timestamps):
                if isinstance(ts, dict):
                    s = ts.get("start", ts.get("start_time", ts.get("start_time_seconds")))
                    e = ts.get("end", ts.get("end_time", ts.get("end_time_seconds")))
                    t = ts.get("text", ts.get("word", ""))
                else:
                    s = getattr(ts, "start", getattr(ts, "start_time_seconds", None))
                    e = getattr(ts, "end", getattr(ts, "end_time_seconds", None))
                    t = getattr(ts, "text", getattr(ts, "word", ""))

                segments.append(
                    BenchmarkSegment(
                        segment_id=i,
                        start=round(float(s), 3) if s is not None else None,
                        end=round(float(e), 3) if e is not None else None,
                        text=str(t).strip(),
                    )
                )

        return segments

