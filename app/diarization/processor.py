"""
Diarization Processor Module.

Orchestrates batch and single-recording speaker diarization.
Supports both Sarvam Saaras v4 Batch API and local pyannote Community-1.
Handles discovery, caching, error recovery, alignment, and summary reporting.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import Settings, get_settings
from app.diarization.diarization_schema import DiarizedTranscript
from app.diarization.sarvam_diarizer import SarvamDiarizer

logger = logging.getLogger("call_audit.diarization.processor")


class DiarizationError(Exception):
    """Raised when diarization processing fails."""
    pass


class DiarizationProcessor:
    """
    Batch and single processor for speaker diarization.
    Supports Sarvam Saaras v4 and local pyannote Community-1.

    Features:
    - Sequential processing of audio files.
    - Caching: skips recordings with existing valid diarized transcripts.
    - Per-recording error isolation (one failure doesn't block the batch).
    - Summary report generation.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        provider: str = "sarvam",
        mode: str = "codemix",
        num_speakers: Optional[int] = 2,
        diarizer: Optional[Any] = None,
    ):
        self.settings = settings or get_settings()
        self.provider = provider.lower()
        self.mode = mode
        self.num_speakers = num_speakers
        self._custom_diarizer = diarizer

        if self.provider == "sarvam":
            self.diarizer = diarizer or SarvamDiarizer(
                api_key=self.settings.SARVAM_API_KEY,
                model=self.settings.SARVAM_MODEL,
                mode=self.mode,
                num_speakers=self.num_speakers or 2,
            )
        else:
            from app.diarization.pyannote_diarizer import PyannoteDiarizer
            self.diarizer = diarizer or PyannoteDiarizer(
                settings=self.settings,
            )

    def process_recording(
        self,
        wav_path: Path,
        output_dir: Optional[Path] = None,
        reuse_existing: bool = True,
    ) -> DiarizedTranscript:
        """
        Process a single audio recording through speaker diarization.
        """
        wav_path = Path(wav_path).resolve()
        if not wav_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        rec_id = wav_path.stem
        out_dir = output_dir or (
            self.settings.onprem_pyannote_diarized_dir
            if self.provider in ("pyannote", "onprem")
            else self.settings.diarized_transcripts_dir
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        cached_path = out_dir / f"{rec_id}.json"

        # Check cached output
        if reuse_existing and cached_path.is_file():
            try:
                cached = DiarizedTranscript.from_json(cached_path)
                if (
                    cached.status == "SUCCESS"
                    and cached.segments
                    and cached.num_speakers_detected > 0
                ):
                    logger.info(
                        "Reusing cached diarized transcript for %s (%d speakers, %d segments)",
                        rec_id,
                        cached.num_speakers_detected,
                        len(cached.segments),
                    )
                    return cached
            except Exception as exc:
                logger.debug("Could not reuse cached diarization for %s: %s", rec_id, exc)

        start_time = time.perf_counter()

        if self.provider in ("pyannote", "onprem"):
            # 1. Obtain STT Transcript (Whisper API or cached)
            from app.transcription.transcriber import AudioTranscriber
            from app.diarization.alignment import align_transcript_and_diarization, save_onprem_diarized_transcript

            stt_transcriber = AudioTranscriber(provider="whisper_api", settings=self.settings)
            transcript = stt_transcriber.transcribe_file(wav_path, recording_id=rec_id, reuse_existing=reuse_existing)

            # 2. Run pyannote diarization
            turns = self.diarizer.diarize_file(
                audio_path=wav_path,
                recording_id=rec_id,
                num_speakers=self.num_speakers,
            )

            proc_time = round(time.perf_counter() - start_time, 2)

            # 3. Align transcript with diarization
            diarized = align_transcript_and_diarization(
                transcript=transcript,
                speaker_turns=turns,
                recording_id=rec_id,
                processing_time_seconds=proc_time,
                num_speakers_requested=self.num_speakers,
                settings=self.settings,
            )

            # 4. Save JSON and TXT
            save_onprem_diarized_transcript(diarized, output_dir=out_dir, settings=self.settings)
            return diarized
        else:
            # Sarvam pipeline
            diarized = self.diarizer.diarize(
                audio_path=wav_path,
                recording_id=rec_id,
                num_speakers=self.num_speakers or 2,
                mode=self.mode,
            )
            if diarized.status == "SUCCESS":
                diarized.save_json(cached_path)
            return diarized

    def process_batch(
        self,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        reuse_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Process all WAV files in the input directory through speaker diarization.
        """
        input_dir = input_dir or self.settings.processed_audio_dir
        out_dir = output_dir or (
            self.settings.onprem_pyannote_diarized_dir
            if self.provider in ("pyannote", "onprem")
            else self.settings.diarized_transcripts_dir
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        wav_files = sorted(list(input_dir.glob("*.wav")))
        if not wav_files:
            logger.warning("No WAV files found in %s", input_dir)
            return self._build_summary([], out_dir, 0.0)

        logger.info(
            "Starting batch diarization: %d file(s), provider=%s, mode=%s",
            len(wav_files), self.provider, self.mode,
        )

        results: List[Dict[str, Any]] = []
        batch_start = time.perf_counter()

        for idx, wav_path in enumerate(wav_files, start=1):
            rec_id = wav_path.stem
            logger.info("[%d/%d] Processing: %s", idx, len(wav_files), wav_path.name)

            cached_path = out_dir / f"{rec_id}.json"
            if reuse_existing and cached_path.is_file():
                try:
                    cached = DiarizedTranscript.from_json(cached_path)
                    if cached.status == "SUCCESS" and cached.segments and cached.num_speakers_detected > 0:
                        logger.info("Reusing cached diarized transcript for %s", rec_id)
                        results.append(self._recording_result(cached, "CACHED"))
                        continue
                except Exception as exc:
                    logger.debug("Could not reuse cached diarization for %s: %s", rec_id, exc)

            try:
                diarized = self.process_recording(wav_path, output_dir=out_dir, reuse_existing=reuse_existing)
                status = "SUCCESS" if diarized.status == "SUCCESS" else "FAILED"
                results.append(self._recording_result(diarized, status))
            except Exception as exc:
                err_msg = str(exc)
                logger.error("Diarization error for %s: %s", rec_id, err_msg)
                results.append({
                    "recording_id": rec_id,
                    "source_file": wav_path.name,
                    "status": "FAILED",
                    "error": err_msg,
                    "num_speakers_detected": 0,
                    "num_speakers_requested": self.num_speakers,
                    "total_segments": 0,
                    "total_words": 0,
                    "duration_seconds": 0.0,
                    "processing_time_seconds": 0.0,
                    "api_cost_inr": 0.0,
                })

        total_time = round(time.perf_counter() - batch_start, 2)
        summary = self._build_summary(results, out_dir, total_time)

        # Save summary report
        summary_path = out_dir / "diarization_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        summary["summary_report_path"] = str(summary_path)

        return summary

    def _recording_result(self, diarized: DiarizedTranscript, status: str) -> Dict[str, Any]:
        result = {
            "recording_id": diarized.recording_id,
            "source_file": diarized.source_file,
            "status": status,
            "num_speakers_detected": diarized.num_speakers_detected,
            "num_speakers_requested": diarized.num_speakers_requested,
            "total_segments": len(diarized.segments),
            "total_words": diarized.total_words,
            "duration_seconds": diarized.duration,
            "processing_time_seconds": diarized.processing_time_seconds,
            "api_cost_inr": diarized.api_cost_inr or 0.0,
            "language": diarized.language,
            "speaker_stats": [
                {
                    "speaker_id": ss.speaker_id,
                    "role": ss.role,
                    "segments": ss.total_segments,
                    "words": ss.total_words,
                    "talk_time_seconds": ss.total_talk_time_seconds,
                    "talk_time_pct": ss.talk_time_percentage,
                }
                for ss in diarized.speaker_stats
            ],
        }
        if diarized.error_details:
            result["error"] = diarized.error_details
        if diarized.warnings:
            result["warnings"] = diarized.warnings
        return result

    def _build_summary(self, results: List[Dict[str, Any]], out_dir: Path, total_time: float) -> Dict[str, Any]:
        successful = sum(1 for r in results if r["status"] in ("SUCCESS", "CACHED"))
        cached = sum(1 for r in results if r["status"] == "CACHED")
        failed = sum(1 for r in results if r["status"] == "FAILED")
        total_cost = sum(r.get("api_cost_inr", 0.0) or 0.0 for r in results)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": self.provider,
            "model": getattr(self.settings, "PYANNOTE_MODEL", "community-1") if self.provider == "pyannote" else self.settings.SARVAM_MODEL,
            "mode": self.mode,
            "num_speakers_requested": self.num_speakers,
            "total_recordings": len(results),
            "successful_diarizations": successful,
            "cached_results": cached,
            "failed_diarizations": failed,
            "total_processing_time_seconds": round(total_time, 2),
            "total_api_cost_inr": round(total_cost, 4),
            "output_directory": str(out_dir),
            "results": results,
        }
