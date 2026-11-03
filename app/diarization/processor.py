"""
Diarization Processor Module.

Orchestrates batch speaker diarization across multiple audio recordings.
Handles discovery, caching, error recovery, and summary reporting.
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
    Batch processor for speaker diarization using Sarvam Saaras v4.

    Features:
    - Sequential processing of multiple audio files.
    - Caching: skips recordings with existing valid diarized transcripts.
    - Per-recording error isolation (one failure doesn't block the batch).
    - Summary report generation.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        mode: str = "codemix",
        num_speakers: int = 2,
    ):
        self.settings = settings or get_settings()
        self.mode = mode
        self.num_speakers = num_speakers

        # Initialize the Sarvam diarizer
        self.diarizer = SarvamDiarizer(
            api_key=self.settings.SARVAM_API_KEY,
            model=self.settings.SARVAM_MODEL,
            mode=self.mode,
            num_speakers=self.num_speakers,
        )

    def process_batch(
        self,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        reuse_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Process all WAV files in the input directory through speaker diarization.

        Args:
            input_dir: Directory with preprocessed WAV files. Defaults to processed_audio_dir.
            output_dir: Output directory for diarized transcripts. Defaults to diarized_transcripts_dir.
            reuse_existing: If True, skip recordings with valid cached results.

        Returns:
            Summary dict with batch processing results.
        """
        input_dir = input_dir or self.settings.processed_audio_dir
        output_dir = output_dir or self.settings.diarized_transcripts_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Discover WAV files
        wav_files = sorted(list(input_dir.glob("*.wav")))
        if not wav_files:
            logger.warning("No WAV files found in %s", input_dir)
            return self._build_summary([], output_dir, 0.0)

        logger.info(
            "Starting batch diarization: %d file(s), mode=%s, num_speakers=%d",
            len(wav_files), self.mode, self.num_speakers,
        )

        results: List[Dict[str, Any]] = []
        batch_start = time.perf_counter()

        for idx, wav_path in enumerate(wav_files, start=1):
            rec_id = wav_path.stem
            logger.info(
                "[%d/%d] Processing: %s", idx, len(wav_files), wav_path.name
            )

            # Check for cached result
            cached_path = output_dir / f"{rec_id}.json"
            if reuse_existing and cached_path.is_file():
                try:
                    cached = DiarizedTranscript.from_json(cached_path)
                    if (
                        cached.status == "SUCCESS"
                        and cached.segments
                        and cached.num_speakers_detected > 0
                    ):
                        logger.info(
                            "Reusing cached diarized transcript for %s "
                            "(%d speakers, %d segments)",
                            rec_id,
                            cached.num_speakers_detected,
                            len(cached.segments),
                        )
                        results.append(
                            self._recording_result(cached, "CACHED")
                        )
                        continue
                except Exception as exc:
                    logger.debug(
                        "Could not reuse cached diarization for %s: %s",
                        rec_id, exc,
                    )

            # Run diarization
            try:
                diarized = self.diarizer.diarize(
                    audio_path=wav_path,
                    recording_id=rec_id,
                    num_speakers=self.num_speakers,
                    mode=self.mode,
                )

                if diarized.status == "SUCCESS":
                    # Save output
                    out_path = diarized.save_json(cached_path)
                    logger.info(
                        "Diarization SUCCESS for %s: %d speakers, %d segments, "
                        "%.1fs processing, saved to %s",
                        rec_id,
                        diarized.num_speakers_detected,
                        len(diarized.segments),
                        diarized.processing_time_seconds,
                        out_path,
                    )
                    results.append(self._recording_result(diarized, "SUCCESS"))
                else:
                    logger.error(
                        "Diarization FAILED for %s: %s",
                        rec_id, diarized.error_details,
                    )
                    results.append(self._recording_result(diarized, "FAILED"))

            except Exception as exc:
                err_msg = str(exc)
                logger.error("Diarization error for %s: %s", rec_id, err_msg)
                results.append({
                    "recording_id": rec_id,
                    "source_file": wav_path.name,
                    "status": "FAILED",
                    "error": err_msg,
                })

        batch_elapsed = round(time.perf_counter() - batch_start, 2)
        return self._build_summary(results, output_dir, batch_elapsed)

    def _recording_result(
        self,
        diarized: DiarizedTranscript,
        status: str,
    ) -> Dict[str, Any]:
        """Build per-recording result dict."""
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
            "api_cost_inr": diarized.api_cost_inr,
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

    def _build_summary(
        self,
        results: List[Dict[str, Any]],
        output_dir: Path,
        total_time: float,
    ) -> Dict[str, Any]:
        """Build and save the batch diarization summary."""
        successful = sum(1 for r in results if r["status"] in ("SUCCESS", "CACHED"))
        cached = sum(1 for r in results if r["status"] == "CACHED")
        failed = sum(1 for r in results if r["status"] == "FAILED")
        total_cost = sum(r.get("api_cost_inr", 0.0) or 0.0 for r in results)

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": "sarvam",
            "model": self.settings.SARVAM_MODEL,
            "mode": self.mode,
            "num_speakers_requested": self.num_speakers,
            "total_recordings": len(results),
            "successful_diarizations": successful,
            "cached_results": cached,
            "failed_diarizations": failed,
            "total_processing_time_seconds": total_time,
            "total_api_cost_inr": round(total_cost, 4),
            "output_directory": str(output_dir),
            "results": results,
        }

        summary_path = output_dir / "diarization_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info("Diarization summary saved to: %s", summary_path)
        summary["summary_report_path"] = str(summary_path)
        return summary
