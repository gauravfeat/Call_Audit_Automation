"""
Audio Processing Orchestrator Module.

Coordinates discovery, metadata extraction, quality assessment, audio standardization,
and persistent audit reporting (JSON & CSV).
"""

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from app.audio.exceptions import AudioProcessingError
from app.audio.ffmpeg_utils import verify_ffmpeg_installation
from app.audio.metadata import SUPPORTED_AUDIO_EXTENSIONS, AudioMetadata, extract_metadata
from app.audio.preprocess import preprocess_audio
from app.audio.quality import AudioQualityMetrics, analyze_audio_quality
from app.core.config import get_settings

logger = logging.getLogger("call_audit.audio.processor")


class AudioFileReport(BaseModel):
    """
    Comprehensive audit report for a single audio recording.
    """

    original_filename: str = Field(description="Name of the original source recording file")
    original_metadata: Optional[AudioMetadata] = Field(default=None, description="Metadata of original file")
    processed_filename: Optional[str] = Field(default=None, description="Name of the standardized WAV file")
    processed_metadata: Optional[AudioMetadata] = Field(default=None, description="Metadata of standardized WAV file")
    audio_quality_metrics: Optional[AudioQualityMetrics] = Field(default=None, description="Audio quality metrics")
    processing_status: str = Field(description="Processing outcome: SUCCESS, WARNING, or FAILED")
    warnings: List[str] = Field(default_factory=list, description="Consolidated quality and compatibility warnings")
    processing_errors: List[str] = Field(default_factory=list, description="Errors encountered during processing")


class AudioPipelineSummary(BaseModel):
    """
    Aggregated summary of an audio batch processing run.
    """

    timestamp: str = Field(description="ISO timestamp of pipeline execution")
    total_recordings_discovered: int = Field(description="Count of discovered input audio files")
    successfully_processed: int = Field(description="Count of successfully standardized recordings")
    failed_recordings: int = Field(description="Count of recordings that failed processing")
    recordings_with_warnings: int = Field(description="Count of processed recordings with quality warnings")
    output_directory: str = Field(description="Directory containing standardized 16kHz WAV files")
    report_path_json: str = Field(description="Path to detailed JSON quality report")
    report_path_csv: str = Field(description="Path to tabular CSV summary report")
    reports: List[AudioFileReport] = Field(default_factory=list, description="Per-file detailed audit reports")


class AudioProcessor:
    """
    Orchestrates the entire Milestone 2 Audio Processing workflow.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def discover_recordings(self, input_dir: Optional[Union[str, Path]] = None) -> List[Path]:
        """
        Discover all supported audio files inside the specified or configured recordings directory.

        Args:
            input_dir: Optional custom input directory. Defaults to settings.recordings_dir.

        Returns:
            Sorted list of resolved audio file paths.
        """
        target_dir = Path(input_dir).resolve() if input_dir else self.settings.recordings_dir
        if not target_dir.exists():
            logger.warning("Recordings directory does not exist: %s", target_dir)
            return []

        discovered: List[Path] = []
        for file_path in target_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                discovered.append(file_path)

        discovered.sort(key=lambda p: p.name)
        logger.info("Discovered %d audio files in %s", len(discovered), target_dir)
        return discovered

    def process_recording(
        self,
        recording_path: Path,
        output_dir: Optional[Path] = None,
    ) -> AudioFileReport:
        """
        Execute full analysis and preprocessing pipeline for a single recording:
        1. Original metadata extraction
        2. Audio quality analysis
        3. Audio standardization (16kHz mono WAV)
        4. Standardized metadata extraction
        5. Warning consolidation and status determination

        Never raises unhandled exceptions; errors are captured in the report.

        Args:
            recording_path: Path to the target audio file.
            output_dir: Optional custom destination directory for processed audio.

        Returns:
            AudioFileReport populated with metrics, metadata, warnings, and status.
        """
        filename = recording_path.name
        warnings: List[str] = []
        errors: List[str] = []
        orig_metadata: Optional[AudioMetadata] = None
        proc_metadata: Optional[AudioMetadata] = None
        quality_metrics: Optional[AudioQualityMetrics] = None
        processed_file_path: Optional[Path] = None

        logger.info("Processing recording: %s", filename)

        # Step 1: Extract original metadata
        try:
            orig_metadata = extract_metadata(recording_path)
            # Channel and sample rate notice
            if orig_metadata.channels > 1:
                warnings.append(
                    f"Stereo recording detected ({orig_metadata.channels} channels); "
                    "downmixing to mono for transcription compatibility."
                )
            if orig_metadata.sample_rate < 16000:
                warnings.append(
                    f"Telephony low sample rate detected ({orig_metadata.sample_rate} Hz); "
                    "resampling up to 16,000 Hz."
                )
        except Exception as exc:
            err_msg = f"Metadata extraction failed: {exc}"
            logger.error("Error for %s: %s", filename, err_msg)
            errors.append(err_msg)

        # Step 2: Quality Analysis
        if orig_metadata:
            try:
                quality_metrics = analyze_audio_quality(
                    recording_path,
                    known_duration=orig_metadata.duration_seconds,
                )
                warnings.extend(quality_metrics.warnings)
            except Exception as exc:
                err_msg = f"Audio quality analysis failed: {exc}"
                logger.error("Error for %s: %s", filename, err_msg)
                errors.append(err_msg)

        # Step 3: Audio Preprocessing / Conversion to 16kHz Mono WAV
        try:
            processed_file_path = preprocess_audio(
                input_path=recording_path,
                output_dir=output_dir,
            )
        except Exception as exc:
            err_msg = f"Audio preprocessing failed: {exc}"
            logger.error("Error for %s: %s", filename, err_msg)
            errors.append(err_msg)

        # Step 4: Extract metadata of processed file to verify conformity
        if processed_file_path and processed_file_path.is_file():
            try:
                proc_metadata = extract_metadata(processed_file_path)
            except Exception as exc:
                err_msg = f"Failed to extract metadata for processed audio: {exc}"
                logger.error("Error for %s: %s", filename, err_msg)
                errors.append(err_msg)

        # Step 5: Determine overall processing status
        if errors:
            status = "FAILED"
        elif warnings:
            status = "WARNING"
        else:
            status = "SUCCESS"

        report = AudioFileReport(
            original_filename=filename,
            original_metadata=orig_metadata,
            processed_filename=processed_file_path.name if processed_file_path else None,
            processed_metadata=proc_metadata,
            audio_quality_metrics=quality_metrics,
            processing_status=status,
            warnings=warnings,
            processing_errors=errors,
        )

        logger.info("Finished %s -> Status: %s (Warnings: %d, Errors: %d)", filename, status, len(warnings), len(errors))
        return report

    def process_all(
        self,
        input_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> AudioPipelineSummary:
        """
        Process all discovered audio recordings and generate JSON and CSV audit reports.

        Args:
            input_dir: Optional custom input directory.
            output_dir: Optional custom output directory for processed files.

        Returns:
            AudioPipelineSummary containing batch statistics and file reports.

        Raises:
            AudioProcessingError: If FFmpeg verification fails.
        """
        # Ensure directories
        self.settings.ensure_directories()

        # Verify FFmpeg & FFprobe
        verify_ffmpeg_installation()

        resolved_input_dir = Path(input_dir).resolve() if input_dir else self.settings.recordings_dir
        resolved_output_dir = Path(output_dir).resolve() if output_dir else self.settings.processed_audio_dir

        recordings = self.discover_recordings(resolved_input_dir)
        reports: List[AudioFileReport] = []

        for recording in recordings:
            file_report = self.process_recording(recording, output_dir=resolved_output_dir)
            reports.append(file_report)

        # Calculate summary metrics
        total = len(recordings)
        failed = sum(1 for r in reports if r.processing_status == "FAILED")
        with_warnings = sum(1 for r in reports if r.processing_status == "WARNING")
        success = sum(1 for r in reports if r.processing_status in ("SUCCESS", "WARNING"))

        # Save JSON Report
        json_report_path = self.settings.audio_reports_dir / "audio_quality_report.json"
        summary_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_recordings_discovered": total,
            "successfully_processed": success,
            "failed_recordings": failed,
            "recordings_with_warnings": with_warnings,
            "output_directory": str(resolved_output_dir),
            "reports": [r.model_dump() for r in reports],
        }

        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, ensure_ascii=False)
        logger.info("Saved JSON quality report to: %s", json_report_path)

        # Save CSV Summary
        csv_report_path = self.settings.audio_reports_dir / "audio_quality_summary.csv"
        self._write_csv_summary(csv_report_path, reports)
        logger.info("Saved CSV summary report to: %s", csv_report_path)

        summary = AudioPipelineSummary(
            timestamp=summary_payload["timestamp"],
            total_recordings_discovered=total,
            successfully_processed=success,
            failed_recordings=failed,
            recordings_with_warnings=with_warnings,
            output_directory=str(resolved_output_dir),
            report_path_json=str(json_report_path),
            report_path_csv=str(csv_report_path),
            reports=reports,
        )

        return summary

    def _write_csv_summary(self, csv_path: Path, reports: List[AudioFileReport]) -> None:
        """
        Write tabular summary of all processed audio files to a CSV file.
        """
        fieldnames = [
            "original_filename",
            "processing_status",
            "duration_seconds",
            "original_format",
            "original_sample_rate",
            "mean_volume_db",
            "peak_amplitude_db",
            "is_clipped",
            "silence_percentage",
            "longest_silence_seconds",
            "processed_filename",
            "warnings_count",
            "warnings",
            "errors",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in reports:
                orig = r.original_metadata
                qm = r.audio_quality_metrics
                writer.writerow(
                    {
                        "original_filename": r.original_filename,
                        "processing_status": r.processing_status,
                        "duration_seconds": orig.duration_seconds if orig else "",
                        "original_format": orig.audio_format if orig else "",
                        "original_sample_rate": orig.sample_rate if orig else "",
                        "mean_volume_db": qm.mean_volume_db if qm else "",
                        "peak_amplitude_db": qm.peak_amplitude_db if qm else "",
                        "is_clipped": qm.is_clipped if qm else "",
                        "silence_percentage": qm.silence_percentage if qm else "",
                        "longest_silence_seconds": qm.longest_silence_seconds if qm else "",
                        "processed_filename": r.processed_filename or "",
                        "warnings_count": len(r.warnings),
                        "warnings": " | ".join(r.warnings),
                        "errors": " | ".join(r.processing_errors),
                    }
                )
