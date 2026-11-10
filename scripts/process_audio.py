"""
Audio Preprocessing CLI Script.

Executes the Milestone 2 Audio Processing Pipeline:
- Discovers raw recordings in input/recordings/
- Extracts metadata and calculates audio quality metrics
- Converts each recording to standardized 16kHz mono WAV format
- Generates JSON and CSV quality reports in output/audio_reports/
- Prints an executive console summary

Usage:
    python scripts/process_audio.py [--input-dir PATH] [--output-dir PATH]
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.audio.exceptions import AudioProcessingError, FFmpegNotFoundError
from app.audio.ffmpeg_utils import verify_ffmpeg_installation
from app.audio.processor import AudioProcessor
from app.core.config import get_settings


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Call Audit Automation - Audio Preprocessing & Quality Analysis CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Custom directory containing raw audio recordings (default: input/recordings)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom directory for processed 16kHz WAV files (default: output/processed_audio)",
    )
    parser.add_argument(
        "--recording",
        type=str,
        default=None,
        help="Specific recording ID or filename to process (e.g. 'MOCK_CALL_002').",
    )
    parser.add_argument(
        "--stt",
        action="store_true",
        default=False,
        help="Automatically run Speech-to-Text transcription after audio preprocessing.",
    )
    parser.add_argument(
        "--diarization",
        action="store_true",
        default=False,
        help="Automatically run Speaker Diarization after audio preprocessing.",
    )
    parser.add_argument(
        "--stt-provider",
        type=str,
        default="whisper_api",
        choices=["whisper_api", "sarvam", "whisper"],
        help="Speech-to-Text provider if --stt is enabled (default: 'whisper_api').",
    )
    parser.add_argument(
        "--diarization-provider",
        type=str,
        default="pyannote",
        choices=["pyannote", "sarvam"],
        help="Diarization provider if --diarization is enabled (default: 'pyannote').",
    )
    return parser.parse_args()


def print_banner() -> None:
    """Print executive CLI header."""
    print("=" * 80)
    print("  CALL AUDIT AUTOMATION - AUDIO PREPROCESSING & QUALITY PIPELINE (M2)")
    print("=" * 80)


def print_summary(summary) -> None:
    """Format and display executive processing metrics."""
    print("\n" + "=" * 80)
    print("  AUDIO PROCESSING EXECUTION SUMMARY")
    print("=" * 80)
    print(f"  Total recordings discovered:     {summary.total_recordings_discovered}")
    print(f"  Successfully processed:          {summary.successfully_processed}")
    print(f"  Failed recordings:               {summary.failed_recordings}")
    print(f"  Recordings with quality warnings: {summary.recordings_with_warnings}")
    print("-" * 80)
    print(f"  Processed Audio Directory:       {summary.output_directory}")
    print(f"  JSON Quality Report:             {summary.report_path_json}")
    print(f"  CSV Quality Summary:             {summary.report_path_csv}")
    print("=" * 80)

    if summary.reports:
        print("\nPer-Recording Breakdown:")
        print(f"  {'Filename':<45} {'Duration':<10} {'Status':<10} {'Warnings'}")
        print("  " + "-" * 76)
        for r in summary.reports:
            dur_str = f"{r.original_metadata.duration_seconds:.1f}s" if r.original_metadata else "N/A"
            warn_count = len(r.warnings)
            status_display = r.processing_status
            print(f"  {r.original_filename[:43]:<45} {dur_str:<10} {status_display:<10} {warn_count} warning(s)")
            for w in r.warnings:
                print(f"    - [WARN] {w}")
            for e in r.processing_errors:
                print(f"    - [ERROR] {e}")
        print()


def main() -> int:
    """Main CLI execution flow."""
    args = parse_args()
    print_banner()

    # Step 1: Pre-flight check for FFmpeg & FFprobe
    print("\n[1/3] Verifying FFmpeg & FFprobe system binaries...")
    try:
        binaries = verify_ffmpeg_installation()
        for name, info in binaries.items():
            print(f"  + {name}: {info}")
    except FFmpegNotFoundError as exc:
        print(f"\n[ERROR] FFmpeg Verification Failed:\n{exc}", file=sys.stderr)
        return 1

    # Step 2: Initialize processor and discover recordings
    settings = get_settings()
    input_path = Path(args.input_dir).resolve() if args.input_dir else settings.recordings_dir
    output_path = Path(args.output_dir).resolve() if args.output_dir else settings.processed_audio_dir

    print(f"\n[2/3] Discovering recordings in: {input_path}")
    processor = AudioProcessor()
    discovered = processor.discover_recordings(input_path)
    if not discovered:
        print(f"  ! No supported audio files (.mp3, .wav, .m4a) found in {input_path}.")
        print("  Please place client recordings into input/recordings/ and re-run.")
        return 0

    print(f"  Found {len(discovered)} recording(s) to process.")

    # Step 3: Run pipeline
    print(f"\n[3/3] Processing recordings and standardizing to 16kHz mono WAV...")
    if args.recording:
        target_stem = Path(args.recording).stem
        discovered = [f for f in discovered if f.stem == target_stem]
        if not discovered:
            print(f"[ERROR] Specified recording '{args.recording}' not found in {input_path}.", file=sys.stderr)
            return 1

    try:
        if args.recording:
            report = processor.process_recording(discovered[0], output_dir=output_path)
            from datetime import datetime, timezone
            from app.audio.processor import AudioPipelineSummary
            summary = AudioPipelineSummary(
                timestamp=datetime.now(timezone.utc).isoformat(),
                total_recordings_discovered=1,
                successfully_processed=1 if report.processing_status in ("SUCCESS", "WARNING") else 0,
                failed_recordings=1 if report.processing_status == "FAILED" else 0,
                recordings_with_warnings=1 if report.processing_status == "WARNING" else 0,
                output_directory=str(output_path),
                report_path_json=str(settings.audio_reports_dir / "audio_quality_report.json"),
                report_path_csv=str(settings.audio_reports_dir / "audio_quality_summary.csv"),
                reports=[report],
            )
        else:
            summary = processor.process_all(input_dir=input_path, output_dir=output_path)
    except AudioProcessingError as exc:
        print(f"\n[FATAL ERROR] Audio pipeline failed: {exc}", file=sys.stderr)
        return 1

    # Step 4: Display executive summary
    print_summary(summary)

    # Optional Step 5: Automatic STT Transcription
    if args.stt and summary.successfully_processed > 0:
        print("\n" + "=" * 80)
        print(f"  AUTOMATIC STT TRANSCRIPTION (provider='{args.stt_provider}')")
        print("=" * 80)
        from app.transcription.transcriber import AudioTranscriber
        stt_transcriber = AudioTranscriber(provider=args.stt_provider, settings=settings)
        for r in summary.reports:
            if r.processing_status in ("SUCCESS", "WARNING") and r.processed_filename:
                target_file = output_path / r.processed_filename
                try:
                    print(f"  Transcribing {r.processed_filename}...")
                    t = stt_transcriber.transcribe_file(target_file)
                    print(f"  [SUCCESS] {t.recording_id}: {len(t.segments)} segments, duration {t.duration:.1f}s")
                except Exception as exc:
                    print(f"  [ERROR] STT failed for {r.original_filename}: {exc}", file=sys.stderr)

    # Optional Step 6: Automatic Diarization
    if args.diarization and summary.successfully_processed > 0:
        print("\n" + "=" * 80)
        print(f"  AUTOMATIC SPEAKER DIARIZATION (provider='{args.diarization_provider}')")
        print("=" * 80)
        from app.diarization.processor import DiarizationProcessor
        diar_processor = DiarizationProcessor(provider=args.diarization_provider, settings=settings)
        for r in summary.reports:
            if r.processing_status in ("SUCCESS", "WARNING") and r.processed_filename:
                target_file = output_path / r.processed_filename
                try:
                    print(f"  Diarizing {r.processed_filename}...")
                    d = diar_processor.process_recording(target_file)
                    print(f"  [SUCCESS] {d.recording_id}: {d.num_speakers_detected} speakers, {len(d.segments)} segments")
                except Exception as exc:
                    print(f"  [ERROR] Diarization failed for {r.original_filename}: {exc}", file=sys.stderr)

    if summary.failed_recordings > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
