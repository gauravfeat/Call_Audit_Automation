#!/usr/bin/env python
"""
M4 Speaker Diarization CLI Script.

Processes preprocessed WAV audio recordings through Sarvam Saaras v4
Batch API with speaker diarization enabled.

Usage:
    python scripts/diarize_audio.py
    python scripts/diarize_audio.py --num-speakers 2 --mode codemix
    python scripts/diarize_audio.py --no-reuse
    python scripts/diarize_audio.py --input-dir path/to/wavs --output-dir path/to/output

Outputs per-recording diarized transcript JSON files under:
    output/diarized_transcripts/<recording_id>.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.diarization.processor import DiarizationProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M4: Speaker Diarization using Sarvam Saaras v4 Batch API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing preprocessed WAV files (default: output/processed_audio/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for diarized transcripts (default: output/diarized_transcripts/)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="codemix",
        choices=["codemix", "verbatim"],
        help="Sarvam transcription mode (default: codemix)",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=2,
        help="Expected number of speakers per recording (default: 2)",
    )
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        default=False,
        help="Force re-processing even if cached diarized transcripts exist",
    )
    return parser.parse_args()


def print_banner(mode: str, num_speakers: int) -> None:
    print("=" * 80)
    print("  M4: SPEAKER DIARIZATION — Sarvam Saaras v4 Batch API")
    print("=" * 80)
    print(f"  Provider:        Sarvam Saaras v4")
    print(f"  Mode:            {mode}")
    print(f"  Num Speakers:    {num_speakers}")
    print(f"  Diarization:     ENABLED")
    print("=" * 80)


def print_summary(summary: dict) -> None:
    print()
    print("=" * 80)
    print("  DIARIZATION SUMMARY")
    print("-" * 80)
    print(f"  Total Recordings:     {summary['total_recordings']}")
    print(f"  Successful:           {summary['successful_diarizations']}")
    print(f"  Cached (reused):      {summary['cached_results']}")
    print(f"  Failed:               {summary['failed_diarizations']}")
    print(f"  Total Processing:     {summary['total_processing_time_seconds']:.1f}s")
    print(f"  Total API Cost (INR): {summary['total_api_cost_inr']:.4f}")
    print(f"  Output Directory:     {summary['output_directory']}")
    print("=" * 80)

    results = summary.get("results", [])
    if results:
        print("\nPer-Recording Breakdown:")
        print(
            f"  {'Filename':<45} {'Spkrs':<6} {'Segs':<6} {'Words':<7} "
            f"{'Duration':<9} {'Time':<8} {'Cost':<10} {'Status'}"
        )
        print("  " + "-" * 105)
        for r in results:
            fname = r.get("source_file", "N/A")[:43]
            spkrs = str(r.get("num_speakers_detected", "?"))
            segs = str(r.get("total_segments", 0))
            words = str(r.get("total_words", 0))
            dur = f"{r.get('duration_seconds', 0):.1f}s"
            proc = f"{r.get('processing_time_seconds', 0):.1f}s"
            cost = f"INR {r.get('api_cost_inr', 0.0):.4f}" if r.get("api_cost_inr") else "INR 0.0"
            status = r["status"]
            print(
                f"  {fname:<45} {spkrs:<6} {segs:<6} {words:<7} "
                f"{dur:<9} {proc:<8} {cost:<10} {status}"
            )
            if r.get("error"):
                print(f"    - [ERROR] {r['error']}")
            elif r.get("speaker_stats"):
                for ss in r["speaker_stats"]:
                    print(
                        f"    - {ss['speaker_id']}: "
                        f"{ss['segments']} segs, {ss['words']} words, "
                        f"{ss['talk_time_seconds']:.1f}s ({ss['talk_time_pct']:.1f}%)"
                    )
        print()


def main() -> int:
    """Main execution flow for batch speaker diarization."""
    args = parse_args()
    settings = get_settings()

    mode = args.mode.lower()
    num_speakers = args.num_speakers
    reuse_existing = not args.no_reuse

    print_banner(mode, num_speakers)

    # Step 1: Determine input/output directories
    input_dir = (
        Path(args.input_dir).resolve() if args.input_dir
        else settings.processed_audio_dir
    )
    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir
        else settings.diarized_transcripts_dir
    )

    print(f"\n[1/3] Discovering preprocessed audio files in: {input_dir}")
    if not input_dir.exists():
        print(f"[ERROR] Directory does not exist: {input_dir}", file=sys.stderr)
        print("Please run 'python scripts/process_audio.py' first.")
        return 1

    wav_files = sorted(list(input_dir.glob("*.wav")))
    if not wav_files:
        print(f"[ERROR] No .wav files found in {input_dir}.", file=sys.stderr)
        print("Please run 'python scripts/process_audio.py' first.")
        return 1

    print(f"  Found {len(wav_files)} recording(s) to diarize.")

    # Step 2: Initialize Processor
    print(f"\n[2/3] Initializing DiarizationProcessor (mode={mode}, num_speakers={num_speakers})...")
    processor = DiarizationProcessor(
        settings=settings,
        mode=mode,
        num_speakers=num_speakers,
    )

    # Step 3: Run Batch Diarization
    print(f"\n[3/3] Diarizing {len(wav_files)} recording(s)... (reuse_existing={reuse_existing})")
    summary = processor.process_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        reuse_existing=reuse_existing,
    )

    # Step 4: Display Summary
    print_summary(summary)

    failed = summary.get("failed_diarizations", 0)
    successful = summary.get("successful_diarizations", 0)

    if failed > 0 and successful == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
