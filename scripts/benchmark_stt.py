#!/usr/bin/env python3
"""
STT Benchmark CLI.

Runs ASR benchmarking across Whisper Small (baseline), Whisper Large-v3,
and Sarvam Saaras v4 on the project's mock recordings.

Usage:
    python scripts/benchmark_stt.py --model all
    python scripts/benchmark_stt.py --model sarvam --sarvam-mode both
    python scripts/benchmark_stt.py --model whisper-large --reuse
    python scripts/benchmark_stt.py --model baseline
    python scripts/benchmark_stt.py --model sarvam --sarvam-mode codemix --reuse
"""

import argparse
import logging
import os
import sys
import time

# Fix Windows console encoding for Hindi/Devanagari output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is on sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, ".env"))


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)-40s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STT Benchmark CLI — Compare ASR models on project recordings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "baseline", "sarvam", "whisper-large"],
        help="Which model(s) to benchmark (default: all).",
    )
    parser.add_argument(
        "--sarvam-mode",
        type=str,
        default="both",
        choices=["codemix", "verbatim", "both"],
        help="Sarvam transcription mode (default: both).",
    )
    parser.add_argument(
        "--audio-type",
        type=str,
        default="wav",
        choices=["wav", "mp3"],
        help="Audio input type: wav (16kHz processed) or mp3 (8kHz original). Default: wav.",
    )
    parser.add_argument(
        "--consolidate",
        action="store_true",
        help="Load all existing cached results across all models and generate consolidated reports.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse previously generated results if available.",
    )
    parser.add_argument(
        "--recording",
        type=str,
        default=None,
        help="Specific recording stem to benchmark (default: all recordings).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    for group_key, group_data in summary.get("results_by_model", {}).items():
        print(f"\n  [{group_key}]")
        print(f"    Model:            {group_data['model_name']}")
        print(f"    Mode:             {group_data['transcription_mode']}")
        print(f"    Recordings:       {group_data['total_recordings']} "
              f"(OK: {group_data['successful']}, FAIL: {group_data['failed']})")
        print(f"    Avg Time:         {group_data['avg_processing_time_seconds']:.2f}s")
        print(f"    Avg Words:        {group_data['avg_word_count']:.0f}")
        print(f"    API Cost (INR):   {group_data['total_api_cost_inr']:.4f}")
        print(f"    Has Timestamps:   {group_data['has_timestamps']}")
        print(f"    Repetitive:       {group_data['repetitive_transcripts']}")
        print(f"    Fin. Term Cov.:   {group_data['avg_financial_term_coverage']:.2%}")

    print("\n" + "=" * 70)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("benchmark_stt")

    overall_start = time.perf_counter()

    from app.benchmark.evaluator import BenchmarkRunner
    runner = BenchmarkRunner()

    # If --consolidate was passed, simply load all cached results and report
    if args.consolidate:
        logger.info("Consolidating all cached benchmark results...")
        cached_results = runner.load_all_cached_results()
        if not cached_results:
            logger.warning("No cached benchmark results found.")
            return 0
        runner.evaluate_all()
        saved = runner.save_reports()
        for name, path in saved.items():
            logger.info("Saved: %s -> %s", name, path)
        summary = runner.generate_summary()
        _print_summary(summary)
        return 0

    logger.info("=" * 70)
    logger.info("STT Benchmark Runner")
    logger.info("=" * 70)
    logger.info("Model: %s | Sarvam Mode: %s | Audio: %s | Reuse: %s",
                args.model, args.sarvam_mode, args.audio_type, args.reuse)

    # Discover recordings
    audio_files = runner.discover_recordings(audio_type=args.audio_type)
    if not audio_files:
        logger.error("No audio recordings found. Aborting.")
        return 1

    # Filter to specific recording if requested
    if args.recording:
        audio_files = [f for f in audio_files if args.recording in f.stem]
        if not audio_files:
            logger.error("Recording '%s' not found.", args.recording)
            return 1
        logger.info("Filtered to recording: %s", audio_files[0].stem)

    logger.info("Benchmarking %d recording(s).", len(audio_files))

    # Determine which models to run
    run_baseline = args.model in ("all", "baseline")
    run_sarvam = args.model in ("all", "sarvam")
    run_whisper_large = args.model in ("all", "whisper-large")

    # ---------- Baseline: Whisper Small ----------
    if run_baseline:
        logger.info("-" * 60)
        logger.info("Running BASELINE (Whisper Small)...")
        try:
            runner.run_model(
                model_key="baseline",
                audio_files=audio_files,
                audio_type=args.audio_type,
                reuse=args.reuse,
            )
            logger.info("Baseline completed.")
        except Exception as exc:
            logger.error("Baseline failed: %s", exc)

    # ---------- Sarvam Saaras v4 ----------
    if run_sarvam:
        from app.core.config import get_settings
        settings = get_settings()
        if not settings.SARVAM_API_KEY:
            logger.warning("SARVAM_API_KEY not set. Skipping Sarvam benchmarks.")
        else:
            sarvam_modes = []
            if args.sarvam_mode in ("codemix", "both"):
                sarvam_modes.append("codemix")
            if args.sarvam_mode in ("verbatim", "both"):
                sarvam_modes.append("verbatim")

            for mode in sarvam_modes:
                logger.info("-" * 60)
                logger.info("Running SARVAM SAARAS v4 (mode=%s)...", mode)
                try:
                    runner.run_model(
                        model_key=f"sarvam-{mode}",
                        audio_files=audio_files,
                        audio_type=args.audio_type,
                        reuse=args.reuse,
                        mode=mode,
                    )
                    logger.info("Sarvam %s completed.", mode)
                except Exception as exc:
                    logger.error("Sarvam %s failed: %s", mode, exc)

    # ---------- Whisper Large-v3 ----------
    if run_whisper_large:
        logger.info("-" * 60)
        logger.info("Running WHISPER LARGE-V3...")
        try:
            runner.run_model(
                model_key="whisper-large",
                audio_files=audio_files,
                audio_type=args.audio_type,
                reuse=args.reuse,
            )
            logger.info("Whisper Large-v3 completed.")
        except Exception as exc:
            logger.error("Whisper Large-v3 failed: %s", exc)

    # ---------- Evaluate and Report ----------
    logger.info("=" * 60)
    logger.info("Evaluating results...")
    runner.evaluate_all()
    saved = runner.save_reports()

    total_time = round(time.perf_counter() - overall_start, 2)

    logger.info("=" * 70)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 70)
    logger.info("Total runs: %d", len(runner.results))
    logger.info("Total time: %.2fs", total_time)

    for name, path in saved.items():
        logger.info("Saved: %s -> %s", name, path)

    # Print summary to console
    summary = runner.generate_summary()
    _print_summary(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
