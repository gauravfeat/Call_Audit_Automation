#!/usr/bin/env python
"""
M4 Speaker Diarization Quality Validation CLI.

Validates all 5 diarized transcripts for:
- Timestamp consistency against audio duration.
- Speaker consistency, alternation patterns, and talk-time distribution.
- Overlapping speech duration and cross-talk vs heavy overlaps.
- Unassigned speech segments and measurable unassigned duration.
- Suspiciously short segments (< 0.4s).
- Speaker statistics verification.
- Text preservation compared to M3 primary STT transcripts.
- Manual auditor review recommendation.

Outputs comprehensive JSON report to:
    output/diarized_transcripts/diarization_quality_report.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.diarization.quality_validator import DiarizationQualityValidator
from app.diarization.role_mapper import RoleMappingManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M4: Speaker Diarization Quality Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--diarized-dir",
        type=str,
        default=None,
        help="Directory containing diarized transcripts (default: output/diarized_transcripts/)",
    )
    parser.add_argument(
        "--primary-dir",
        type=str,
        default=None,
        help="Directory containing primary M3 transcripts (default: output/transcripts/sarvam_v4_codemix/)",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Path for quality validation JSON report (default: output/diarized_transcripts/diarization_quality_report.json)",
    )
    return parser.parse_args()


def print_banner() -> None:
    print("=" * 90)
    print("  M4: SPEAKER DIARIZATION QUALITY VALIDATION REPORT")
    print("  Engine: Sarvam Saaras v4 Batch API (codemix) | Ground Truth: None Available")
    print("=" * 90)


def print_report_summary(report) -> None:
    s = report.summary
    print("\n" + "=" * 90)
    print("  VALIDATION SUMMARY")
    print("-" * 90)
    print(f"  Total Recordings Validated:        {s['total_recordings_validated']}")
    print(f"  Total Audio Duration:             {s['total_audio_duration_seconds']:.1f}s ({s['total_audio_duration_seconds']/60:.1f} min)")
    print(f"  Total Speech Duration:            {s['total_speech_duration_seconds']:.1f}s")
    print(f"  Total Unassigned / Pause Duration:{s['total_unassigned_duration_seconds']:.1f}s (avg {s['average_unassigned_percentage']:.1f}%)")
    print(f"  Total Simultaneous Speech:        {s['total_overlapping_speech_seconds']:.1f}s (avg {s['average_overlap_percentage']:.1f}%)")
    print(f"  All Timestamps Valid:             {'YES (0 anomalies)' if s['timestamp_validation_all_passed'] else 'NO (anomalies found)'}")
    print(f"  Recordings Flagged for Review:    {s['recordings_requiring_manual_review']} of {s['total_recordings_validated']}")
    print(f"  Overall Status:                   {s['overall_status']}")
    print("=" * 90)

    print("\nPER-RECORDING QUALITY BREAKDOWN:")
    print(
        f"  {'Filename':<35} {'Dur':<7} {'Segs':<5} {'Speakers':<12} "
        f"{'Speech%':<8} {'Overl%':<7} {'Text Status':<12} {'Review Req'}"
    )
    print("  " + "-" * 98)

    for r in report.recordings:
        fname = r.source_file[:33]
        dur = f"{r.audio_duration_seconds:.1f}s"
        segs = str(r.segment_count)
        spks = f"{len(r.detected_speakers)} spkrs"
        speech_pct = f"{100 - r.unassigned_percentage:.1f}%"
        ov_pct = f"{r.overlap_metrics.simultaneous_speech_percentage:.1f}%"
        txt_status = r.text_preservation.status
        rev_req = "YES" if r.manual_review_required else "NO"

        print(
            f"  {fname:<35} {dur:<7} {segs:<5} {spks:<12} "
            f"{speech_pct:<8} {ov_pct:<7} {txt_status:<12} {rev_req}"
        )

        # Print consistency / disparity details
        for w in r.speaker_consistency_warnings:
            print(f"    - [CONSISTENCY] {w}")

        # Print overlap details
        for w in r.overlap_warnings:
            print(f"    - [OVERLAP] {w}")

        # Print review reasons
        if r.manual_review_reasons:
            for reason in r.manual_review_reasons:
                print(f"    - [MANUAL REVIEW REASON] {reason}")

        # Print text preservation delta
        tp = r.text_preservation
        if tp.primary_words is not None:
            print(f"    - [TEXT] Diarized: {tp.diarized_words} words vs Primary: {tp.primary_words} words (Delta: {tp.word_delta:+d} words, {tp.notes})")

    print()


def main() -> int:
    args = parse_args()
    settings = get_settings()

    diarized_dir = (
        Path(args.diarized_dir).resolve()
        if args.diarized_dir
        else settings.diarized_transcripts_dir
    )
    primary_dir = (
        Path(args.primary_dir).resolve()
        if args.primary_dir
        else settings.primary_transcripts_dir
    )
    output_report_path = (
        Path(args.output_report).resolve()
        if args.output_report
        else diarized_dir / "diarization_quality_report.json"
    )

    print_banner()
    print(f"\n[1/3] Scanning diarized transcripts in: {diarized_dir}")
    if not diarized_dir.is_dir():
        print(f"[ERROR] Diarized directory not found: {diarized_dir}", file=sys.stderr)
        return 1

    # Also initialize manual role mapping template if not present
    print("\n[2/3] Checking manual speaker role mapping registry...")
    role_mgr = RoleMappingManager(storage_path=diarized_dir / "speaker_role_mappings.json")
    registry = role_mgr.initialize_from_transcripts(diarized_dir, overwrite=False)
    print(f"  Role Registry: {registry.total_recordings} recording(s) initialized.")
    print(f"  Pending Auditor Reviews: {registry.pending_reviews} | Completed: {registry.completed_reviews}")

    # Run validation
    print(f"\n[3/3] Running quality validation across all recordings...")
    validator = DiarizationQualityValidator()
    report = validator.validate_batch(
        diarized_dir=diarized_dir,
        primary_dir=primary_dir,
        output_report_path=output_report_path,
    )

    # Print summary
    print_report_summary(report)
    print(f"Detailed quality report saved to:\n  {output_report_path}")
    print(f"Speaker role mappings stored in:\n  {role_mgr.storage_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
