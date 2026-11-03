"""
Unit Tests for M4 Diarization Quality Validation and Manual Speaker Role Mapping.

Tests cover:
- Timestamp validation (bounds, ordering, non-inversion).
- Overlap detection and simultaneous speech calculation.
- Suspicious short segments (< 0.4s) and empty text handling.
- Speaker consistency, turn alternations, and talk-time disparity flags.
- Text preservation checks against primary transcripts.
- Role mapping registry initialization (UNKNOWN default).
- Role assignment validation (RM, CUSTOMER, UNKNOWN) and review status updates.
- Applying manual role mappings to DiarizedTranscript.
- End-to-end quality validation batch reporting.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.diarization.diarization_schema import (
    DiarizedSegment,
    DiarizedTranscript,
    SpeakerStats,
)
from app.diarization.quality_validator import (
    DiarizationQualityValidator,
    merge_intervals,
)
from app.diarization.role_mapper import (
    RoleMappingManager,
    RecordingRoleMapping,
    SpeakerRoleRegistry,
    VALID_ROLES,
)


# ======================== Helpers ========================

def _create_test_transcript(
    duration: float = 30.0,
    segments: list = None,
    recording_id: str = "test_rec_01",
) -> DiarizedTranscript:
    if segments is None:
        segments = [
            DiarizedSegment(
                segment_id=0,
                speaker_id="SPEAKER_00",
                start=0.5,
                end=5.0,
                text="Namaste HDFC Securities se bol raha hoon.",
            ),
            DiarizedSegment(
                segment_id=1,
                speaker_id="SPEAKER_01",
                start=5.2,
                end=8.0,
                text="Haan boliye sir.",
            ),
            DiarizedSegment(
                segment_id=2,
                speaker_id="SPEAKER_00",
                start=8.2,
                end=15.0,
                text="Aapke portfolio mein shortfall aaya hai.",
            ),
        ]

    # Compute basic stats
    dur0 = sum(s.duration for s in segments if s.speaker_id == "SPEAKER_00")
    dur1 = sum(s.duration for s in segments if s.speaker_id == "SPEAKER_01")
    stats = [
        SpeakerStats(
            speaker_id="SPEAKER_00",
            role="UNKNOWN",
            total_segments=sum(1 for s in segments if s.speaker_id == "SPEAKER_00"),
            total_words=sum(len(s.text.split()) for s in segments if s.speaker_id == "SPEAKER_00"),
            total_talk_time_seconds=round(dur0, 3),
            talk_time_percentage=round(dur0 / duration * 100, 2) if duration > 0 else 0.0,
            first_segment_start=0.5,
            last_segment_end=15.0,
        ),
        SpeakerStats(
            speaker_id="SPEAKER_01",
            role="UNKNOWN",
            total_segments=sum(1 for s in segments if s.speaker_id == "SPEAKER_01"),
            total_words=sum(len(s.text.split()) for s in segments if s.speaker_id == "SPEAKER_01"),
            total_talk_time_seconds=round(dur1, 3),
            talk_time_percentage=round(dur1 / duration * 100, 2) if duration > 0 else 0.0,
            first_segment_start=5.2,
            last_segment_end=8.0,
        ),
    ]

    return DiarizedTranscript(
        recording_id=recording_id,
        source_file=f"{recording_id}.wav",
        duration=duration,
        num_speakers_detected=2,
        model="sarvam-saaras-v4:saaras:v4",
        mode="codemix",
        processing_time_seconds=1.2,
        segments=segments,
        speaker_stats=stats,
        full_transcript=" ".join(s.text for s in segments),
        status="SUCCESS",
    )


# ======================== Quality Validator Tests ========================

class TestIntervalMerging:
    def test_empty_intervals(self):
        assert merge_intervals([]) == []

    def test_non_overlapping_intervals(self):
        inv = [(0.0, 2.0), (3.0, 5.0), (6.0, 8.0)]
        assert merge_intervals(inv) == [(0.0, 2.0), (3.0, 5.0), (6.0, 8.0)]

    def test_overlapping_intervals(self):
        inv = [(0.0, 3.0), (2.0, 5.0), (4.5, 7.0)]
        assert merge_intervals(inv) == [(0.0, 7.0)]

    def test_subset_intervals(self):
        inv = [(1.0, 10.0), (3.0, 5.0)]
        assert merge_intervals(inv) == [(1.0, 10.0)]


class TestTimestampValidation:
    def test_valid_timestamps_pass(self):
        dt = _create_test_transcript(duration=20.0)
        val = DiarizationQualityValidator()
        report = val.validate_recording(dt)
        assert report.timestamp_validation.is_valid is True
        assert report.timestamp_validation.duration_within_bounds is True
        assert report.timestamp_validation.inverted_segments_count == 0

    def test_timestamp_exceeding_duration_flags_review(self):
        segs = [
            DiarizedSegment(segment_id=0, speaker_id="SPEAKER_00", start=0.0, end=25.0, text="Speaking past end"),
        ]
        dt = _create_test_transcript(duration=10.0, segments=segs)
        val = DiarizationQualityValidator()
        report = val.validate_recording(dt)
        assert report.timestamp_validation.is_valid is False
        assert report.timestamp_validation.exceeded_duration_count == 1
        assert report.manual_review_required is True


class TestOverlapDetection:
    def test_no_overlaps(self):
        dt = _create_test_transcript()
        val = DiarizationQualityValidator()
        report = val.validate_recording(dt)
        assert report.overlap_metrics.overlap_events_count == 0
        assert report.overlap_metrics.total_simultaneous_speech_seconds == 0.0

    def test_cross_talk_overlap(self):
        segs = [
            DiarizedSegment(segment_id=0, speaker_id="SPEAKER_00", start=1.0, end=6.0, text="Explaining details"),
            DiarizedSegment(segment_id=1, speaker_id="SPEAKER_01", start=4.0, end=7.0, text="Interjecting question"),
        ]
        dt = _create_test_transcript(duration=10.0, segments=segs)
        val = DiarizationQualityValidator()
        report = val.validate_recording(dt)
        assert report.overlap_metrics.overlap_events_count == 1
        # Overlap interval: [4.0, 6.0] -> 2.0s
        assert report.overlap_metrics.total_simultaneous_speech_seconds == 2.0
        assert report.overlap_metrics.cross_talk_pairs_count == 1
        assert report.overlap_metrics.same_speaker_overlaps_count == 0

    def test_same_speaker_overlap_flags_review(self):
        segs = [
            DiarizedSegment(segment_id=0, speaker_id="SPEAKER_00", start=1.0, end=5.0, text="Part one"),
            DiarizedSegment(segment_id=1, speaker_id="SPEAKER_00", start=3.0, end=6.0, text="Part two overlapping"),
        ]
        dt = _create_test_transcript(duration=10.0, segments=segs)
        val = DiarizationQualityValidator()
        report = val.validate_recording(dt)
        assert report.overlap_metrics.same_speaker_overlaps_count == 1
        assert report.manual_review_required is True


class TestShortSegments:
    def test_short_segment_identification(self):
        segs = [
            DiarizedSegment(segment_id=0, speaker_id="SPEAKER_00", start=0.0, end=1.0, text="Hello"),
            DiarizedSegment(segment_id=1, speaker_id="SPEAKER_01", start=1.2, end=1.45, text="Haan"),  # 0.25s
        ]
        dt = _create_test_transcript(duration=5.0, segments=segs)
        val = DiarizationQualityValidator()
        report = val.validate_recording(dt)
        assert report.short_segments.count == 1
        assert report.short_segments.samples[0]["text"] == "Haan"
        assert report.short_segments.empty_segments_count == 0


class TestSpeakerConsistency:
    def test_balanced_alternation(self):
        dt = _create_test_transcript(duration=30.0)
        val = DiarizationQualityValidator()
        report = val.validate_recording(dt)
        assert report.speaker_consistency.speaker_turns_count == 2
        assert report.speaker_consistency.turn_alternation_ratio == 1.0

    def test_disparity_flag_triggers_manual_review(self):
        # Speaker 01 talks for 0.5s out of 100s (< 15%)
        segs = [
            DiarizedSegment(segment_id=0, speaker_id="SPEAKER_00", start=0.0, end=80.0, text="Long monologue"),
            DiarizedSegment(segment_id=1, speaker_id="SPEAKER_01", start=80.5, end=81.0, text="Yes"),
        ]
        dt = _create_test_transcript(duration=100.0, segments=segs)
        val = DiarizationQualityValidator()
        report = val.validate_recording(dt)
        assert report.speaker_consistency.talk_time_disparity_flag is True
        assert report.manual_review_required is True
        assert any("disparity" in r.lower() for r in report.manual_review_reasons)


class TestTextPreservation:
    def test_comparison_with_primary_transcript(self, tmp_path):
        prim_data = {
            "recording_id": "test_rec_01",
            "segments": [
                {"segment_id": 0, "text": "Namaste HDFC Securities se bol raha hoon."},
                {"segment_id": 1, "text": "Haan boliye sir."},
            ],
        }
        prim_path = tmp_path / "test_rec_01.json"
        prim_path.write_text(json.dumps(prim_data), encoding="utf-8")

        dt = _create_test_transcript()
        val = DiarizationQualityValidator()
        report = val.validate_recording(dt, primary_transcript_path=prim_path)

        assert report.text_preservation.primary_words == 10
        assert report.text_preservation.diarized_words == 16
        assert report.text_preservation.status == "ENRICHED"
        assert report.text_preservation.word_delta == 6


# ======================== Role Mapper Tests ========================

class TestRoleMappingManager:
    def test_initialization_defaults_to_unknown(self, tmp_path):
        diar_dir = tmp_path / "diar"
        diar_dir.mkdir()
        storage = tmp_path / "speaker_role_mappings.json"

        # Write sample diarized transcript
        dt = _create_test_transcript()
        dt.save_json(diar_dir / "test_rec_01.json")

        mgr = RoleMappingManager(storage_path=storage)
        reg = mgr.initialize_from_transcripts(diar_dir)

        assert reg.total_recordings == 1
        assert reg.pending_reviews == 1
        assert reg.completed_reviews == 0

        mapping = reg.mappings["test_rec_01"]
        assert mapping.roles["SPEAKER_00"] == "UNKNOWN"
        assert mapping.roles["SPEAKER_01"] == "UNKNOWN"
        assert mapping.review_status == "PENDING"

    def test_assign_valid_role(self, tmp_path):
        diar_dir = tmp_path / "diar"
        diar_dir.mkdir()
        storage = tmp_path / "speaker_role_mappings.json"

        dt = _create_test_transcript()
        dt.save_json(diar_dir / "test_rec_01.json")

        mgr = RoleMappingManager(storage_path=storage)
        mgr.initialize_from_transcripts(diar_dir)

        # Assign RM to SPEAKER_00
        m = mgr.assign_role(
            recording_id="test_rec_01",
            speaker_id="SPEAKER_00",
            role="RM",
            auditor_id="Auditor_Gaurav",
            notes="Identified self as Saurabh",
        )
        assert m.roles["SPEAKER_00"] == "RM"
        assert m.roles["SPEAKER_01"] == "UNKNOWN"
        assert m.review_status == "IN_REVIEW"

        # Assign CUSTOMER to SPEAKER_01 -> Completed
        m = mgr.assign_role(
            recording_id="test_rec_01",
            speaker_id="SPEAKER_01",
            role="CUSTOMER",
        )
        assert m.roles["SPEAKER_01"] == "CUSTOMER"
        assert m.review_status == "COMPLETED"

    def test_invalid_role_raises_value_error(self, tmp_path):
        storage = tmp_path / "speaker_role_mappings.json"
        mgr = RoleMappingManager(storage_path=storage)
        mgr.registry.mappings["test_rec_01"] = RecordingRoleMapping(
            recording_id="test_rec_01",
            source_file="test_rec_01.wav",
        )

        with pytest.raises(ValueError, match="Invalid role"):
            mgr.assign_role("test_rec_01", "SPEAKER_00", "INVALID_ROLE")

    def test_apply_mapping_to_transcript(self, tmp_path):
        storage = tmp_path / "speaker_role_mappings.json"
        mgr = RoleMappingManager(storage_path=storage)
        mgr.registry.mappings["test_rec_01"] = RecordingRoleMapping(
            recording_id="test_rec_01",
            source_file="test_rec_01.wav",
            roles={"SPEAKER_00": "RM", "SPEAKER_01": "CUSTOMER"},
            review_status="COMPLETED",
        )

        dt = _create_test_transcript()
        updated = mgr.apply_to_transcript(dt)

        assert updated.speaker_stats[0].speaker_id == "SPEAKER_00"
        assert updated.speaker_stats[0].role == "RM"
        assert updated.speaker_stats[1].speaker_id == "SPEAKER_01"
        assert updated.speaker_stats[1].role == "CUSTOMER"


class TestBatchQualityValidation:
    def test_validate_batch_e2e(self, tmp_path):
        diar_dir = tmp_path / "diar"
        diar_dir.mkdir()
        out_report = tmp_path / "diarization_quality_report.json"

        dt1 = _create_test_transcript(recording_id="rec_01")
        dt2 = _create_test_transcript(recording_id="rec_02")
        dt1.save_json(diar_dir / "rec_01.json")
        dt2.save_json(diar_dir / "rec_02.json")

        val = DiarizationQualityValidator()
        report = val.validate_batch(
            diarized_dir=diar_dir,
            output_report_path=out_report,
        )

        assert report.summary["total_recordings_validated"] == 2
        assert report.summary["timestamp_validation_all_passed"] is True
        assert out_report.exists()
