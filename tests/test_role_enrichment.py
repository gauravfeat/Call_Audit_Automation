"""
Comprehensive tests for M4 Role Enrichment Pipeline.

Validates:
1. Enrichment with COMPLETED review status (roles applied correctly).
2. Enrichment with PENDING review status (UNKNOWN fallback).
3. Enrichment with MISSING mapping (UNKNOWN fallback).
4. Enrichment with IN_REVIEW status (partial, still UNKNOWN).
5. Original diarized files are NEVER modified.
6. Role-labeled TXT output format.
7. Role-enriched JSON output contains correct metadata.
8. Batch enrichment (enrich_all).
9. Single recording enrichment (enrich_recording).
10. Auto-enrichment trigger on assignment.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.diarization.diarization_schema import (
    DiarizedSegment,
    DiarizedTranscript,
    SpeakerStats,
)
from app.diarization.role_mapper import (
    RoleMappingManager,
    RecordingRoleMapping,
    SpeakerRoleRegistry,
)
from app.diarization.role_enricher import RoleEnricher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_sample_transcript(recording_id: str = "test_recording_001") -> DiarizedTranscript:
    """Create a minimal valid DiarizedTranscript for testing."""
    return DiarizedTranscript(
        recording_id=recording_id,
        source_file=f"{recording_id}.wav",
        duration=120.0,
        num_speakers_detected=2,
        model="sarvam-saaras-v4:saaras:v4",
        mode="codemix",
        processing_time_seconds=8.5,
        segments=[
            DiarizedSegment(
                segment_id=0,
                speaker_id="SPEAKER_00",
                start=0.0,
                end=5.5,
                text="Hello sir, main aapka relationship manager bol raha hoon.",
            ),
            DiarizedSegment(
                segment_id=1,
                speaker_id="SPEAKER_01",
                start=5.6,
                end=10.2,
                text="Haan ji boliye, kya baat hai?",
            ),
            DiarizedSegment(
                segment_id=2,
                speaker_id="SPEAKER_00",
                start=10.3,
                end=18.0,
                text="Sir aapka mutual fund investment ke baare mein baat karni thi.",
            ),
            DiarizedSegment(
                segment_id=3,
                speaker_id="SPEAKER_01",
                start=18.5,
                end=22.0,
                text="Achha theek hai, bataiye.",
            ),
        ],
        speaker_stats=[
            SpeakerStats(
                speaker_id="SPEAKER_00",
                total_segments=2,
                total_words=16,
                total_talk_time_seconds=13.2,
                talk_time_percentage=55.0,
                first_segment_start=0.0,
                last_segment_end=18.0,
            ),
            SpeakerStats(
                speaker_id="SPEAKER_01",
                total_segments=2,
                total_words=10,
                total_talk_time_seconds=8.1,
                talk_time_percentage=33.75,
                first_segment_start=5.6,
                last_segment_end=22.0,
            ),
        ],
        full_transcript=(
            "Hello sir, main aapka relationship manager bol raha hoon. "
            "Haan ji boliye, kya baat hai? "
            "Sir aapka mutual fund investment ke baare mein baat karni thi. "
            "Achha theek hai, bataiye."
        ),
    )


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace with diarized transcripts and role mappings."""
    diarized_dir = tmp_path / "diarized_transcripts"
    diarized_dir.mkdir()
    enriched_dir = diarized_dir / "role_enriched"
    enriched_dir.mkdir()

    # Save a sample transcript
    transcript = _make_sample_transcript("test_recording_001")
    transcript.save_json(diarized_dir / "test_recording_001.json")

    # Save a second transcript
    transcript2 = _make_sample_transcript("test_recording_002")
    transcript2.save_json(diarized_dir / "test_recording_002.json")

    return tmp_path, diarized_dir, enriched_dir


@pytest.fixture
def completed_registry(tmp_workspace):
    """Registry with COMPLETED review for test_recording_001."""
    tmp_path, diarized_dir, enriched_dir = tmp_workspace
    registry_path = diarized_dir / "speaker_role_mappings.json"

    registry = SpeakerRoleRegistry(
        total_recordings=2,
        pending_reviews=1,
        completed_reviews=1,
        mappings={
            "test_recording_001": RecordingRoleMapping(
                recording_id="test_recording_001",
                source_file="test_recording_001.wav",
                roles={"SPEAKER_00": "RM", "SPEAKER_01": "CUSTOMER"},
                review_status="COMPLETED",
                auditor_id="auditor_gaurav",
                notes="SPEAKER_00 introduced self as RM Saurabh",
            ),
            "test_recording_002": RecordingRoleMapping(
                recording_id="test_recording_002",
                source_file="test_recording_002.wav",
                roles={"SPEAKER_00": "UNKNOWN", "SPEAKER_01": "UNKNOWN"},
                review_status="PENDING",
            ),
        },
    )
    registry.save_json(registry_path)
    return tmp_path, diarized_dir, enriched_dir, registry_path


@pytest.fixture
def pending_registry(tmp_workspace):
    """Registry with only PENDING reviews."""
    tmp_path, diarized_dir, enriched_dir = tmp_workspace
    registry_path = diarized_dir / "speaker_role_mappings.json"

    registry = SpeakerRoleRegistry(
        total_recordings=2,
        pending_reviews=2,
        completed_reviews=0,
        mappings={
            "test_recording_001": RecordingRoleMapping(
                recording_id="test_recording_001",
                source_file="test_recording_001.wav",
                roles={"SPEAKER_00": "UNKNOWN", "SPEAKER_01": "UNKNOWN"},
                review_status="PENDING",
            ),
            "test_recording_002": RecordingRoleMapping(
                recording_id="test_recording_002",
                source_file="test_recording_002.wav",
                roles={"SPEAKER_00": "UNKNOWN", "SPEAKER_01": "UNKNOWN"},
                review_status="PENDING",
            ),
        },
    )
    registry.save_json(registry_path)
    return tmp_path, diarized_dir, enriched_dir, registry_path


# ---------------------------------------------------------------------------
# Test 1: Enrichment with COMPLETED review
# ---------------------------------------------------------------------------

class TestCompletedReviewEnrichment:
    """Roles should be correctly applied when review_status is COMPLETED."""

    def test_segments_have_correct_roles(self, completed_registry):
        _, diarized_dir, enriched_dir, registry_path = completed_registry
        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(
            role_manager=role_mgr,
            output_dir=enriched_dir,
        )

        transcript = DiarizedTranscript.from_json(diarized_dir / "test_recording_001.json")
        enriched = enricher.enrich_transcript(transcript)

        # SPEAKER_00 segments should have role=RM
        for seg in enriched.segments:
            if seg.speaker_id == "SPEAKER_00":
                assert seg.speaker_role == "RM", f"Segment {seg.segment_id}: expected RM"
            elif seg.speaker_id == "SPEAKER_01":
                assert seg.speaker_role == "CUSTOMER", f"Segment {seg.segment_id}: expected CUSTOMER"

    def test_speaker_stats_have_correct_roles(self, completed_registry):
        _, diarized_dir, enriched_dir, registry_path = completed_registry
        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(role_manager=role_mgr, output_dir=enriched_dir)

        transcript = DiarizedTranscript.from_json(diarized_dir / "test_recording_001.json")
        enriched = enricher.enrich_transcript(transcript)

        roles_found = {ss.speaker_id: ss.role for ss in enriched.speaker_stats}
        assert roles_found["SPEAKER_00"] == "RM"
        assert roles_found["SPEAKER_01"] == "CUSTOMER"

    def test_enrichment_metadata_attached(self, completed_registry):
        _, diarized_dir, enriched_dir, registry_path = completed_registry
        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(role_manager=role_mgr, output_dir=enriched_dir)

        transcript = DiarizedTranscript.from_json(diarized_dir / "test_recording_001.json")
        enriched = enricher.enrich_transcript(transcript)

        meta = enriched.provider_metadata
        assert meta is not None
        assert "role_enrichment" in meta
        assert meta["role_enrichment"]["review_status"] == "COMPLETED"
        assert meta["role_enrichment"]["roles_applied"]["SPEAKER_00"] == "RM"


# ---------------------------------------------------------------------------
# Test 2: Enrichment with PENDING review (UNKNOWN fallback)
# ---------------------------------------------------------------------------

class TestPendingReviewEnrichment:
    """All roles should remain UNKNOWN when review_status is PENDING."""

    def test_all_segments_unknown(self, pending_registry):
        _, diarized_dir, enriched_dir, registry_path = pending_registry
        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(role_manager=role_mgr, output_dir=enriched_dir)

        transcript = DiarizedTranscript.from_json(diarized_dir / "test_recording_001.json")
        enriched = enricher.enrich_transcript(transcript)

        for seg in enriched.segments:
            assert seg.speaker_role == "UNKNOWN", (
                f"Segment {seg.segment_id} should be UNKNOWN for PENDING review"
            )

    def test_speaker_stats_unknown(self, pending_registry):
        _, diarized_dir, enriched_dir, registry_path = pending_registry
        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(role_manager=role_mgr, output_dir=enriched_dir)

        transcript = DiarizedTranscript.from_json(diarized_dir / "test_recording_001.json")
        enriched = enricher.enrich_transcript(transcript)

        for ss in enriched.speaker_stats:
            assert ss.role == "UNKNOWN"

    def test_metadata_shows_pending(self, pending_registry):
        _, diarized_dir, enriched_dir, registry_path = pending_registry
        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(role_manager=role_mgr, output_dir=enriched_dir)

        transcript = DiarizedTranscript.from_json(diarized_dir / "test_recording_001.json")
        enriched = enricher.enrich_transcript(transcript)

        meta = enriched.provider_metadata["role_enrichment"]
        assert meta["review_status"] == "PENDING"
        assert meta["roles_applied"] == "NONE_PENDING_REVIEW"


# ---------------------------------------------------------------------------
# Test 3: Enrichment with MISSING mapping
# ---------------------------------------------------------------------------

class TestMissingMappingEnrichment:
    """When no mapping exists for a recording, all roles should be UNKNOWN."""

    def test_unknown_for_unmapped_recording(self, tmp_workspace):
        tmp_path, diarized_dir, enriched_dir = tmp_workspace
        # Empty registry
        registry_path = diarized_dir / "speaker_role_mappings.json"
        SpeakerRoleRegistry().save_json(registry_path)

        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(role_manager=role_mgr, output_dir=enriched_dir)

        transcript = DiarizedTranscript.from_json(diarized_dir / "test_recording_001.json")
        enriched = enricher.enrich_transcript(transcript)

        for seg in enriched.segments:
            assert seg.speaker_role == "UNKNOWN"

        meta = enriched.provider_metadata["role_enrichment"]
        assert meta["review_status"] == "MISSING_MAPPING"


# ---------------------------------------------------------------------------
# Test 4: Enrichment with IN_REVIEW status
# ---------------------------------------------------------------------------

class TestInReviewEnrichment:
    """Partial review (IN_REVIEW) should still default all speakers to UNKNOWN."""

    def test_in_review_defaults_to_unknown(self, tmp_workspace):
        tmp_path, diarized_dir, enriched_dir = tmp_workspace
        registry_path = diarized_dir / "speaker_role_mappings.json"

        registry = SpeakerRoleRegistry(
            total_recordings=1,
            pending_reviews=1,
            completed_reviews=0,
            mappings={
                "test_recording_001": RecordingRoleMapping(
                    recording_id="test_recording_001",
                    source_file="test_recording_001.wav",
                    roles={"SPEAKER_00": "RM", "SPEAKER_01": "UNKNOWN"},
                    review_status="IN_REVIEW",
                    auditor_id="auditor_test",
                ),
            },
        )
        registry.save_json(registry_path)

        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(role_manager=role_mgr, output_dir=enriched_dir)

        transcript = DiarizedTranscript.from_json(diarized_dir / "test_recording_001.json")
        enriched = enricher.enrich_transcript(transcript)

        # Even though SPEAKER_00 was assigned RM in the mapping,
        # IN_REVIEW status means all roles default to UNKNOWN
        for seg in enriched.segments:
            assert seg.speaker_role == "UNKNOWN", (
                f"Seg {seg.segment_id} ({seg.speaker_id}) should be UNKNOWN for IN_REVIEW"
            )


# ---------------------------------------------------------------------------
# Test 5: Original files are NEVER modified
# ---------------------------------------------------------------------------

class TestOriginalFileImmutability:
    """Original diarized JSON and TXT files must never be modified."""

    def test_original_json_unchanged_after_enrichment(self, completed_registry):
        _, diarized_dir, enriched_dir, registry_path = completed_registry
        original_path = diarized_dir / "test_recording_001.json"

        # Snapshot original content
        with open(original_path, "r", encoding="utf-8") as f:
            original_content = f.read()
        original_mtime = original_path.stat().st_mtime

        # Run enrichment
        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(role_manager=role_mgr, output_dir=enriched_dir)
        enricher.enrich_recording("test_recording_001", diarized_dir=diarized_dir)

        # Verify original unchanged
        with open(original_path, "r", encoding="utf-8") as f:
            after_content = f.read()

        assert original_content == after_content, "Original JSON file was modified!"
        assert original_path.stat().st_mtime == original_mtime, "Original file timestamp changed!"

    def test_enriched_files_in_separate_directory(self, completed_registry):
        _, diarized_dir, enriched_dir, registry_path = completed_registry

        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(role_manager=role_mgr, output_dir=enriched_dir)
        result = enricher.enrich_recording("test_recording_001", diarized_dir=diarized_dir)

        assert result is not None
        json_path, txt_path = result

        # Enriched files should be in the role_enriched subdirectory
        assert enriched_dir in json_path.parents or json_path.parent == enriched_dir
        assert enriched_dir in txt_path.parents or txt_path.parent == enriched_dir

        # They should NOT be in the parent diarized_dir directly
        assert json_path.parent != diarized_dir
        assert txt_path.parent != diarized_dir


# ---------------------------------------------------------------------------
# Test 6: Role-labeled TXT output format
# ---------------------------------------------------------------------------

class TestRoleLabeledTextOutput:
    """Validate the human-readable TXT output format."""

    def test_txt_contains_role_labels(self, completed_registry):
        _, diarized_dir, enriched_dir, registry_path = completed_registry

        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(role_manager=role_mgr, output_dir=enriched_dir)

        transcript = DiarizedTranscript.from_json(diarized_dir / "test_recording_001.json")
        enriched = enricher.enrich_transcript(transcript)
        txt_content = enricher.generate_role_labeled_text(enriched)

        assert "SPEAKER_00 (RM):" in txt_content
        assert "SPEAKER_01 (CUSTOMER):" in txt_content
        assert "ROLE-ENRICHED DIARIZED TRANSCRIPT" in txt_content
        assert "SPEAKER STATISTICS:" in txt_content
        assert "ROLE-LABELLED CONVERSATION:" in txt_content

    def test_txt_contains_timestamps(self, completed_registry):
        _, diarized_dir, enriched_dir, registry_path = completed_registry

        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(role_manager=role_mgr, output_dir=enriched_dir)

        transcript = DiarizedTranscript.from_json(diarized_dir / "test_recording_001.json")
        enriched = enricher.enrich_transcript(transcript)
        txt_content = enricher.generate_role_labeled_text(enriched)

        # Should contain timestamp format [mm:ss -> mm:ss]
        assert "[00:00.0 ->" in txt_content


# ---------------------------------------------------------------------------
# Test 7: Role-enriched JSON output validation
# ---------------------------------------------------------------------------

class TestEnrichedJsonOutput:
    """Validate JSON output structure and content."""

    def test_json_output_is_valid(self, completed_registry):
        _, diarized_dir, enriched_dir, registry_path = completed_registry

        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(role_manager=role_mgr, output_dir=enriched_dir)
        result = enricher.enrich_recording("test_recording_001", diarized_dir=diarized_dir)

        assert result is not None
        json_path, _ = result

        # Should be valid JSON and loadable as DiarizedTranscript
        loaded = DiarizedTranscript.from_json(json_path)
        assert loaded.recording_id == "test_recording_001"
        assert len(loaded.segments) == 4

    def test_json_preserves_transcript_text(self, completed_registry):
        _, diarized_dir, enriched_dir, registry_path = completed_registry

        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(role_manager=role_mgr, output_dir=enriched_dir)
        result = enricher.enrich_recording("test_recording_001", diarized_dir=diarized_dir)

        json_path, _ = result
        loaded = DiarizedTranscript.from_json(json_path)

        original = DiarizedTranscript.from_json(diarized_dir / "test_recording_001.json")

        for orig_seg, enr_seg in zip(original.segments, loaded.segments):
            assert orig_seg.text == enr_seg.text, (
                f"Text mismatch at segment {orig_seg.segment_id}"
            )
            assert orig_seg.start == enr_seg.start
            assert orig_seg.end == enr_seg.end


# ---------------------------------------------------------------------------
# Test 8: Batch enrichment (enrich_all)
# ---------------------------------------------------------------------------

class TestBatchEnrichment:
    """Validate enrich_all processes all recordings correctly."""

    def test_enrich_all_processes_both_recordings(self, completed_registry):
        _, diarized_dir, enriched_dir, registry_path = completed_registry

        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(role_manager=role_mgr, output_dir=enriched_dir)

        summary = enricher.enrich_all(diarized_dir=diarized_dir, output_dir=enriched_dir)

        assert summary["total_recordings"] == 2
        assert summary["successfully_enriched"] == 2
        assert len(summary["results"]) == 2

    def test_enrich_all_creates_summary_json(self, completed_registry):
        _, diarized_dir, enriched_dir, registry_path = completed_registry

        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(role_manager=role_mgr, output_dir=enriched_dir)

        enricher.enrich_all(diarized_dir=diarized_dir, output_dir=enriched_dir)

        summary_path = enriched_dir / "role_enrichment_summary.json"
        assert summary_path.exists()

        with open(summary_path, "r") as f:
            data = json.load(f)
        assert data["total_recordings"] == 2

    def test_enrich_all_generates_output_files(self, completed_registry):
        _, diarized_dir, enriched_dir, registry_path = completed_registry

        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(role_manager=role_mgr, output_dir=enriched_dir)

        enricher.enrich_all(diarized_dir=diarized_dir, output_dir=enriched_dir)

        # Both recordings should have JSON and TXT outputs
        assert (enriched_dir / "test_recording_001.json").exists()
        assert (enriched_dir / "test_recording_001.txt").exists()
        assert (enriched_dir / "test_recording_002.json").exists()
        assert (enriched_dir / "test_recording_002.txt").exists()


# ---------------------------------------------------------------------------
# Test 9: Single recording enrichment
# ---------------------------------------------------------------------------

class TestSingleRecordingEnrichment:
    """Validate enrich_recording works for a specific recording."""

    def test_enrich_single_recording(self, completed_registry):
        _, diarized_dir, enriched_dir, registry_path = completed_registry

        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(role_manager=role_mgr, output_dir=enriched_dir)

        result = enricher.enrich_recording(
            "test_recording_001",
            diarized_dir=diarized_dir,
            output_dir=enriched_dir,
        )
        assert result is not None

        json_path, txt_path = result
        assert json_path.exists()
        assert txt_path.exists()

    def test_enrich_nonexistent_recording_returns_none(self, completed_registry):
        _, diarized_dir, enriched_dir, registry_path = completed_registry

        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(role_manager=role_mgr, output_dir=enriched_dir)

        result = enricher.enrich_recording(
            "nonexistent_recording",
            diarized_dir=diarized_dir,
            output_dir=enriched_dir,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Test 10: Never assume SPEAKER_00 is RM
# ---------------------------------------------------------------------------

class TestNoRoleAssumption:
    """SPEAKER_00 should NOT automatically map to RM."""

    def test_speaker_00_can_be_customer(self, tmp_workspace):
        """Verify SPEAKER_00 as CUSTOMER works when mapped that way."""
        tmp_path, diarized_dir, enriched_dir = tmp_workspace
        registry_path = diarized_dir / "speaker_role_mappings.json"

        # Reversed mapping: SPEAKER_00 is CUSTOMER, SPEAKER_01 is RM
        registry = SpeakerRoleRegistry(
            total_recordings=1,
            completed_reviews=1,
            pending_reviews=0,
            mappings={
                "test_recording_001": RecordingRoleMapping(
                    recording_id="test_recording_001",
                    source_file="test_recording_001.wav",
                    roles={"SPEAKER_00": "CUSTOMER", "SPEAKER_01": "RM"},
                    review_status="COMPLETED",
                    auditor_id="auditor_test",
                ),
            },
        )
        registry.save_json(registry_path)

        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(role_manager=role_mgr, output_dir=enriched_dir)

        transcript = DiarizedTranscript.from_json(diarized_dir / "test_recording_001.json")
        enriched = enricher.enrich_transcript(transcript)

        for seg in enriched.segments:
            if seg.speaker_id == "SPEAKER_00":
                assert seg.speaker_role == "CUSTOMER"
            elif seg.speaker_id == "SPEAKER_01":
                assert seg.speaker_role == "RM"

# ---------------------------------------------------------------------------
# Test 11: Edge cases - Unknown speaker ID, missing speaker role, idempotency
# ---------------------------------------------------------------------------

class TestEdgeCasesAndIdempotency:
    """Validate robustness against unknown speaker IDs, missing speaker roles, and re-running."""

    def test_unknown_speaker_id_in_segment_defaults_to_unknown(self, tmp_workspace):
        """A segment with an unexpected speaker ID (e.g. SPEAKER_99) should receive UNKNOWN role."""
        tmp_path, diarized_dir, enriched_dir = tmp_workspace
        registry_path = diarized_dir / "speaker_role_mappings.json"

        registry = SpeakerRoleRegistry(
            total_recordings=1,
            completed_reviews=1,
            pending_reviews=0,
            mappings={
                "test_recording_001": RecordingRoleMapping(
                    recording_id="test_recording_001",
                    source_file="test_recording_001.wav",
                    roles={"SPEAKER_00": "RM", "SPEAKER_01": "CUSTOMER"},
                    review_status="COMPLETED",
                    auditor_id="test_auditor",
                ),
            },
        )
        registry.save_json(registry_path)

        # Create transcript with an unexpected third speaker
        transcript = _make_sample_transcript("test_recording_001")
        transcript.segments.append(
            DiarizedSegment(
                segment_id=4,
                speaker_id="SPEAKER_99",
                start=25.0,
                end=28.0,
                text="Unexpected background speaker.",
            )
        )

        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(role_manager=role_mgr, output_dir=enriched_dir)
        enriched = enricher.enrich_transcript(transcript)

        assert enriched.segments[-1].speaker_id == "SPEAKER_99"
        assert enriched.segments[-1].speaker_role == "UNKNOWN"

    def test_missing_speaker_in_mapping_defaults_to_unknown(self, tmp_workspace):
        """If a speaker is omitted from mapping roles, their segments should be UNKNOWN."""
        tmp_path, diarized_dir, enriched_dir = tmp_workspace
        registry_path = diarized_dir / "speaker_role_mappings.json"

        # Only SPEAKER_00 is mapped, SPEAKER_01 is missing from mapping
        registry = SpeakerRoleRegistry(
            total_recordings=1,
            completed_reviews=1,
            pending_reviews=0,
            mappings={
                "test_recording_001": RecordingRoleMapping(
                    recording_id="test_recording_001",
                    source_file="test_recording_001.wav",
                    roles={"SPEAKER_00": "RM"},  # SPEAKER_01 missing
                    review_status="COMPLETED",
                    auditor_id="test_auditor",
                ),
            },
        )
        registry.save_json(registry_path)

        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(role_manager=role_mgr, output_dir=enriched_dir)

        transcript = _make_sample_transcript("test_recording_001")
        enriched = enricher.enrich_transcript(transcript)

        for seg in enriched.segments:
            if seg.speaker_id == "SPEAKER_00":
                assert seg.speaker_role == "RM"
            elif seg.speaker_id == "SPEAKER_01":
                assert seg.speaker_role == "UNKNOWN"

    def test_rerun_enrichment_is_idempotent(self, completed_registry):
        """Re-running enrichment must produce identical outputs without duplication or corruption."""
        _, diarized_dir, enriched_dir, registry_path = completed_registry

        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(role_manager=role_mgr, output_dir=enriched_dir)

        # First run
        result1 = enricher.enrich_recording("test_recording_001", diarized_dir=diarized_dir)
        assert result1 is not None
        json_path1, txt_path1 = result1
        with open(json_path1, "r", encoding="utf-8") as f:
            json_content1 = json.load(f)
        with open(txt_path1, "r", encoding="utf-8") as f:
            txt_content1 = f.read()

        # Second run (re-enriching)
        result2 = enricher.enrich_recording("test_recording_001", diarized_dir=diarized_dir)
        assert result2 is not None
        json_path2, txt_path2 = result2
        with open(json_path2, "r", encoding="utf-8") as f:
            json_content2 = json.load(f)
        with open(txt_path2, "r", encoding="utf-8") as f:
            txt_content2 = f.read()

        # Segments count must not double
        assert len(json_content1["segments"]) == len(json_content2["segments"]) == 4
        # Content must match exactly (ignoring updated applied_at timestamp)
        assert json_content1["recording_id"] == json_content2["recording_id"]
        assert [s["speaker_role"] for s in json_content1["segments"]] == [s["speaker_role"] for s in json_content2["segments"]]
        assert txt_content1 == txt_content2

    def test_different_speaker_mappings_across_recordings(self, tmp_workspace):
        """Rec 1 has SPEAKER_00=RM, Rec 2 has SPEAKER_00=CUSTOMER. Both apply correctly."""
        tmp_path, diarized_dir, enriched_dir = tmp_workspace
        registry_path = diarized_dir / "speaker_role_mappings.json"

        registry = SpeakerRoleRegistry(
            total_recordings=2,
            completed_reviews=2,
            pending_reviews=0,
            mappings={
                "test_recording_001": RecordingRoleMapping(
                    recording_id="test_recording_001",
                    source_file="test_recording_001.wav",
                    roles={"SPEAKER_00": "RM", "SPEAKER_01": "CUSTOMER"},
                    review_status="COMPLETED",
                    auditor_id="test_auditor",
                ),
                "test_recording_002": RecordingRoleMapping(
                    recording_id="test_recording_002",
                    source_file="test_recording_002.wav",
                    roles={"SPEAKER_00": "CUSTOMER", "SPEAKER_01": "RM"},  # Swapped!
                    review_status="COMPLETED",
                    auditor_id="test_auditor",
                ),
            },
        )
        registry.save_json(registry_path)

        role_mgr = RoleMappingManager(storage_path=registry_path)
        enricher = RoleEnricher(role_manager=role_mgr, output_dir=enriched_dir)

        # Enrich both
        t1 = DiarizedTranscript.from_json(diarized_dir / "test_recording_001.json")
        t2 = DiarizedTranscript.from_json(diarized_dir / "test_recording_002.json")
        e1 = enricher.enrich_transcript(t1)
        e2 = enricher.enrich_transcript(t2)

        # In rec 1: SPEAKER_00 is RM
        assert e1.segments[0].speaker_id == "SPEAKER_00"
        assert e1.segments[0].speaker_role == "RM"

        # In rec 2: SPEAKER_00 is CUSTOMER
        assert e2.segments[0].speaker_id == "SPEAKER_00"
        assert e2.segments[0].speaker_role == "CUSTOMER"
