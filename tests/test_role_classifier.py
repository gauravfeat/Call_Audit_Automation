"""
Comprehensive Tests for M4.5 Automatic RM vs Customer Role Classification.

Validates:
1. Normal speaker mapping (SPEAKER_00 = RM, SPEAKER_01 = CUSTOMER).
2. Reversed speaker mapping (SPEAKER_00 = CUSTOMER, SPEAKER_01 = RM).
3. Customer speaking first (anti-heuristic test).
4. Ambiguous conversations (NEEDS_MANUAL_REVIEW).
5. Low confidence below threshold (NEEDS_MANUAL_REVIEW).
6. Invalid or conflicting roles (e.g. 2 RMs or missing speaker -> NEEDS_MANUAL_REVIEW).
7. Missing API key and API failure handling (graceful error isolation).
8. Existing COMPLETED manual mapping preservation (manual truth is never overwritten).
9. Auditor approval workflow (transferring AI prediction to registry).
10. Original diarized files are NEVER modified.
11. Batch classification and dry-run mode.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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
from app.diarization.role_classifier import (
    RoleClassifier,
    RoleClassificationResult,
    SpeakerEvidence,
    LLMClassificationPayload,
)


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------

def _make_transcript(
    recording_id: str = "call_test_001",
    reversed_speakers: bool = False,
    customer_speaks_first: bool = True,
) -> DiarizedTranscript:
    """Generate mock DiarizedTranscript for testing."""
    rm_spk = "SPEAKER_01" if reversed_speakers else "SPEAKER_00"
    cust_spk = "SPEAKER_00" if reversed_speakers else "SPEAKER_01"

    if customer_speaks_first:
        segments = [
            DiarizedSegment(
                segment_id=0, speaker_id=cust_spk, start=0.5, end=1.5, text="Hello."
            ),
            DiarizedSegment(
                segment_id=1, speaker_id=rm_spk, start=1.8, end=4.5,
                text="Hi Ma'am, Saurabh bol raha hoon HDFC Securities se."
            ),
            DiarizedSegment(
                segment_id=2, speaker_id=cust_spk, start=4.8, end=6.0, text="Haan Saurabh boliye."
            ),
            DiarizedSegment(
                segment_id=3, speaker_id=rm_spk, start=6.2, end=12.0,
                text="Ma'am aapke portfolio ke regarding service call kiya tha. Tata Motors position ke baare mein."
            ),
        ]
    else:
        segments = [
            DiarizedSegment(
                segment_id=0, speaker_id=rm_spk, start=0.5, end=3.5,
                text="Hello Monika Ma'am, Saurabh from HDFC Securities."
            ),
            DiarizedSegment(
                segment_id=1, speaker_id=cust_spk, start=3.8, end=5.0, text="Yes Saurabh."
            ),
        ]

    return DiarizedTranscript(
        recording_id=recording_id,
        source_file=f"{recording_id}.wav",
        duration=60.0,
        num_speakers_detected=2,
        model="sarvam-saaras-v4:saaras:v4",
        mode="codemix",
        processing_time_seconds=2.5,
        segments=segments,
        speaker_stats=[
            SpeakerStats(
                speaker_id="SPEAKER_00", total_segments=2, total_words=12,
                total_talk_time_seconds=15.0, talk_time_percentage=25.0,
                first_segment_start=0.5, last_segment_end=6.0,
            ),
            SpeakerStats(
                speaker_id="SPEAKER_01", total_segments=2, total_words=20,
                total_talk_time_seconds=30.0, talk_time_percentage=50.0,
                first_segment_start=1.8, last_segment_end=12.0,
            ),
        ],
        full_transcript=" ".join(s.text for s in segments),
    )


def _mock_gemini_client(payload: LLMClassificationPayload):
    """Create a mock Gemini client returning the payload."""
    mock_resp = MagicMock()
    mock_resp.text = payload.model_dump_json()
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_resp
    return mock_client


@pytest.fixture
def test_env(tmp_path):
    """Set up temporary working folders for diarized transcripts and AI outputs."""
    diar_dir = tmp_path / "diarized_transcripts"
    diar_dir.mkdir()
    ai_dir = diar_dir / "ai_role_classifications"
    ai_dir.mkdir()
    enriched_dir = diar_dir / "role_enriched"
    enriched_dir.mkdir()

    # Save sample diarized JSON
    t1 = _make_transcript("call_test_001", reversed_speakers=False)
    t1.save_json(diar_dir / "call_test_001.json")

    # Save second (reversed) diarized JSON
    t2 = _make_transcript("call_test_002", reversed_speakers=True)
    t2.save_json(diar_dir / "call_test_002.json")

    return tmp_path, diar_dir, ai_dir, enriched_dir


# ---------------------------------------------------------------------------
# Test 1: Normal Speaker Mapping (SPEAKER_00 = RM, SPEAKER_01 = CUSTOMER)
# ---------------------------------------------------------------------------

class TestNormalSpeakerClassification:
    """SPEAKER_00 is RM, SPEAKER_01 is Customer."""

    def test_normal_mapping_auto_classified(self, test_env):
        _, diar_dir, ai_dir, _ = test_env
        transcript = DiarizedTranscript.from_json(diar_dir / "call_test_001.json")

        payload = LLMClassificationPayload(
            speakers=[
                SpeakerEvidence(
                    speaker_id="SPEAKER_00",
                    predicted_role="RM",
                    confidence=0.96,
                    role_indicators=["Identifies as Saurabh from HDFC Securities", "Initiates portfolio review"],
                    supporting_quotes=["Hi Ma'am, Saurabh bol raha hoon HDFC Securities se."],
                ),
                SpeakerEvidence(
                    speaker_id="SPEAKER_01",
                    predicted_role="CUSTOMER",
                    confidence=0.94,
                    role_indicators=["Receives call", "Acknowledges portfolio inquiry"],
                    supporting_quotes=["Haan Saurabh boliye."],
                ),
            ],
            rationale="SPEAKER_00 explicitly introduces themselves as HDFC Securities RM. SPEAKER_01 is the client.",
        )

        mock_client = _mock_gemini_client(payload)
        classifier = RoleClassifier(api_key="mock_key", client=mock_client, output_dir=ai_dir)
        result = classifier.classify_transcript(transcript)

        assert result.status == "AUTO_CLASSIFIED"
        assert result.predicted_roles["SPEAKER_00"] == "RM"
        assert result.predicted_roles["SPEAKER_01"] == "CUSTOMER"
        assert result.overall_confidence == 0.94
        assert len(result.speakers) == 2
        rm_evidence = next(s for s in result.speakers if s.speaker_id == "SPEAKER_00")
        assert "HDFC Securities" in rm_evidence.supporting_quotes[0]


# ---------------------------------------------------------------------------
# Test 2: Reversed Speaker Mapping (SPEAKER_00 = CUSTOMER, SPEAKER_01 = RM)
# ---------------------------------------------------------------------------

class TestReversedSpeakerClassification:
    """SPEAKER_00 is Customer, SPEAKER_01 is RM (like Call 2)."""

    def test_reversed_mapping_never_assumes_speaker_00_is_rm(self, test_env):
        _, diar_dir, ai_dir, _ = test_env
        transcript = DiarizedTranscript.from_json(diar_dir / "call_test_002.json")

        payload = LLMClassificationPayload(
            speakers=[
                SpeakerEvidence(
                    speaker_id="SPEAKER_00",
                    predicted_role="CUSTOMER",
                    confidence=0.95,
                    role_indicators=["Answers call", "Acknowledges call purpose"],
                    supporting_quotes=["Hello."],
                ),
                SpeakerEvidence(
                    speaker_id="SPEAKER_01",
                    predicted_role="RM",
                    confidence=0.97,
                    role_indicators=["Introduces as Saurabh from HDFC Securities", "Mentions Tata Motors portfolio"],
                    supporting_quotes=["Hi Ma'am, Saurabh bol raha hoon HDFC Securities se."],
                ),
            ],
            rationale="SPEAKER_01 self-identifies as HDFC Securities representative. SPEAKER_00 is the customer.",
        )

        mock_client = _mock_gemini_client(payload)
        classifier = RoleClassifier(api_key="mock_key", client=mock_client, output_dir=ai_dir)
        result = classifier.classify_transcript(transcript)

        assert result.status == "AUTO_CLASSIFIED"
        assert result.predicted_roles["SPEAKER_00"] == "CUSTOMER"
        assert result.predicted_roles["SPEAKER_01"] == "RM"
        assert result.overall_confidence == 0.95


# ---------------------------------------------------------------------------
# Test 3: Customer Speaking First
# ---------------------------------------------------------------------------

class TestCustomerSpeakingFirst:
    """Outbound call where Customer answers first."""

    def test_customer_speaking_first_classified_correctly(self, test_env):
        _, diar_dir, ai_dir, _ = test_env
        transcript = _make_transcript("call_cust_first", customer_speaks_first=True)

        payload = LLMClassificationPayload(
            speakers=[
                SpeakerEvidence(
                    speaker_id="SPEAKER_01",
                    predicted_role="CUSTOMER",
                    confidence=0.93,
                    role_indicators=["Answers phone with greeting"],
                    supporting_quotes=["Hello."],
                ),
                SpeakerEvidence(
                    speaker_id="SPEAKER_00",
                    predicted_role="RM",
                    confidence=0.96,
                    role_indicators=["Identifies as RM from HDFC Securities"],
                    supporting_quotes=["Hi Ma'am, Saurabh bol raha hoon HDFC Securities se."],
                ),
            ],
            rationale="Customer answered first, but SPEAKER_00 introduced themselves as the RM.",
        )

        mock_client = _mock_gemini_client(payload)
        classifier = RoleClassifier(api_key="mock_key", client=mock_client, output_dir=ai_dir)
        result = classifier.classify_transcript(transcript)

        assert result.status == "AUTO_CLASSIFIED"
        assert result.predicted_roles["SPEAKER_01"] == "CUSTOMER"
        assert result.predicted_roles["SPEAKER_00"] == "RM"


# ---------------------------------------------------------------------------
# Test 4: Ambiguous Conversations (NEEDS_MANUAL_REVIEW)
# ---------------------------------------------------------------------------

class TestAmbiguousConversations:
    """Insufficient conversation triggers NEEDS_MANUAL_REVIEW."""

    def test_ambiguous_dialogue_requires_manual_review(self, test_env):
        _, diar_dir, ai_dir, _ = test_env
        transcript = _make_transcript("call_ambiguous")

        payload = LLMClassificationPayload(
            speakers=[
                SpeakerEvidence(
                    speaker_id="SPEAKER_00",
                    predicted_role="UNKNOWN",
                    confidence=0.45,
                    role_indicators=["Unclear identity", "Call disconnected"],
                    supporting_quotes=["Hello?"],
                ),
                SpeakerEvidence(
                    speaker_id="SPEAKER_01",
                    predicted_role="UNKNOWN",
                    confidence=0.40,
                    role_indicators=["Unclear identity"],
                    supporting_quotes=["Can you hear me?"],
                ),
            ],
            rationale="Call terminated before either party stated their identity or business purpose.",
        )

        mock_client = _mock_gemini_client(payload)
        classifier = RoleClassifier(api_key="mock_key", client=mock_client, output_dir=ai_dir)
        result = classifier.classify_transcript(transcript)

        assert result.status == "NEEDS_MANUAL_REVIEW"
        assert result.overall_confidence == 0.40
        assert result.predicted_roles["SPEAKER_00"] == "UNKNOWN"
        assert result.predicted_roles["SPEAKER_01"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# Test 5: Low Confidence Below Threshold (NEEDS_MANUAL_REVIEW)
# ---------------------------------------------------------------------------

class TestLowConfidenceClassification:
    """Confidence 0.75 < 0.85 threshold defaults to NEEDS_MANUAL_REVIEW."""

    def test_confidence_below_threshold_flags_review(self, test_env):
        _, diar_dir, ai_dir, _ = test_env
        transcript = _make_transcript("call_low_conf")

        payload = LLMClassificationPayload(
            speakers=[
                SpeakerEvidence(
                    speaker_id="SPEAKER_00",
                    predicted_role="RM",
                    confidence=0.75,  # Below 0.85
                    role_indicators=["Tentatively RM"],
                ),
                SpeakerEvidence(
                    speaker_id="SPEAKER_01",
                    predicted_role="CUSTOMER",
                    confidence=0.72,  # Below 0.85
                    role_indicators=["Tentatively Customer"],
                ),
            ],
            rationale="Audio quality degraded; partial mention of portfolio.",
        )

        mock_client = _mock_gemini_client(payload)
        classifier = RoleClassifier(
            api_key="mock_key", client=mock_client, confidence_threshold=0.85, output_dir=ai_dir
        )
        result = classifier.classify_transcript(transcript)

        assert result.status == "NEEDS_MANUAL_REVIEW"
        assert result.overall_confidence == 0.72


# ---------------------------------------------------------------------------
# Test 6: Invalid or Conflicting Roles (Mutual Exclusivity)
# ---------------------------------------------------------------------------

class TestInvalidOrConflictingRoles:
    """Enforce exactly one RM and one CUSTOMER for two-speaker calls."""

    def test_two_rms_rejected_to_manual_review(self, test_env):
        _, diar_dir, ai_dir, _ = test_env
        transcript = _make_transcript("call_conflicting")

        payload = LLMClassificationPayload(
            speakers=[
                SpeakerEvidence(speaker_id="SPEAKER_00", predicted_role="RM", confidence=0.95),
                SpeakerEvidence(speaker_id="SPEAKER_01", predicted_role="RM", confidence=0.90),  # Both RM!
            ],
            rationale="Conflicting evidence.",
        )

        mock_client = _mock_gemini_client(payload)
        classifier = RoleClassifier(api_key="mock_key", client=mock_client, output_dir=ai_dir)
        result = classifier.classify_transcript(transcript)

        assert result.status == "NEEDS_MANUAL_REVIEW"

    def test_missing_speaker_in_payload_triggers_manual_review(self, test_env):
        _, diar_dir, ai_dir, _ = test_env
        transcript = _make_transcript("call_missing_spk")

        # LLM only returned SPEAKER_00, omitted SPEAKER_01
        payload = LLMClassificationPayload(
            speakers=[
                SpeakerEvidence(speaker_id="SPEAKER_00", predicted_role="RM", confidence=0.95),
            ],
            rationale="Partial classification.",
        )

        mock_client = _mock_gemini_client(payload)
        classifier = RoleClassifier(api_key="mock_key", client=mock_client, output_dir=ai_dir)
        result = classifier.classify_transcript(transcript)

        assert result.status == "NEEDS_MANUAL_REVIEW"
        assert result.predicted_roles["SPEAKER_01"] == "UNKNOWN"
        assert result.overall_confidence == 0.0


# ---------------------------------------------------------------------------
# Test 7: Missing API Key and API Failure Handling
# ---------------------------------------------------------------------------

class TestMissingApiKeyAndApiFailure:
    """Graceful degradation without unhandled exceptions."""

    def test_missing_api_key_returns_error_result(self, test_env):
        _, diar_dir, ai_dir, _ = test_env
        transcript = DiarizedTranscript.from_json(diar_dir / "call_test_001.json")

        classifier = RoleClassifier(api_key="", output_dir=ai_dir)
        result = classifier.classify_transcript(transcript)

        assert result.status == "ERROR"
        assert "GEMINI_API_KEY is missing" in result.error_details
        assert result.overall_confidence == 0.0

    def test_gemini_api_network_exception_handled_gracefully(self, test_env):
        _, diar_dir, ai_dir, _ = test_env
        transcript = DiarizedTranscript.from_json(diar_dir / "call_test_001.json")

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("503 Service Unavailable")

        classifier = RoleClassifier(api_key="mock_key", client=mock_client, output_dir=ai_dir)
        result = classifier.classify_transcript(transcript)

        assert result.status == "ERROR"
        assert "503 Service Unavailable" in result.error_details


# ---------------------------------------------------------------------------
# Test 8: Preserve COMPLETED Manual Mapping (Source of Truth)
# ---------------------------------------------------------------------------

class TestPreserveCompletedManualMapping:
    """Human auditor completed mappings are never overwritten by AI."""

    def test_completed_mapping_protected_against_ai_overwrite(self, test_env):
        _, diar_dir, ai_dir, _ = test_env
        reg_file = diar_dir / "speaker_role_mappings.json"

        # Pre-populate registry with COMPLETED review
        registry = SpeakerRoleRegistry(
            total_recordings=1,
            completed_reviews=1,
            pending_reviews=0,
            mappings={
                "call_test_001": RecordingRoleMapping(
                    recording_id="call_test_001",
                    source_file="call_test_001.wav",
                    roles={"SPEAKER_00": "RM", "SPEAKER_01": "CUSTOMER"},
                    review_status="COMPLETED",
                    auditor_id="SeniorAuditor",
                    notes="Manually audited",
                )
            },
        )
        registry.save_json(reg_file)

        # Create AI classification file
        ai_res = RoleClassificationResult(
            recording_id="call_test_001",
            status="AUTO_CLASSIFIED",
            overall_confidence=0.99,
            predicted_roles={"SPEAKER_00": "CUSTOMER", "SPEAKER_01": "RM"},  # Swapped!
            rationale="AI thinks differently",
        )
        ai_res.save_json(ai_dir / "call_test_001.json")

        role_mgr = RoleMappingManager(storage_path=reg_file)
        classifier = RoleClassifier(api_key="mock_key", output_dir=ai_dir)

        # Attempt to apply without force
        success, msg = classifier.apply_prediction_to_registry(
            "call_test_001", role_manager=role_mgr, force=False
        )

        assert success is False
        assert "Preserving manual truth" in msg

        # Verify registry was NOT modified
        reg_after = SpeakerRoleRegistry.from_json(reg_file)
        assert reg_after.mappings["call_test_001"].roles["SPEAKER_00"] == "RM"
        assert reg_after.mappings["call_test_001"].auditor_id == "SeniorAuditor"

    def test_force_override_updates_completed_mapping(self, test_env):
        _, diar_dir, ai_dir, _ = test_env
        reg_file = diar_dir / "speaker_role_mappings.json"

        registry = SpeakerRoleRegistry(
            total_recordings=1,
            completed_reviews=1,
            pending_reviews=0,
            mappings={
                "call_test_001": RecordingRoleMapping(
                    recording_id="call_test_001",
                    source_file="call_test_001.wav",
                    roles={"SPEAKER_00": "CUSTOMER", "SPEAKER_01": "RM"},
                    review_status="COMPLETED",
                    auditor_id="OldAuditor",
                )
            },
        )
        registry.save_json(reg_file)

        ai_res = RoleClassificationResult(
            recording_id="call_test_001",
            status="AUTO_CLASSIFIED",
            overall_confidence=0.99,
            predicted_roles={"SPEAKER_00": "RM", "SPEAKER_01": "CUSTOMER"},
            rationale="Corrected by AI",
        )
        ai_res.save_json(ai_dir / "call_test_001.json")

        role_mgr = RoleMappingManager(storage_path=reg_file)
        classifier = RoleClassifier(api_key="mock_key", output_dir=ai_dir)

        success, msg = classifier.apply_prediction_to_registry(
            "call_test_001", role_manager=role_mgr, force=True
        )

        assert success is True
        reg_after = SpeakerRoleRegistry.from_json(reg_file)
        assert reg_after.mappings["call_test_001"].roles["SPEAKER_00"] == "RM"
        assert reg_after.mappings["call_test_001"].roles["SPEAKER_01"] == "CUSTOMER"


# ---------------------------------------------------------------------------
# Test 9: Auditor Approval Workflow
# ---------------------------------------------------------------------------

class TestAuditorApprovalWorkflow:
    """Transfer AI prediction to PENDING recording in registry upon auditor approval."""

    def test_approve_ai_prediction_updates_registry(self, test_env):
        _, diar_dir, ai_dir, _ = test_env
        reg_file = diar_dir / "speaker_role_mappings.json"

        # PENDING entry in registry
        registry = SpeakerRoleRegistry(
            total_recordings=1,
            completed_reviews=0,
            pending_reviews=1,
            mappings={
                "call_test_001": RecordingRoleMapping(
                    recording_id="call_test_001",
                    source_file="call_test_001.wav",
                    roles={"SPEAKER_00": "UNKNOWN", "SPEAKER_01": "UNKNOWN"},
                    review_status="PENDING",
                )
            },
        )
        registry.save_json(reg_file)

        # AI prediction
        ai_res = RoleClassificationResult(
            recording_id="call_test_001",
            status="AUTO_CLASSIFIED",
            overall_confidence=0.95,
            predicted_roles={"SPEAKER_00": "RM", "SPEAKER_01": "CUSTOMER"},
            rationale="Clear RM intro",
        )
        ai_res.save_json(ai_dir / "call_test_001.json")

        role_mgr = RoleMappingManager(storage_path=reg_file)
        classifier = RoleClassifier(api_key="mock_key", output_dir=ai_dir)

        # Auditor approves
        success, msg = classifier.apply_prediction_to_registry(
            "call_test_001", role_manager=role_mgr, auditor_id="AuditorGaurav"
        )

        assert success is True
        assert "Successfully applied" in msg

        # Verify updated registry
        reg_after = SpeakerRoleRegistry.from_json(reg_file)
        m = reg_after.mappings["call_test_001"]
        assert m.roles["SPEAKER_00"] == "RM"
        assert m.roles["SPEAKER_01"] == "CUSTOMER"
        assert m.review_status == "COMPLETED"
        assert "AI_APPROVED (AuditorGaurav)" in m.auditor_id


# ---------------------------------------------------------------------------
# Test 10: Original Diarized Files Immutability
# ---------------------------------------------------------------------------

class TestOriginalFileImmutability:
    """Original diarized JSON files must remain 100% untouched."""

    def test_original_file_unchanged_during_classification(self, test_env):
        _, diar_dir, ai_dir, _ = test_env
        orig_file = diar_dir / "call_test_001.json"

        with open(orig_file, "r", encoding="utf-8") as f:
            content_before = f.read()
        mtime_before = orig_file.stat().st_mtime

        payload = LLMClassificationPayload(
            speakers=[
                SpeakerEvidence(speaker_id="SPEAKER_00", predicted_role="RM", confidence=0.95),
                SpeakerEvidence(speaker_id="SPEAKER_01", predicted_role="CUSTOMER", confidence=0.95),
            ],
            rationale="Test",
        )

        mock_client = _mock_gemini_client(payload)
        classifier = RoleClassifier(api_key="mock_key", client=mock_client, output_dir=ai_dir)
        classifier.classify_recording("call_test_001", diarized_dir=diar_dir, save_output=True)

        with open(orig_file, "r", encoding="utf-8") as f:
            content_after = f.read()

        assert content_before == content_after
        assert orig_file.stat().st_mtime == mtime_before
        assert (ai_dir / "call_test_001.json").exists()


# ---------------------------------------------------------------------------
# Test 11: Batch Classification and Dry-Run
# ---------------------------------------------------------------------------

class TestBatchClassificationAndDryRun:
    """Verify batch processing and dry-run flag."""

    def test_dry_run_does_not_write_files(self, test_env):
        _, diar_dir, ai_dir, _ = test_env

        payload = LLMClassificationPayload(
            speakers=[
                SpeakerEvidence(speaker_id="SPEAKER_00", predicted_role="RM", confidence=0.95),
                SpeakerEvidence(speaker_id="SPEAKER_01", predicted_role="CUSTOMER", confidence=0.95),
            ],
            rationale="Batch run",
        )

        mock_client = _mock_gemini_client(payload)
        classifier = RoleClassifier(api_key="mock_key", client=mock_client, output_dir=ai_dir)
        summary = classifier.classify_all(diarized_dir=diar_dir, dry_run=True)

        assert summary["total_recordings"] == 2
        assert summary["auto_classified"] == 2
        # No files written to ai_dir
        assert len(list(ai_dir.glob("*.json"))) == 0

    def test_batch_write_generates_summary(self, test_env):
        _, diar_dir, ai_dir, _ = test_env

        payload = LLMClassificationPayload(
            speakers=[
                SpeakerEvidence(speaker_id="SPEAKER_00", predicted_role="RM", confidence=0.95),
                SpeakerEvidence(speaker_id="SPEAKER_01", predicted_role="CUSTOMER", confidence=0.95),
            ],
            rationale="Batch run",
        )

        mock_client = _mock_gemini_client(payload)
        classifier = RoleClassifier(api_key="mock_key", client=mock_client, output_dir=ai_dir)
        summary = classifier.classify_all(diarized_dir=diar_dir, dry_run=False)

        assert summary["total_recordings"] == 2
        assert (ai_dir / "ai_classification_summary.json").exists()
        assert (ai_dir / "call_test_001.json").exists()
        assert (ai_dir / "call_test_002.json").exists()
