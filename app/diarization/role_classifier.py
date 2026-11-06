"""
M4.5 Automatic RM vs Customer Role Classification Module.

Leverages Google Gemini (gemini-2.5-flash via google-genai SDK) to classify generic
speaker IDs (e.g. 'SPEAKER_00', 'SPEAKER_01') into business roles ('RM' vs 'CUSTOMER')
by analyzing the complete contextual dialogue.

Core Principles:
- Contextual Analysis: Evaluates the entire conversation holistically (introductions,
  institutional identity, portfolio discussions, questions/responses) rather than
  shallow keywords or speaker sequence.
- Anti-Heuristic: Explicitly rejects assuming SPEAKER_00 is RM or that the first speaker is RM.
- Safety & Fallback: Low-confidence (< 0.85), ambiguous, or conflicting classifications
  are marked as 'NEEDS_MANUAL_REVIEW' with roles set to 'UNKNOWN'.
- Isolation: AI predictions are saved separately under
  output/diarized_transcripts/ai_role_classifications/.
  They NEVER overwrite original diarized transcripts or verified manual mappings.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from app.core.config import Settings, get_settings
from app.diarization.diarization_schema import DiarizedTranscript
from app.diarization.role_mapper import RoleMappingManager

logger = logging.getLogger("call_audit.diarization.role_classifier")

# ---------------------------------------------------------------------------
# Structured Schemas for Gemini Response and Persisted Results
# ---------------------------------------------------------------------------

class SpeakerEvidence(BaseModel):
    """Classification and evidence for an individual speaker."""
    speaker_id: str = Field(description="Generic speaker identifier, e.g. 'SPEAKER_00', 'SPEAKER_01'")
    predicted_role: str = Field(description="Predicted business role: 'RM', 'CUSTOMER', or 'UNKNOWN'")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    role_indicators: List[str] = Field(
        default_factory=list,
        description="Behavioral and contextual indicators observed for this speaker",
    )
    supporting_quotes: List[str] = Field(
        default_factory=list,
        description="Exact quotes from transcript supporting this role assignment",
    )

    @field_validator("predicted_role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        norm = v.strip().upper()
        if norm not in {"RM", "CUSTOMER", "UNKNOWN"}:
            return "UNKNOWN"
        return norm


class LLMClassificationPayload(BaseModel):
    """Schema enforced on the Gemini structured output."""
    speakers: List[SpeakerEvidence] = Field(
        description="List of classifications for all detected speakers in the conversation",
    )
    rationale: str = Field(
        description="Detailed contextual rationale explaining who represents HDFC Securities and who is the client",
    )


class RoleClassificationResult(BaseModel):
    """Persisted classification result for an audio recording."""
    recording_id: str
    source_file: str = ""
    status: str = Field(
        description="Classification status: 'AUTO_CLASSIFIED', 'NEEDS_MANUAL_REVIEW', or 'ERROR'",
    )
    overall_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall decision confidence (minimum of speaker confidences)",
    )
    predicted_roles: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of speaker_id -> predicted_role",
    )
    rationale: str = ""
    speakers: List[SpeakerEvidence] = Field(default_factory=list)
    model_name: str = "gemini-2.5-flash"
    confidence_threshold: float = 0.85
    classified_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    error_details: Optional[str] = None

    def save_json(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return output_path

    @classmethod
    def from_json(cls, file_path: Path) -> "RoleClassificationResult":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Prompt Engineering
# ---------------------------------------------------------------------------

CLASSIFICATION_SYSTEM_INSTRUCTION = """
You are an expert Indian banking call audit intelligence system specializing in HDFC Securities relationship management calls.
Your objective is to accurately identify which speaker is the Relationship Manager (RM) and which speaker is the Customer.

CONVERSATIONAL CONTEXT GUIDELINES:
1. RELATIONSHIP MANAGER (RM):
   - Represents the financial institution (HDFC Securities / HDFC Bank).
   - Identifies themselves and their organization (e.g., "Hi Saurabh बोल रहा हूं HDFC Securities से", "service call किया था").
   - Discusses investment products, trading accounts, portfolio status, equity holdings (e.g. Tata Motors), E-Margin positions (e.g., 275 days completion), margin shortfall alerts, or mutual fund offerings.
   - Typically asks if it is a convenient time to speak ("good time to talk?", "दो मिनट बात हो सकती है?").
   - Explains procedures, solutions, or advises averaging loss positions.

2. CUSTOMER:
   - Account holder or authorized representative (e.g. spouse, family member answering the phone).
   - In outbound calls, the Customer ALMOST ALWAYS answers the phone first (e.g. "Hello", "हाँ बोलो", "Wait one minute", "रुको देवी को देती हूं").
   - Responds to RM's inquiries, confirms identity, asks questions about their investments or shortfalls, or acknowledges messages ("हाँ वो आते रहता है", "क्या हुआ?").

CRITICAL NEGATIVE CONSTRAINTS (DO NOT VIOLATE):
- NEVER assume SPEAKER_00 is always the RM. In many calls, SPEAKER_00 is the Customer and SPEAKER_01 is the RM.
- NEVER assume the first speaker in the transcript is the RM. The party receiving the call speaks first.
- NEVER rely solely on single keyword occurrences. For example, both speakers may mention the RM's name (the customer may greet "हाँ सौरभ" before the RM introduces himself). Analyze who is addressing whom.
- In a valid two-speaker customer relationship call, exactly one speaker must be RM and one speaker must be CUSTOMER.
- Assign a confidence score (0.0 to 1.0) for each speaker reflecting unambiguous textual evidence.
- If the call is disconnected immediately, is purely IVR/voicemail, or evidence is contradictory or insufficient, assign role UNKNOWN and confidence below 0.60.
"""


# ---------------------------------------------------------------------------
# Classifier Service Class
# ---------------------------------------------------------------------------

class RoleClassifier:
    """
    Automated Speaker Role Classifier using Gemini LLM.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        output_dir: Optional[Path] = None,
        client: Optional[Any] = None,
    ):
        self.settings = settings or get_settings()
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = self.settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name or self.settings.GEMINI_MODEL
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.settings.ROLE_CLASSIFICATION_CONFIDENCE_THRESHOLD
        )
        self.output_dir = output_dir or self.settings.ai_role_classifications_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._client = client

    @property
    def client(self):
        """Lazy initialization of the Google GenAI client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in .env "
                    "or pass api_key to RoleClassifier."
                )
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "google-genai package is not installed. Run: pip install google-genai"
                )
        return self._client

    @client.setter
    def client(self, value):
        self._client = value

    def _build_conversation_prompt(self, diarized: DiarizedTranscript) -> str:
        """Format the entire transcript with timing and statistics for LLM evaluation."""
        prompt_lines = [
            f"RECORDING ID: {diarized.recording_id}",
            f"DURATION: {diarized.duration:.1f} seconds",
            f"NUMBER OF SPEAKERS DETECTED: {diarized.num_speakers_detected}",
            "",
            "SPEAKER OVERVIEW & TALK TIME:",
        ]
        for ss in diarized.speaker_stats:
            prompt_lines.append(
                f"- {ss.speaker_id}: {ss.total_segments} segments, {ss.total_words} words, "
                f"{ss.total_talk_time_seconds:.1f}s ({ss.talk_time_percentage:.1f}%)"
            )

        prompt_lines.extend(["", "COMPLETE CHRONOLOGICAL DIARIZED TRANSCRIPT:", ""])
        for seg in diarized.segments:
            m_s, s_s = divmod(seg.start, 60)
            m_e, s_e = divmod(seg.end, 60)
            ts = f"[{int(m_s):02d}:{s_s:04.1f} -> {int(m_e):02d}:{s_e:04.1f}]"
            prompt_lines.append(f"{ts} {seg.speaker_id}: {seg.text}")

        prompt_lines.extend([
            "",
            "TASK INSTRUCTIONS:",
            "Analyze the above complete conversation and determine which speaker is the Relationship Manager (RM) "
            "and which is the Customer. Provide exact supporting quotes and key indicators for each speaker.",
        ])
        return "\n".join(prompt_lines)

    def classify_transcript(
        self,
        diarized: DiarizedTranscript,
    ) -> RoleClassificationResult:
        """
        Classify speaker roles for a DiarizedTranscript using Gemini.
        Returns a validated RoleClassificationResult.
        """
        rec_id = diarized.recording_id

        # Safeguard: check API key availability
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not configured for %s", rec_id)
            return RoleClassificationResult(
                recording_id=rec_id,
                source_file=diarized.source_file,
                status="ERROR",
                overall_confidence=0.0,
                predicted_roles={spk: "UNKNOWN" for spk in diarized.speaker_ids},
                rationale="GEMINI_API_KEY is not configured.",
                model_name=self.model_name,
                confidence_threshold=self.confidence_threshold,
                error_details="GEMINI_API_KEY is missing. Cannot perform AI classification.",
            )

        # Safeguard: check if audio has valid segments
        if not diarized.segments or diarized.num_speakers_detected == 0:
            return RoleClassificationResult(
                recording_id=rec_id,
                source_file=diarized.source_file,
                status="NEEDS_MANUAL_REVIEW",
                overall_confidence=0.0,
                predicted_roles={},
                rationale="No speech segments or speakers detected in transcript.",
                model_name=self.model_name,
                confidence_threshold=self.confidence_threshold,
            )

        prompt_text = self._build_conversation_prompt(diarized)

        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                system_instruction=CLASSIFICATION_SYSTEM_INSTRUCTION,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=LLMClassificationPayload,
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt_text,
                config=config,
            )

            # Parse structured response
            raw_text = response.text
            payload_data = json.loads(raw_text)
            parsed_payload = LLMClassificationPayload.model_validate(payload_data)

            # Evaluate confidence and validity
            return self._validate_and_build_result(
                diarized=diarized,
                payload=parsed_payload,
            )

        except Exception as exc:
            logger.error("AI role classification failed for %s: %s", rec_id, exc)
            return RoleClassificationResult(
                recording_id=rec_id,
                source_file=diarized.source_file,
                status="ERROR",
                overall_confidence=0.0,
                predicted_roles={spk: "UNKNOWN" for spk in diarized.speaker_ids},
                rationale=f"LLM classification failure: {str(exc)}",
                model_name=self.model_name,
                confidence_threshold=self.confidence_threshold,
                error_details=str(exc),
            )

    def _validate_and_build_result(
        self,
        diarized: DiarizedTranscript,
        payload: LLMClassificationPayload,
    ) -> RoleClassificationResult:
        """
        Validate LLM response against business rules:
        - Must contain all speakers.
        - Two-speaker calls must have exactly one RM and one CUSTOMER.
        - Both speakers must satisfy confidence >= threshold (0.85).
        """
        rec_id = diarized.recording_id
        speaker_map = {s.speaker_id: s for s in payload.speakers}
        detected_speakers = set(diarized.speaker_ids)

        roles_dict: Dict[str, str] = {}
        confidences: List[float] = []
        validated_speakers: List[SpeakerEvidence] = []

        for spk_id in diarized.speaker_ids:
            if spk_id in speaker_map:
                evidence = speaker_map[spk_id]
                roles_dict[spk_id] = evidence.predicted_role
                confidences.append(evidence.confidence)
                validated_speakers.append(evidence)
            else:
                # Speaker omitted by LLM
                roles_dict[spk_id] = "UNKNOWN"
                confidences.append(0.0)
                validated_speakers.append(
                    SpeakerEvidence(
                        speaker_id=spk_id,
                        predicted_role="UNKNOWN",
                        confidence=0.0,
                        role_indicators=["Speaker omitted from model classification payload."],
                    )
                )

        overall_conf = min(confidences) if confidences else 0.0

        # Check two-speaker mutual exclusivity requirement
        role_counts: Dict[str, int] = {}
        for r in roles_dict.values():
            role_counts[r] = role_counts.get(r, 0) + 1

        is_valid_two_speaker = (
            len(detected_speakers) == 2
            and role_counts.get("RM", 0) == 1
            and role_counts.get("CUSTOMER", 0) == 1
        )

        passes_confidence = overall_conf >= self.confidence_threshold

        if is_valid_two_speaker and passes_confidence:
            status = "AUTO_CLASSIFIED"
        else:
            status = "NEEDS_MANUAL_REVIEW"
            if not is_valid_two_speaker:
                logger.info(
                    "Recording %s flagged for manual review: invalid role balance %s",
                    rec_id, role_counts,
                )
            elif not passes_confidence:
                logger.info(
                    "Recording %s flagged for manual review: confidence %.2f < threshold %.2f",
                    rec_id, overall_conf, self.confidence_threshold,
                )

        return RoleClassificationResult(
            recording_id=rec_id,
            source_file=diarized.source_file,
            status=status,
            overall_confidence=round(overall_conf, 3),
            predicted_roles=roles_dict,
            rationale=payload.rationale,
            speakers=validated_speakers,
            model_name=self.model_name,
            confidence_threshold=self.confidence_threshold,
        )

    def classify_recording(
        self,
        recording_id: str,
        diarized_dir: Optional[Path] = None,
        save_output: bool = True,
    ) -> Optional[RoleClassificationResult]:
        """Load original diarized JSON, classify, and save result."""
        in_dir = diarized_dir or self.settings.diarized_transcripts_dir
        json_file = in_dir / f"{recording_id}.json"

        if not json_file.is_file():
            matching = [
                f for f in in_dir.glob(f"*{recording_id}*.json")
                if not f.name.endswith("_summary.json")
                and not f.name.endswith("_report.json")
                and not f.name.endswith("_mappings.json")
            ]
            if matching:
                json_file = matching[0]
            else:
                logger.error("Original diarized file for '%s' not found in %s", recording_id, in_dir)
                return None

        transcript = DiarizedTranscript.from_json(json_file)
        result = self.classify_transcript(transcript)

        if save_output:
            out_file = self.output_dir / f"{transcript.recording_id}.json"
            result.save_json(out_file)
            logger.info("Saved AI role classification for %s to %s", transcript.recording_id, out_file)

        return result

    def classify_all(
        self,
        diarized_dir: Optional[Path] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Classify all diarized recordings in diarized_dir."""
        in_dir = diarized_dir or self.settings.diarized_transcripts_dir
        json_files = sorted([
            f for f in in_dir.glob("*.json")
            if not f.name.endswith("_summary.json")
            and not f.name.endswith("_report.json")
            and not f.name.endswith("_mappings.json")
        ])

        if not json_files:
            return {"total_recordings": 0, "results": []}

        results: List[Dict[str, Any]] = []
        for jf in json_files:
            try:
                transcript = DiarizedTranscript.from_json(jf)
                res = self.classify_transcript(transcript)

                if not dry_run:
                    out_path = self.output_dir / f"{transcript.recording_id}.json"
                    res.save_json(out_path)

                results.append(res.model_dump())
            except Exception as exc:
                logger.error("Failed batch classification for %s: %s", jf.name, exc)
                results.append({
                    "recording_id": jf.stem,
                    "status": "ERROR",
                    "error": str(exc),
                })

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": self.model_name,
            "confidence_threshold": self.confidence_threshold,
            "total_recordings": len(json_files),
            "auto_classified": sum(1 for r in results if r.get("status") == "AUTO_CLASSIFIED"),
            "needs_manual_review": sum(1 for r in results if r.get("status") == "NEEDS_MANUAL_REVIEW"),
            "errors": sum(1 for r in results if r.get("status") == "ERROR"),
            "output_directory": str(self.output_dir) if not dry_run else "(dry-run)",
            "results": results,
        }

        if not dry_run:
            summary_path = self.output_dir / "ai_classification_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            summary["summary_path"] = str(summary_path)

        return summary

    def apply_prediction_to_registry(
        self,
        recording_id: str,
        role_manager: Optional[RoleMappingManager] = None,
        auditor_id: str = "Auditor",
        force: bool = False,
    ) -> Tuple[bool, str]:
        """
        Auditor approval workflow:
        Explicitly transfer an AI prediction into the manual role registry (speaker_role_mappings.json).

        Safety Rules:
        - If existing status is already 'COMPLETED', NEVER overwrite unless force=True.
        - Only transfers if AI classification file exists.
        """
        role_mgr = role_manager or RoleMappingManager()
        ai_file = self.output_dir / f"{recording_id}.json"

        if not ai_file.is_file():
            # Try finding by matching stem
            matching = list(self.output_dir.glob(f"*{recording_id}*.json"))
            matching = [m for m in matching if not m.name.endswith("_summary.json")]
            if matching:
                ai_file = matching[0]
            else:
                return False, f"No AI classification found for '{recording_id}'."

        ai_result = RoleClassificationResult.from_json(ai_file)

        # Check existing manual mapping
        existing = role_mgr.registry.mappings.get(ai_result.recording_id)
        if existing and existing.review_status == "COMPLETED" and not force:
            return False, (
                f"Recording '{ai_result.recording_id}' already has COMPLETED manual review by '{existing.auditor_id}'. "
                "Preserving manual truth (use force=True to override)."
            )

        # Apply AI predicted roles
        for spk_id, role in ai_result.predicted_roles.items():
            if role in {"RM", "CUSTOMER"}:
                role_mgr.assign_role(
                    recording_id=ai_result.recording_id,
                    speaker_id=spk_id,
                    role=role,
                    auditor_id=f"AI_APPROVED ({auditor_id})",
                    notes=f"Approved AI prediction ({ai_result.model_name}, conf={ai_result.overall_confidence:.2f}): {ai_result.rationale[:120]}...",
                )

        return True, f"Successfully applied AI roles for '{ai_result.recording_id}'."
