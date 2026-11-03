"""
Speaker Role Verification and Manual Mapping Module.

Provides a dedicated manual role-mapping mechanism for auditors to assign
business roles ('RM' or 'CUSTOMER') to generic speaker labels ('SPEAKER_00', 'SPEAKER_01')
after manual review of the call recordings.

Per project requirements:
- Initial state is strictly UNKNOWN for all speakers.
- Business roles are NEVER assigned automatically by heuristic or assumption.
- Mapping state is saved separately in:
    output/diarized_transcripts/speaker_role_mappings.json
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator

from app.diarization.diarization_schema import DiarizedTranscript, SpeakerStats

logger = logging.getLogger("call_audit.diarization.role_mapper")

VALID_ROLES: Set[str] = {"UNKNOWN", "RM", "CUSTOMER"}
VALID_STATUSES: Set[str] = {"PENDING", "IN_REVIEW", "COMPLETED"}


class RecordingRoleMapping(BaseModel):
    """Manual role mapping record for a single audio recording."""
    recording_id: str
    source_file: str
    roles: Dict[str, str] = Field(
        default_factory=lambda: {"SPEAKER_00": "UNKNOWN", "SPEAKER_01": "UNKNOWN"},
        description="Map of speaker_id (e.g. 'SPEAKER_00') to business role ('UNKNOWN', 'RM', 'CUSTOMER')",
    )
    review_status: str = Field(
        default="PENDING",
        description="Review workflow status: PENDING, IN_REVIEW, COMPLETED",
    )
    auditor_id: Optional[str] = Field(
        default=None,
        description="Identifier of the human auditor who assigned or confirmed roles",
    )
    notes: Optional[str] = Field(
        default="",
        description="Auditor observations (e.g. 'Speaker 00 introduces self as Saurabh HDFC RM')",
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of last update",
    )

    @field_validator("roles")
    @classmethod
    def validate_roles(cls, v: Dict[str, str]) -> Dict[str, str]:
        validated = {}
        for spk, role in v.items():
            normalized = role.strip().upper()
            if normalized not in VALID_ROLES:
                raise ValueError(
                    f"Invalid speaker role '{role}'. Allowed roles: {sorted(list(VALID_ROLES))}"
                )
            validated[spk] = normalized
        return validated

    @field_validator("review_status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        norm = v.strip().upper()
        if norm not in VALID_STATUSES:
            raise ValueError(
                f"Invalid review status '{v}'. Allowed statuses: {sorted(list(VALID_STATUSES))}"
            )
        return norm


class SpeakerRoleRegistry(BaseModel):
    """Registry holding role mappings for all recordings."""
    version: str = "1.0"
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_recordings: int = 0
    pending_reviews: int = 0
    completed_reviews: int = 0
    mappings: Dict[str, RecordingRoleMapping] = Field(default_factory=dict)

    def save_json(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return output_path

    @classmethod
    def from_json(cls, file_path: Path) -> "SpeakerRoleRegistry":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)


class RoleMappingManager:
    """
    Manages loading, updating, and applying auditor role mappings.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            from app.core.config import get_settings
            storage_path = get_settings().diarized_transcripts_dir / "speaker_role_mappings.json"
        self.storage_path = Path(storage_path)
        self.registry: SpeakerRoleRegistry = self._load_or_create()

    def _load_or_create(self) -> SpeakerRoleRegistry:
        if self.storage_path.is_file():
            try:
                return SpeakerRoleRegistry.from_json(self.storage_path)
            except Exception as exc:
                logger.warning(
                    "Could not load role registry from %s: %s. Creating new.",
                    self.storage_path, exc,
                )
        return SpeakerRoleRegistry()

    def _update_counts(self) -> None:
        self.registry.total_recordings = len(self.registry.mappings)
        self.registry.completed_reviews = sum(
            1 for m in self.registry.mappings.values() if m.review_status == "COMPLETED"
        )
        self.registry.pending_reviews = (
            self.registry.total_recordings - self.registry.completed_reviews
        )
        self.registry.updated_at = datetime.now(timezone.utc).isoformat()

    def initialize_from_transcripts(
        self,
        diarized_dir: Path,
        overwrite: bool = False,
    ) -> SpeakerRoleRegistry:
        """
        Scan diarized transcripts and initialize role mappings with UNKNOWN
        for all discovered speakers.
        """
        json_files = sorted(
            [f for f in diarized_dir.glob("*.json") if not f.name.endswith("_summary.json") and not f.name.endswith("_report.json") and not f.name.endswith("_mappings.json")]
        )

        for jf in json_files:
            try:
                dt = DiarizedTranscript.from_json(jf)
                rec_id = dt.recording_id

                if rec_id in self.registry.mappings and not overwrite:
                    # Keep existing auditor mapping
                    continue

                initial_roles = {spk: "UNKNOWN" for spk in dt.speaker_ids}
                if not initial_roles:
                    initial_roles = {"SPEAKER_00": "UNKNOWN", "SPEAKER_01": "UNKNOWN"}

                self.registry.mappings[rec_id] = RecordingRoleMapping(
                    recording_id=rec_id,
                    source_file=dt.source_file,
                    roles=initial_roles,
                    review_status="PENDING",
                    auditor_id=None,
                    notes="",
                )
            except Exception as exc:
                logger.warning("Error reading transcript %s during role init: %s", jf, exc)

        self._update_counts()
        self.save()
        return self.registry

    def assign_role(
        self,
        recording_id: str,
        speaker_id: str,
        role: str,
        auditor_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> RecordingRoleMapping:
        """
        Assign a verified business role ('RM', 'CUSTOMER', or 'UNKNOWN') to a speaker.
        """
        norm_role = role.strip().upper()
        if norm_role not in VALID_ROLES:
            raise ValueError(f"Invalid role '{role}'. Allowed: {sorted(list(VALID_ROLES))}")

        norm_speaker = speaker_id.strip().upper()

        mapping = self.registry.mappings.get(recording_id)
        if not mapping:
            raise KeyError(f"Recording '{recording_id}' not found in role registry.")

        mapping.roles[norm_speaker] = norm_role
        if auditor_id:
            mapping.auditor_id = auditor_id
        if notes:
            mapping.notes = notes

        # Update status
        all_assigned = all(r != "UNKNOWN" for r in mapping.roles.values())
        mapping.review_status = "COMPLETED" if all_assigned else "IN_REVIEW"
        mapping.updated_at = datetime.now(timezone.utc).isoformat()

        self._update_counts()
        self.save()
        logger.info(
            "Assigned role for %s: %s -> %s (status: %s)",
            recording_id, norm_speaker, norm_role, mapping.review_status,
        )
        return mapping

    def get_roles(self, recording_id: str) -> Dict[str, str]:
        """Get role mapping dictionary for a recording."""
        mapping = self.registry.mappings.get(recording_id)
        if mapping:
            return dict(mapping.roles)
        return {"SPEAKER_00": "UNKNOWN", "SPEAKER_01": "UNKNOWN"}

    def apply_to_transcript(
        self,
        diarized: DiarizedTranscript,
    ) -> DiarizedTranscript:
        """
        Return a copy of DiarizedTranscript with speaker_stats.role updated
        according to the current manual role mappings.
        """
        roles = self.get_roles(diarized.recording_id)
        updated_stats = []
        for ss in diarized.speaker_stats:
            assigned = roles.get(ss.speaker_id, "UNKNOWN")
            updated_stats.append(
                SpeakerStats(
                    speaker_id=ss.speaker_id,
                    role=assigned,
                    total_segments=ss.total_segments,
                    total_words=ss.total_words,
                    total_talk_time_seconds=ss.total_talk_time_seconds,
                    talk_time_percentage=ss.talk_time_percentage,
                    first_segment_start=ss.first_segment_start,
                    last_segment_end=ss.last_segment_end,
                )
            )

        data = diarized.model_dump()
        data["speaker_stats"] = [s.model_dump() for s in updated_stats]
        return DiarizedTranscript.model_validate(data)

    def save(self, path: Optional[Path] = None) -> Path:
        save_path = path or self.storage_path
        return self.registry.save_json(save_path)
