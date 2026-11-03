"""
Speaker diarization and RM vs customer role separation module.

M4: Identifies distinct speakers in mono call recordings using
Sarvam Saaras v4 Batch API with diarization support.
Includes quality validation, manual speaker role verification,
and safe role enrichment for downstream compliance auditing.
"""

from app.diarization.diarization_schema import (
    DiarizedSegment,
    DiarizedTranscript,
    SpeakerStats,
)
from app.diarization.sarvam_diarizer import SarvamDiarizer
from app.diarization.processor import DiarizationProcessor, DiarizationError
from app.diarization.quality_validator import (
    DiarizationQualityValidator,
    RecordingQualityReport,
    DiarizationQualityReport,
)
from app.diarization.role_mapper import (
    RoleMappingManager,
    RecordingRoleMapping,
    SpeakerRoleRegistry,
    VALID_ROLES,
)
from app.diarization.role_enricher import RoleEnricher

__all__ = [
    "DiarizedSegment",
    "DiarizedTranscript",
    "SpeakerStats",
    "SarvamDiarizer",
    "DiarizationProcessor",
    "DiarizationError",
    "DiarizationQualityValidator",
    "RecordingQualityReport",
    "DiarizationQualityReport",
    "RoleMappingManager",
    "RecordingRoleMapping",
    "SpeakerRoleRegistry",
    "VALID_ROLES",
    "RoleEnricher",
]
