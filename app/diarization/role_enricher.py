"""
Role Enrichment Service for M4 Diarization Transcripts.

Transforms raw speaker-diarized transcripts into role-enriched transcripts by
attaching business roles ('RM', 'CUSTOMER', or 'UNKNOWN') to every segment
based on verified auditor mappings from speaker_role_mappings.json.

Safety Guarantees:
- Original Sarvam diarized JSON and TXT files are NEVER modified.
- Outputs are saved in a dedicated subfolder: output/diarized_transcripts/role_enriched/
- Mappings are ONLY applied when review_status == 'COMPLETED'. Otherwise, 'UNKNOWN' is used.
- SPEAKER_00 is NEVER assumed to be RM; recording-specific verified mappings are used.
- Missing or invalid mapping files are handled gracefully with clear logging.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import Settings, get_settings
from app.diarization.diarization_schema import (
    DiarizedSegment,
    DiarizedTranscript,
    SpeakerStats,
)
from app.diarization.role_mapper import RoleMappingManager, SpeakerRoleRegistry

logger = logging.getLogger("call_audit.diarization.role_enricher")


class RoleEnricher:
    """
    Enriches diarized transcripts with verified speaker roles.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        role_manager: Optional[RoleMappingManager] = None,
        output_dir: Optional[Path] = None,
    ):
        self.settings = settings or get_settings()
        self.role_manager = role_manager or RoleMappingManager()
        self.output_dir = output_dir or self.settings.role_enriched_transcripts_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_verified_roles(self, recording_id: str) -> Tuple[Dict[str, str], str]:
        """
        Retrieve verified roles for a recording.

        Returns:
            (roles_dict, review_status)
            If review_status != 'COMPLETED', roles will safely be all 'UNKNOWN'.
        """
        try:
            mapping = self.role_manager.registry.mappings.get(recording_id)
            if not mapping:
                logger.warning(
                    "Recording '%s' not found in role registry. Falling back to UNKNOWN.",
                    recording_id,
                )
                return {}, "MISSING_MAPPING"

            status = mapping.review_status.upper()
            if status == "COMPLETED":
                logger.info(
                    "Applying COMPLETED role mapping for '%s': %s",
                    recording_id, mapping.roles,
                )
                return dict(mapping.roles), "COMPLETED"
            else:
                logger.info(
                    "Recording '%s' status is '%s' (not COMPLETED). Preserving UNKNOWN roles.",
                    recording_id, status,
                )
                # Safeguard: All speakers remain UNKNOWN until review is fully completed
                safe_roles = {spk: "UNKNOWN" for spk in mapping.roles.keys()}
                return safe_roles, status

        except Exception as exc:
            logger.error(
                "Error retrieving role mapping for '%s': %s. Falling back to UNKNOWN.",
                recording_id, exc,
            )
            return {}, "ERROR"

    def enrich_transcript(
        self,
        diarized: DiarizedTranscript,
    ) -> DiarizedTranscript:
        """
        Create a new role-enriched DiarizedTranscript from an existing one.
        The input transcript object is not modified.
        """
        rec_id = diarized.recording_id
        roles, status = self.get_verified_roles(rec_id)

        # Enrich segments with speaker_role
        enriched_segments: List[DiarizedSegment] = []
        for seg in diarized.segments:
            role = roles.get(seg.speaker_id, "UNKNOWN")
            enriched_segments.append(
                DiarizedSegment(
                    segment_id=seg.segment_id,
                    speaker_id=seg.speaker_id,
                    speaker_role=role,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                )
            )

        # Enrich speaker stats with role
        enriched_stats: List[SpeakerStats] = []
        for ss in diarized.speaker_stats:
            role = roles.get(ss.speaker_id, "UNKNOWN")
            enriched_stats.append(
                SpeakerStats(
                    speaker_id=ss.speaker_id,
                    role=role,
                    total_segments=ss.total_segments,
                    total_words=ss.total_words,
                    total_talk_time_seconds=ss.total_talk_time_seconds,
                    talk_time_percentage=ss.talk_time_percentage,
                    first_segment_start=ss.first_segment_start,
                    last_segment_end=ss.last_segment_end,
                )
            )

        data = diarized.model_dump()
        data["segments"] = [s.model_dump() for s in enriched_segments]
        data["speaker_stats"] = [s.model_dump() for s in enriched_stats]

        # Attach enrichment metadata without overwriting provider metadata
        prov_meta = data.get("provider_metadata") or {}
        prov_meta["role_enrichment"] = {
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "review_status": status,
            "roles_applied": roles if status == "COMPLETED" else "NONE_PENDING_REVIEW",
        }
        data["provider_metadata"] = prov_meta

        return DiarizedTranscript.model_validate(data)

    def generate_role_labeled_text(self, enriched: DiarizedTranscript) -> str:
        """
        Generate a human-readable role-labeled script text.
        Format:
            [mm:ss -> mm:ss] SPEAKER_00 (RM): text...
        """
        spk_summary = ", ".join(
            f"{ss.speaker_id} ({ss.role})" for ss in enriched.speaker_stats
        )

        lines = [
            "=" * 85,
            f"ROLE-ENRICHED DIARIZED TRANSCRIPT: {enriched.recording_id}",
            f"RECORDING:          {enriched.recording_id}",
            f"DURATION:           {enriched.duration:.1f} seconds",
            f"Source Audio:       {enriched.source_file}",
            f"Speakers & Roles:   {spk_summary}",
            f"Engine:             {enriched.model} (mode: {enriched.mode})",
            "=" * 85,
            "",
            "SPEAKER STATISTICS:",
        ]

        for ss in enriched.speaker_stats:
            lines.append(
                f"  - {ss.speaker_id} ({ss.role}): "
                f"{ss.total_segments} segments, {ss.total_words} words, "
                f"{ss.total_talk_time_seconds:.1f}s ({ss.talk_time_percentage:.1f}%)"
            )

        lines.extend(["", "-" * 85, "ROLE-LABELLED CONVERSATION:", "-" * 85, ""])

        for seg in enriched.segments:
            m_s, s_s = divmod(seg.start, 60)
            m_e, s_e = divmod(seg.end, 60)
            ts = f"[{int(m_s):02d}:{s_s:04.1f} -> {int(m_e):02d}:{s_e:04.1f}]"
            label = f"{seg.speaker_id} ({seg.speaker_role}):"
            lines.append(f"{ts:<22} {label:<26} {seg.text}")

        return "\n".join(lines)

    def save_enriched_outputs(
        self,
        enriched: DiarizedTranscript,
        output_dir: Optional[Path] = None,
    ) -> Tuple[Path, Path]:
        """
        Save enriched JSON and TXT outputs into the role_enriched directory.
        Original files are never overwritten.
        """
        out_dir = output_dir or self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / f"{enriched.recording_id}.json"
        txt_path = out_dir / f"{enriched.recording_id}.txt"

        # 1. Save JSON
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(enriched.model_dump_json(indent=2))

        # 2. Save TXT
        txt_content = self.generate_role_labeled_text(enriched)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)

        logger.info(
            "Saved role-enriched outputs for %s to %s and %s",
            enriched.recording_id, json_path.name, txt_path.name,
        )
        return (json_path, txt_path)

    def enrich_recording(
        self,
        recording_id: str,
        diarized_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> Optional[Tuple[Path, Path]]:
        """
        Load an original diarized JSON by recording_id, enrich it, and save to role_enriched/.
        """
        in_dir = diarized_dir or self.settings.diarized_transcripts_dir
        json_file = in_dir / f"{recording_id}.json"

        if not json_file.is_file():
            # Try finding by matching stem
            matching = list(in_dir.glob(f"*{recording_id}*.json"))
            matching = [
                m for m in matching
                if not m.name.endswith("_summary.json")
                and not m.name.endswith("_report.json")
                and not m.name.endswith("_mappings.json")
            ]
            if matching:
                json_file = matching[0]
            else:
                logger.error(
                    "Original diarized file for '%s' not found in %s",
                    recording_id, in_dir,
                )
                return None

        # Load original in read-only fashion
        original = DiarizedTranscript.from_json(json_file)
        enriched = self.enrich_transcript(original)
        return self.save_enriched_outputs(enriched, output_dir=output_dir)

    def enrich_all(
        self,
        diarized_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Enrich all original diarized transcripts in diarized_dir and save to role_enriched/.
        """
        in_dir = diarized_dir or self.settings.diarized_transcripts_dir
        out_dir = output_dir or self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Discover original JSON files (ignoring subdirectories and summaries)
        json_files = sorted([
            f for f in in_dir.glob("*.json")
            if not f.name.endswith("_summary.json")
            and not f.name.endswith("_report.json")
            and not f.name.endswith("_mappings.json")
        ])

        if not json_files:
            logger.warning("No diarized JSON files found to enrich in %s", in_dir)
            return {"total_enriched": 0, "results": []}

        results: List[Dict[str, Any]] = []
        for jf in json_files:
            try:
                original = DiarizedTranscript.from_json(jf)
                enriched = self.enrich_transcript(original)
                json_path, txt_path = self.save_enriched_outputs(enriched, output_dir=out_dir)

                roles, status = self.get_verified_roles(original.recording_id)
                results.append({
                    "recording_id": original.recording_id,
                    "review_status": status,
                    "roles_applied": roles if status == "COMPLETED" else "UNKNOWN",
                    "json_output": str(json_path),
                    "txt_output": str(txt_path),
                    "segment_count": len(enriched.segments),
                })
            except Exception as exc:
                logger.error("Failed to enrich %s: %s", jf.name, exc)
                results.append({
                    "recording_id": jf.stem,
                    "error": str(exc),
                })

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_recordings": len(json_files),
            "successfully_enriched": sum(1 for r in results if "error" not in r),
            "completed_reviews_applied": sum(1 for r in results if r.get("review_status") == "COMPLETED"),
            "pending_reviews": sum(1 for r in results if r.get("review_status") != "COMPLETED"),
            "output_directory": str(out_dir),
            "results": results,
        }

        summary_path = out_dir / "role_enrichment_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        summary["summary_report_path"] = str(summary_path)
        logger.info("Batch role enrichment complete. Summary saved to %s", summary_path)
        return summary
