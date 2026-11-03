#!/usr/bin/env python
"""
Speaker Role Management CLI for Human Auditors.

Allows an auditor to:
- List all recordings and their current speaker roles (SPEAKER_00, SPEAKER_01).
- Initialize or reset role mappings with default 'UNKNOWN'.
- Manually assign 'RM' or 'CUSTOMER' to speakers after listening to a recording.
- Add auditor review notes and track completion status.
- Trigger role enrichment (--enrich) for all or a specific recording.
- Auto-regenerate enriched outputs when --assign completes a review.

Usage:
    python scripts/manage_speaker_roles.py --list
    python scripts/manage_speaker_roles.py --init
    python scripts/manage_speaker_roles.py --assign --recording <id> --speaker SPEAKER_00 --role RM --auditor "Auditor1" --notes "Self-identified as RM Saurabh"
    python scripts/manage_speaker_roles.py --summary
    python scripts/manage_speaker_roles.py --enrich
    python scripts/manage_speaker_roles.py --enrich --recording <id>
"""

import argparse
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
from app.diarization.role_mapper import RoleMappingManager, VALID_ROLES
from app.diarization.role_enricher import RoleEnricher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auditor Speaker Role Management (RM vs Customer Manual Assignment)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List all recordings and their role assignments")
    group.add_argument("--init", action="store_true", help="Initialize/refresh role mappings with default UNKNOWN")
    group.add_argument("--assign", action="store_true", help="Assign a role to a specific speaker")
    group.add_argument("--summary", action="store_true", help="Show auditor review progress summary")
    group.add_argument("--enrich", action="store_true", help="Generate role-enriched outputs for all or a specific recording")

    parser.add_argument("--recording", type=str, help="Recording ID or stem (required for --assign, optional for --enrich)")
    parser.add_argument("--speaker", type=str, help="Speaker ID (e.g. SPEAKER_00, SPEAKER_01)")
    parser.add_argument(
        "--role",
        type=str,
        choices=["RM", "CUSTOMER", "UNKNOWN"],
        help="Business role to assign (RM, CUSTOMER, UNKNOWN)",
    )
    parser.add_argument("--auditor", type=str, default=None, help="Auditor name or ID")
    parser.add_argument("--notes", type=str, default=None, help="Auditor review notes/rationale")
    parser.add_argument("--force", action="store_true", help="Force overwrite during --init")

    return parser.parse_args()


def _auto_enrich_recording(enricher: RoleEnricher, recording_id: str) -> None:
    """Auto-trigger role enrichment for a specific recording after role assignment."""
    print(f"\n  [AUTO-ENRICH] Regenerating role-enriched output for: {recording_id}")
    result = enricher.enrich_recording(recording_id)
    if result:
        json_path, txt_path = result
        print(f"  [AUTO-ENRICH] JSON: {json_path}")
        print(f"  [AUTO-ENRICH] TXT:  {txt_path}")
    else:
        print(f"  [AUTO-ENRICH] Warning: Could not enrich '{recording_id}' (original file not found)")


def main() -> int:
    args = parse_args()
    settings = get_settings()
    diarized_dir = settings.diarized_transcripts_dir
    role_mgr = RoleMappingManager(storage_path=diarized_dir / "speaker_role_mappings.json")
    enricher = RoleEnricher(settings=settings, role_manager=role_mgr)

    if args.init:
        print("Initializing role registry from diarized transcripts...")
        registry = role_mgr.initialize_from_transcripts(diarized_dir, overwrite=args.force)
        print(f"Registry updated at: {role_mgr.storage_path}")
        print(f"Total recordings: {registry.total_recordings} | Pending review: {registry.pending_reviews}")
        return 0

    if args.summary:
        reg = role_mgr.registry
        print("=" * 70)
        print("  SPEAKER ROLE AUDIT PROGRESS")
        print("=" * 70)
        print(f"  Registry File:      {role_mgr.storage_path}")
        print(f"  Total Recordings:   {reg.total_recordings}")
        print(f"  Pending Reviews:    {reg.pending_reviews}")
        print(f"  Completed Reviews:  {reg.completed_reviews}")
        print(f"  Last Updated:       {reg.updated_at}")
        print("=" * 70)
        return 0

    if args.list:
        reg = role_mgr.registry
        if not reg.mappings:
            print("[INFO] No mappings found in registry. Run with '--init' first.")
            return 0

        print("=" * 95)
        print(f"  SPEAKER ROLE MAPPINGS ({len(reg.mappings)} recordings)")
        print("=" * 95)
        fmt_hdr = "  {:<42} {:<11} {:<25} {}"
        print(fmt_hdr.format("Recording ID", "Status", "Roles", "Auditor"))
        print("  " + "-" * 91)
        for rec_id, m in reg.mappings.items():
            roles_str = ", ".join(f"{spk}={role}" for spk, role in m.roles.items())
            auditor_str = m.auditor_id or "(none)"
            print(fmt_hdr.format(rec_id, m.review_status, roles_str, auditor_str))
            if m.notes:
                print(f"    - Notes: {m.notes}")
        print("=" * 95)
        return 0

    if args.assign:
        if not args.recording or not args.speaker or not args.role:
            print("[ERROR] --assign requires --recording, --speaker, and --role", file=sys.stderr)
            return 1

        rec_id = args.recording.strip()
        # Allow passing full filename with .wav or .json
        if rec_id.endswith(".json") or rec_id.endswith(".wav"):
            rec_id = Path(rec_id).stem

        # Match prefix or exact stem
        matching = [k for k in role_mgr.registry.mappings.keys() if rec_id in k]
        if not matching:
            print(f"[ERROR] No recording found matching '{rec_id}' in registry.", file=sys.stderr)
            return 1
        target_rec = matching[0]

        try:
            m = role_mgr.assign_role(
                recording_id=target_rec,
                speaker_id=args.speaker,
                role=args.role,
                auditor_id=args.auditor,
                notes=args.notes,
            )
            print(f"[SUCCESS] Assigned role for {target_rec}:")
            print(f"  Speaker: {args.speaker.upper()} -> {args.role.upper()}")
            print(f"  Current Roles: {m.roles}")
            print(f"  Status: {m.review_status}")
            if m.auditor_id:
                print(f"  Auditor: {m.auditor_id}")
            if m.notes:
                print(f"  Notes: {m.notes}")

            # Auto-trigger enrichment when review becomes COMPLETED
            if m.review_status == "COMPLETED":
                _auto_enrich_recording(enricher, target_rec)

            return 0
        except Exception as exc:
            print(f"[ERROR] Failed to assign role: {exc}", file=sys.stderr)
            return 1

    if args.enrich:
        if args.recording:
            # Enrich a single recording
            rec_id = args.recording.strip()
            if rec_id.endswith(".json") or rec_id.endswith(".wav"):
                rec_id = Path(rec_id).stem

            print(f"Enriching single recording: {rec_id}")
            result = enricher.enrich_recording(rec_id)
            if result:
                json_path, txt_path = result
                print(f"  JSON: {json_path}")
                print(f"  TXT:  {txt_path}")
                print("[SUCCESS] Role enrichment complete.")
            else:
                print(f"[ERROR] Could not enrich '{rec_id}'. Original file not found.", file=sys.stderr)
                return 1
        else:
            # Batch enrich all recordings
            print("Enriching all diarized recordings...")
            summary = enricher.enrich_all()
            total = summary.get("total_recordings", 0)
            enriched = summary.get("successfully_enriched", 0)
            completed = summary.get("completed_reviews_applied", 0)
            pending = summary.get("pending_reviews", 0)
            out_dir = summary.get("output_directory", "")
            print(f"\n  Total Recordings:       {total}")
            print(f"  Successfully Enriched:  {enriched}")
            print(f"  Completed Reviews:      {completed}")
            print(f"  Pending Reviews:        {pending}")
            print(f"  Output Directory:       {out_dir}")
            if "summary_report_path" in summary:
                print(f"  Summary Report:         {summary['summary_report_path']}")
            print("\n[SUCCESS] Batch role enrichment complete.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
