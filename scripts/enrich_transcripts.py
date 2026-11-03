#!/usr/bin/env python
"""
M4 Role-Enriched Transcript Generation CLI.

Combines original Sarvam diarized transcripts with manually verified speaker-role
mappings from speaker_role_mappings.json to generate separate role-enriched JSON and TXT files.

Outputs are written exclusively to:
    output/diarized_transcripts/role_enriched/

Safety:
- NEVER modifies original diarized JSON or TXT files.
- NEVER modifies speaker_role_mappings.json during enrichment.
- Does not call Sarvam API or perform new transcription/diarization.
- Roles are applied ONLY when review_status == 'COMPLETED'. Otherwise 'UNKNOWN'.

Usage:
    # Process all available recordings:
    python scripts/enrich_transcripts.py

    # Process a specific recording:
    python scripts/enrich_transcripts.py --recording H15029__Thane_Virtual_SL__118__198804820151025__8087967360__2025-08-22_11-52-11

    # Show enrichment status summary:
    python scripts/enrich_transcripts.py --status
"""

import argparse
import json
import logging
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
from app.diarization.role_enricher import RoleEnricher
from app.diarization.role_mapper import RoleMappingManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("enrich_transcripts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M4 Role-Enriched Transcript Generator: Combine Sarvam diarized transcripts with verified role mappings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--recording",
        type=str,
        default=None,
        help="Specific recording ID or filename to enrich (processes all if omitted)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Display current enrichment and review status summary",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-generation of enriched outputs",
    )
    return parser.parse_args()


def display_status(enricher: RoleEnricher, role_mgr: RoleMappingManager) -> int:
    """Display current status of role mappings and enriched outputs."""
    reg = role_mgr.registry
    out_dir = enricher.output_dir

    print("=" * 80)
    print("  ROLE ENRICHMENT STATUS SUMMARY")
    print("=" * 80)
    print(f"  Role Registry:       {role_mgr.storage_path}")
    print(f"  Enriched Output Dir: {out_dir}")
    print(f"  Total in Registry:   {reg.total_recordings}")
    print(f"  Completed Reviews:   {reg.completed_reviews}")
    print(f"  Pending Reviews:     {reg.pending_reviews}")
    print("-" * 80)

    # Check existing enriched files
    json_count = len(list(out_dir.glob("*.json"))) - (1 if (out_dir / "role_enrichment_summary.json").exists() else 0)
    txt_count = len(list(out_dir.glob("*.txt")))
    print(f"  Enriched JSON Files: {json_count}")
    print(f"  Enriched TXT Files:  {txt_count}")
    print("=" * 80)
    return 0


def main() -> int:
    args = parse_args()
    settings = get_settings()
    diarized_dir = settings.diarized_transcripts_dir
    role_mgr = RoleMappingManager(storage_path=diarized_dir / "speaker_role_mappings.json")
    enricher = RoleEnricher(settings=settings, role_manager=role_mgr)

    if args.status:
        return display_status(enricher, role_mgr)

    if args.recording:
        rec_id = args.recording.strip()
        if rec_id.endswith(".json") or rec_id.endswith(".wav"):
            rec_id = Path(rec_id).stem

        print("=" * 80)
        print(f"  ENRICHING SINGLE RECORDING: {rec_id}")
        print("=" * 80)

        result = enricher.enrich_recording(rec_id)
        if not result:
            print(f"[ERROR] Could not find or enrich recording '{rec_id}'.", file=sys.stderr)
            return 1

        json_path, txt_path = result
        roles, status = enricher.get_verified_roles(rec_id)
        print(f"  Status:       {status}")
        print(f"  Roles:        {roles}")
        print(f"  Enriched JSON: {json_path}")
        print(f"  Enriched TXT:  {txt_path}")
        print("[SUCCESS] Recording enriched successfully.")
        return 0

    # Batch mode: process all
    print("=" * 80)
    print("  BATCH ROLE ENRICHMENT")
    print("=" * 80)
    print(f"  Source Directory: {diarized_dir}")
    print(f"  Output Directory: {enricher.output_dir}")
    print(f"  Registry:         {role_mgr.storage_path}")
    print("-" * 80)

    summary = enricher.enrich_all()
    total = summary.get("total_recordings", 0)
    enriched = summary.get("successfully_enriched", 0)
    completed = summary.get("completed_reviews_applied", 0)
    pending = summary.get("pending_reviews", 0)

    print(f"\n  Total Processed:           {total}")
    print(f"  Successfully Enriched:     {enriched}")
    print(f"  Completed Reviews Applied: {completed}")
    print(f"  Pending / Unknown Roles:   {pending}")

    print("\n  Per-Recording Results:")
    for res in summary.get("results", []):
        rec_id = res.get("recording_id", "unknown")
        st = res.get("review_status", "UNKNOWN")
        roles = res.get("roles_applied", {})
        if isinstance(roles, dict):
            roles_str = ", ".join(f"{k}={v}" for k, v in roles.items())
        else:
            roles_str = str(roles)
        print(f"    - {rec_id}")
        print(f"        Status: {st} | Roles: {roles_str}")

    if "summary_report_path" in summary:
        print(f"\n  Summary Report: {summary['summary_report_path']}")

    print("\n[SUCCESS] All recordings processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
