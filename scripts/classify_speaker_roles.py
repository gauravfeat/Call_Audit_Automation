#!/usr/bin/env python
"""
M4.5 Automatic Speaker Role Classification CLI.

Uses Google Gemini (gemini-2.5-flash via google-genai SDK) to classify generic speaker IDs
(SPEAKER_00, SPEAKER_01) into business roles ('RM' vs 'CUSTOMER') by analyzing
the full conversational dialogue context.

Predictions are stored separately in:
    output/diarized_transcripts/ai_role_classifications/

Auditor Approval Workflow:
    Use --apply <recording_id> or --apply-all to review and promote verified AI predictions
    into speaker_role_mappings.json, which automatically triggers role enrichment.

Usage:
    # Classify all recordings:
    python scripts/classify_speaker_roles.py --all

    # Classify single recording:
    python scripts/classify_speaker_roles.py --recording <id>

    # Dry-run without saving:
    python scripts/classify_speaker_roles.py --all --dry-run

    # Show AI classification status:
    python scripts/classify_speaker_roles.py --status

    # Approve and transfer an AI prediction into manual registry:
    python scripts/classify_speaker_roles.py --apply <id> --auditor "Gaurav"
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
from app.diarization.role_classifier import RoleClassifier, RoleClassificationResult
from app.diarization.role_mapper import RoleMappingManager
from app.diarization.role_enricher import RoleEnricher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("classify_speaker_roles")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M4.5 AI Speaker Role Classification (RM vs Customer Contextual Classifier)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Classify all diarized recordings")
    group.add_argument("--recording", type=str, help="Classify a specific recording by ID or stem")
    group.add_argument("--status", action="store_true", help="Display summary of AI classifications and manual registry")
    group.add_argument("--apply", type=str, help="Approve and promote AI prediction for recording ID into manual registry")
    group.add_argument("--apply-all", action="store_true", help="Approve and promote all AUTO_CLASSIFIED predictions into registry")

    parser.add_argument("--dry-run", action="store_true", help="Run classification without writing prediction files to disk")
    parser.add_argument("--auditor", type=str, default="Auditor", help="Auditor name for approval workflow")
    parser.add_argument("--force", action="store_true", help="Force overwrite even if manual review is COMPLETED")

    return parser.parse_args()


def display_status(classifier: RoleClassifier, role_mgr: RoleMappingManager) -> int:
    """Display comprehensive status comparing AI classifications and manual registry."""
    ai_dir = classifier.output_dir
    reg = role_mgr.registry

    ai_files = sorted(list(ai_dir.glob("*.json")))
    ai_files = [f for f in ai_files if not f.name.endswith("_summary.json")]

    print("=" * 95)
    print("  M4.5 AI ROLE CLASSIFICATION STATUS")
    print("=" * 95)
    print(f"  AI Predictions Dir:   {ai_dir}")
    print(f"  Manual Registry File: {role_mgr.storage_path}")
    print(f"  Model Configured:     {classifier.model_name}")
    print(f"  Confidence Threshold: {classifier.confidence_threshold:.2f}")
    print("-" * 95)

    if not ai_files:
        print("  [INFO] No AI classification files found yet. Run with '--all' to generate predictions.")
    else:
        fmt = "  {:<45} {:<18} {:<10} {}"
        print(fmt.format("Recording ID", "Status", "Confidence", "Predicted Roles"))
        print("  " + "-" * 91)
        auto_count = 0
        review_count = 0
        error_count = 0
        for f in ai_files:
            try:
                res = RoleClassificationResult.from_json(f)
                roles_str = ", ".join(f"{k}={v}" for k, v in res.predicted_roles.items())
                print(fmt.format(res.recording_id[:45], res.status, f"{res.overall_confidence:.2f}", roles_str))
                if res.status == "AUTO_CLASSIFIED":
                    auto_count += 1
                elif res.status == "NEEDS_MANUAL_REVIEW":
                    review_count += 1
                else:
                    error_count += 1
            except Exception as e:
                print(f"  Error reading {f.name}: {e}")

        print("-" * 95)
        print(f"  Total Predictions: {len(ai_files)} | AUTO_CLASSIFIED: {auto_count} | NEEDS_REVIEW: {review_count} | ERRORS: {error_count}")

    print("=" * 95)
    print(f"  Manual Registry Overview: Total: {reg.total_recordings} | Completed: {reg.completed_reviews} | Pending: {reg.pending_reviews}")
    print("=" * 95)
    return 0


def main() -> int:
    args = parse_args()
    settings = get_settings()
    diarized_dir = settings.diarized_transcripts_dir
    role_mgr = RoleMappingManager(storage_path=diarized_dir / "speaker_role_mappings.json")
    classifier = RoleClassifier(settings=settings)
    enricher = RoleEnricher(settings=settings, role_manager=role_mgr)

    if args.status:
        return display_status(classifier, role_mgr)

    if args.apply:
        rec_id = args.apply.strip()
        if rec_id.endswith(".json") or rec_id.endswith(".wav"):
            rec_id = Path(rec_id).stem

        print(f"[AUDITOR APPROVAL] Approving AI prediction for: {rec_id}")
        success, message = classifier.apply_prediction_to_registry(
            recording_id=rec_id,
            role_manager=role_mgr,
            auditor_id=args.auditor,
            force=args.force,
        )
        if success:
            print(f"[SUCCESS] {message}")
            # Auto-regenerate role-enriched outputs
            print(f"  [ENRICH] Regenerating role-enriched files...")
            enrich_res = enricher.enrich_recording(rec_id)
            if enrich_res:
                j_out, t_out = enrich_res
                print(f"  [ENRICH] JSON: {j_out}")
                print(f"  [ENRICH] TXT:  {t_out}")
            return 0
        else:
            print(f"[REJECTED/SKIPPED] {message}", file=sys.stderr)
            return 1

    if args.apply_all:
        ai_dir = classifier.output_dir
        ai_files = sorted(list(ai_dir.glob("*.json")))
        ai_files = [f for f in ai_files if not f.name.endswith("_summary.json")]

        if not ai_files:
            print("[INFO] No AI classification files to approve. Run '--all' first.")
            return 0

        applied = 0
        skipped = 0
        for f in ai_files:
            try:
                res = RoleClassificationResult.from_json(f)
                if res.status != "AUTO_CLASSIFIED" and not args.force:
                    print(f"  [SKIP] {res.recording_id}: Status is '{res.status}' (not AUTO_CLASSIFIED)")
                    skipped += 1
                    continue

                success, msg = classifier.apply_prediction_to_registry(
                    recording_id=res.recording_id,
                    role_manager=role_mgr,
                    auditor_id=args.auditor,
                    force=args.force,
                )
                if success:
                    print(f"  [APPROVED] {res.recording_id}: {res.predicted_roles}")
                    enricher.enrich_recording(res.recording_id)
                    applied += 1
                else:
                    print(f"  [SKIPPED] {res.recording_id}: {msg}")
                    skipped += 1
            except Exception as exc:
                print(f"  [ERROR] Failed to process {f.name}: {exc}")

        print(f"\n[APPROVAL SUMMARY] Approved & Enriched: {applied} | Skipped/Preserved: {skipped}")
        return 0

    if args.recording:
        rec_id = args.recording.strip()
        if rec_id.endswith(".json") or rec_id.endswith(".wav"):
            rec_id = Path(rec_id).stem

        print("=" * 80)
        print(f"  CLASSIFYING SINGLE RECORDING: {rec_id}")
        print("=" * 80)

        result = classifier.classify_recording(
            recording_id=rec_id,
            diarized_dir=diarized_dir,
            save_output=not args.dry_run,
        )

        if not result:
            print(f"[ERROR] Could not find or classify recording '{rec_id}'.", file=sys.stderr)
            return 1

        print(f"  Status:       {result.status}")
        print(f"  Confidence:   {result.overall_confidence:.2f} (threshold: {classifier.confidence_threshold:.2f})")
        print(f"  Predictions:  {result.predicted_roles}")
        print(f"  Rationale:    {result.rationale}")
        print("\n  Speaker Evidence:")
        for spk in result.speakers:
            print(f"    - {spk.speaker_id} ({spk.predicted_role}, conf={spk.confidence:.2f}):")
            for ind in spk.role_indicators:
                print(f"        * Indicator: {ind}")
            for q in spk.supporting_quotes[:2]:
                print(f"        * Quote: \"{q}\"")

        if result.error_details:
            print(f"  Error Details: {result.error_details}")

        if not args.dry_run:
            print(f"\n[SAVED] Prediction written to: {classifier.output_dir / f'{rec_id}.json'}")
        else:
            print("\n[DRY RUN] Prediction not written to disk.")

        return 0

    if args.all:
        print("=" * 80)
        print("  BATCH AI ROLE CLASSIFICATION")
        print("=" * 80)
        print(f"  Source Directory: {diarized_dir}")
        print(f"  Output Directory: {classifier.output_dir if not args.dry_run else '(dry-run)'}")
        print(f"  Model:            {classifier.model_name}")
        print(f"  Threshold:        {classifier.confidence_threshold:.2f}")
        print("-" * 80)

        summary = classifier.classify_all(diarized_dir=diarized_dir, dry_run=args.dry_run)
        total = summary.get("total_recordings", 0)
        auto_cls = summary.get("auto_classified", 0)
        manual_rev = summary.get("needs_manual_review", 0)
        errs = summary.get("errors", 0)

        print(f"\n  Total Processed:      {total}")
        print(f"  AUTO_CLASSIFIED:      {auto_cls}")
        print(f"  NEEDS_MANUAL_REVIEW:  {manual_rev}")
        print(f"  Errors / Missing Key: {errs}")

        print("\n  Per-Recording Results:")
        for res in summary.get("results", []):
            rid = res.get("recording_id", "unknown")
            st = res.get("status", "UNKNOWN")
            conf = res.get("overall_confidence", 0.0)
            roles = res.get("predicted_roles", {})
            roles_str = ", ".join(f"{k}={v}" for k, v in roles.items())
            print(f"    - {rid}")
            print(f"        Status: {st} (conf: {conf:.2f}) | Roles: {roles_str}")

        if "summary_path" in summary:
            print(f"\n  Summary Report: {summary['summary_path']}")

        print("\n[SUCCESS] AI role classification completed.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
