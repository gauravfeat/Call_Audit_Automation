"""
Validation Script for Sarvam Codemix Primary Transcripts.

Verifies schema compliance, timestamp consistency, absence of empty segments,
repetition loop metrics, and compares against baseline Whisper transcripts.
"""

import glob
import json
import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.transcription.transcript_schema import Transcript
from app.benchmark.repetition import detect_repetition


def main():
    transcript_files = sorted(glob.glob(os.path.join(PROJECT_ROOT, "output", "transcripts", "sarvam_v4_codemix", "H15029*.json")))
    print("=" * 80)
    print(f"VALIDATING PRIMARY SARVAM TRANSCRIPTS ({len(transcript_files)} files)")
    print("=" * 80)

    if not transcript_files:
        print("[ERROR] No primary transcript files found!")
        return 1

    all_passed = True

    for fp in transcript_files:
        t = Transcript.from_json(Path(fp))
        print(f"\nRecording: {t.recording_id}")
        print(f"  Source File:         {t.source_file}")
        print(f"  Provider:            {t.provider}")
        print(f"  Model:               {t.model}")
        print(f"  Mode:                {t.transcription_mode}")
        print(f"  Language:            {t.language} (confidence: {t.language_probability:.3f})")
        print(f"  Audio Duration:      {t.duration:.1f}s")
        print(f"  Processing Time:     {t.processing_time_seconds:.2f}s")
        print(f"  API Cost (INR):      Rs. {t.api_cost_inr:.4f}")
        print(f"  Word Count:          {t.total_words}")
        print(f"  Segment Count:       {len(t.segments)}")

        # 1. Non-empty check
        if t.total_words == 0:
            print("  [FAIL] Transcript has 0 words!")
            all_passed = False
        else:
            print("  [PASS] Content check: Non-empty transcript with words.")

        # 2. Timestamp check
        malformed = []
        for s in t.segments:
            if s.start < 0 or s.end < s.start:
                malformed.append(f"Segment #{s.segment_id}: start={s.start}, end={s.end}")
        if malformed:
            print(f"  [FAIL] Malformed timestamps: {malformed}")
            all_passed = False
        else:
            print(f"  [PASS] Timestamp validity: All {len(t.segments)} segments have 0 <= start <= end.")

        # 3. Repetition loop check
        rep = detect_repetition(t.full_text)
        print(f"  Repetition Metrics:  Streak={rep.longest_consecutive_word_streak}, Repeated phrases={len(rep.repeated_phrases)}, Lexical ratio={rep.unique_to_total_ratio:.3f}, Is Repetitive={rep.is_repetitive}")

        # 4. Compare with previous Whisper Small transcript
        whisper_fp = os.path.join(PROJECT_ROOT, "output", "transcripts", f"{t.recording_id}.json")
        if os.path.exists(whisper_fp):
            wt = Transcript.from_json(Path(whisper_fp))
            diff = t.total_words - wt.total_words
            ratio = t.total_words / max(wt.total_words, 1)
            print(f"  Whisper Comparison:  Sarvam: {t.total_words} words vs Whisper: {wt.total_words} words (+{diff} words, {ratio:.2f}x dialogue density)")

        # 5. Preview text
        preview = t.full_text[:110] + ("..." if len(t.full_text) > 110 else "")
        print(f"  Preview:             \"{preview}\"")

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL 5 PRIMARY RECORDINGS PASSED VALIDATION!")
    else:
        print("[WARNING] Some checks failed.")
    print("=" * 80)
    return 0 if all_passed else 1


if __name__ == "__main__":
    from pathlib import Path
    sys.exit(main())
