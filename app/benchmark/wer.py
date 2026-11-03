"""
Word Error Rate (WER) and Character Error Rate (CER) Evaluation Module.

Provides normalized error rate calculation between ground truth / reference transcripts
and model hypotheses using jiwer. Supports Devanagari and Latin script normalization.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import jiwer
except ImportError:
    jiwer = None


def normalize_text_for_evaluation(text: str) -> str:
    """
    Normalize text for fair WER/CER comparison across different models.

    Steps:
    1. Lowercase Latin characters.
    2. Replace punctuation (including Hindi Purna Viram '।') with space.
    3. Remove multiple spaces and trim whitespace.
    """
    if not text:
        return ""

    # Convert to lowercase
    norm = text.lower()

    # Replace punctuation, symbols, and Hindi danda (।) with spaces
    norm = re.sub(r"[।\.,\?!;:\"'„“”—\-_()\[\]{}<>/\\|~`@#$%^&*+=]", " ", norm)

    # Collapse consecutive whitespace characters
    norm = re.sub(r"\s+", " ", norm).strip()

    return norm


def calculate_wer_cer(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute Word Error Rate (WER) and Character Error Rate (CER).

    Args:
        reference: Ground-truth reference transcript text.
        hypothesis: Model-generated hypothesis transcript text.

    Returns:
        Dict with 'wer', 'cer', 'insertions', 'deletions', 'substitutions', 'word_count'.
    """
    norm_ref = normalize_text_for_evaluation(reference)
    norm_hyp = normalize_text_for_evaluation(hypothesis)

    if not norm_ref:
        return {
            "wer": 1.0 if norm_hyp else 0.0,
            "cer": 1.0 if norm_hyp else 0.0,
            "insertions": len(norm_hyp.split()),
            "deletions": 0,
            "substitutions": 0,
            "reference_words": 0,
            "hypothesis_words": len(norm_hyp.split()),
        }

    if not jiwer:
        # Fallback if jiwer is missing
        ref_words = norm_ref.split()
        hyp_words = norm_hyp.split()
        return {
            "wer": float(ref_words != hyp_words),
            "cer": float(norm_ref != norm_hyp),
            "reference_words": len(ref_words),
            "hypothesis_words": len(hyp_words),
            "warning": "jiwer library not installed; simple equality fallback used",
        }

    wer_score = float(jiwer.wer(norm_ref, norm_hyp))
    cer_score = float(jiwer.cer(norm_ref, norm_hyp))

    output = jiwer.process_words(norm_ref, norm_hyp)

    return {
        "wer": round(wer_score, 4),
        "cer": round(cer_score, 4),
        "substitutions": output.substitutions,
        "deletions": output.deletions,
        "insertions": output.insertions,
        "hits": output.hits,
        "reference_words": len(norm_ref.split()),
        "hypothesis_words": len(norm_hyp.split()),
    }


def load_reference_transcript(reference_path: Path) -> Optional[str]:
    """
    Load ground-truth reference transcript from a .txt or .json file.
    """
    if not reference_path.exists():
        return None

    if reference_path.suffix.lower() == ".json":
        with open(reference_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Support both direct string or dictionary with 'transcript' / 'text'
            if isinstance(data, dict):
                return data.get("transcript") or data.get("text") or data.get("reference")
            return str(data)
    else:
        with open(reference_path, "r", encoding="utf-8") as f:
            return f.read().strip()


def save_reference_template(recording_id: str, sample_text: str, output_dir: Path) -> Path:
    """
    Create a reference transcript template for manual human verification.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{recording_id}_reference.txt"
    if not target.exists():
        with open(target, "w", encoding="utf-8") as f:
            f.write(f"# Reference transcript for {recording_id}\n")
            f.write("# Replace or edit the text below with verified human-audited ground truth:\n\n")
            f.write(sample_text)
    return target
