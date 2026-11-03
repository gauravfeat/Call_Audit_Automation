"""
Automatic Repetition and Hallucination Loop Detection Module.

Analyzes ASR transcript text for unnatural repetition loops, stuck tokens,
and consecutive duplicate phrases without modifying the underlying transcript.
"""

import re
from collections import Counter
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class RepetitionMetrics(BaseModel):
    """
    Structured metrics detailing detected repetition loops in an ASR transcript.
    """

    is_repetitive: bool = Field(description="True if abnormal or suspicious repetition was detected")
    unique_words_count: int = Field(description="Count of distinct words in transcript")
    total_words_count: int = Field(description="Total word count in transcript")
    unique_to_total_ratio: float = Field(description="Lexical diversity ratio (unique/total words)")
    longest_consecutive_word_streak: int = Field(description="Maximum consecutive identical word repetitions")
    flagged_repeated_words: List[str] = Field(default_factory=list, description="Words appearing consecutively >= 3 times")
    repeated_phrases: List[Dict[str, Any]] = Field(default_factory=list, description="Repeated n-grams (>2 occurrences)")
    warnings: List[str] = Field(default_factory=list, description="Descriptive hallucination warnings")


def detect_repetition(text: str) -> RepetitionMetrics:
    """
    Analyze text for suspicious repeated words and hallucinated decoding loops.

    Args:
        text: Raw transcript text to analyze.

    Returns:
        RepetitionMetrics detailing detected repetition patterns and warnings.
    """
    cleaned = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
    words = [w.strip() for w in cleaned.split() if w.strip()]

    total_words = len(words)
    if total_words == 0:
        return RepetitionMetrics(
            is_repetitive=False,
            unique_words_count=0,
            total_words_count=0,
            unique_to_total_ratio=1.0,
            longest_consecutive_word_streak=0,
            warnings=[],
        )

    unique_words = set(words)
    diversity_ratio = round(len(unique_words) / total_words, 4)

    # 1. Consecutive identical word streaks (e.g. "अपका अपका अपका...")
    max_streak = 1
    current_streak = 1
    flagged_words: List[str] = []

    for i in range(1, total_words):
        if words[i].lower() == words[i - 1].lower():
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
            if current_streak == 3:
                flagged_words.append(words[i])
        else:
            current_streak = 1

    # 2. Repeated n-gram phrases (e.g. 3-grams repeated multiple times)
    repeated_phrases: List[Dict[str, Any]] = []
    for n in [2, 3, 4]:
        if total_words >= n * 2:
            ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
            counts = Counter(ngrams)
            for phrase, count in counts.items():
                if count >= 4 and len(phrase) > 4:
                    repeated_phrases.append({"phrase": phrase, "count": count, "n": n})

    # Deduplicate phrases
    seen_phrases = set()
    unique_repeated_phrases = []
    for item in repeated_phrases:
        if item["phrase"] not in seen_phrases:
            seen_phrases.add(item["phrase"])
            unique_repeated_phrases.append(item)

    # 3. Construct warnings
    warnings: List[str] = []
    is_repetitive = False

    if max_streak >= 3:
        is_repetitive = True
        warnings.append(
            f"Detected consecutive word repetition (streak of {max_streak} identical words; e.g. '{words[0]}')."
        )

    if unique_repeated_phrases:
        is_repetitive = True
        top_phrase = max(unique_repeated_phrases, key=lambda p: p["count"])
        warnings.append(
            f"Detected repeating phrase loop: \"{top_phrase['phrase']}\" repeated {top_phrase['count']} times."
        )

    if total_words > 50 and diversity_ratio < 0.25:
        is_repetitive = True
        warnings.append(
            f"Abnormally low vocabulary diversity ({diversity_ratio:.2f}); transcript likely trapped in a loop."
        )

    return RepetitionMetrics(
        is_repetitive=is_repetitive,
        unique_words_count=len(unique_words),
        total_words_count=total_words,
        unique_to_total_ratio=diversity_ratio,
        longest_consecutive_word_streak=max_streak,
        flagged_repeated_words=list(set(flagged_words)),
        repeated_phrases=unique_repeated_phrases[:10],
        warnings=warnings,
    )
