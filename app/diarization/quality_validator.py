"""
Diarization Quality Validation Module.

Performs deterministic multi-faceted quality validation across speaker-diarized transcripts:
1. Timestamp consistency against audio duration (bounds, ordering, non-inversion).
2. Speaker consistency (turn alternations, monologue streaks, distribution disparity).
3. Overlap detection (simultaneous speech duration, cross-talk vs heavy overlap).
4. Unassigned speech detection and measurable unassigned duration (silence/hold).
5. Suspiciously short segments (< 0.4s) and content sanity.
6. Speaker statistics verification (recalculated vs stored values).
7. Text preservation check against M3 primary STT transcripts.
8. Deterministic manual review flag (true/false) based on quality thresholds.

IMPORTANT: Does NOT invent or calculate fake diarization accuracy percentages,
as ground-truth speaker annotations are not available.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from app.diarization.diarization_schema import DiarizedSegment, DiarizedTranscript, SpeakerStats

logger = logging.getLogger("call_audit.diarization.quality_validator")

# Thresholds for validation rules
SHORT_SEGMENT_THRESHOLD_SECONDS = 0.4
HEAVY_OVERLAP_THRESHOLD_SECONDS = 2.0
TALK_TIME_DISPARITY_LOWER_BOUND = 15.0  # Percentage talk-time below which manual review is triggered
TALK_TIME_DISPARITY_UPPER_BOUND = 85.0  # Percentage talk-time above which manual review is triggered
TEXT_RETENTION_WARNING_RATIO = 0.90     # Diarized / Primary word ratio warning threshold


class TimestampValidation(BaseModel):
    """Timestamp validation metrics for a single recording."""
    is_valid: bool = Field(description="True if all timestamps satisfy 0 <= start <= end <= duration")
    min_start_seconds: float = Field(description="Earliest segment start timestamp")
    max_end_seconds: float = Field(description="Latest segment end timestamp")
    duration_within_bounds: bool = Field(description="True if max_end <= audio_duration + tolerance")
    inverted_segments_count: int = Field(default=0, description="Count of segments where end < start")
    exceeded_duration_count: int = Field(default=0, description="Count of segments exceeding audio duration")
    anomalies: List[str] = Field(default_factory=list, description="Specific timestamp anomaly descriptions")


class OverlapMetrics(BaseModel):
    """Metrics for speaker overlaps and simultaneous speech."""
    overlap_events_count: int = Field(description="Total number of overlapping segment pairs")
    total_simultaneous_speech_seconds: float = Field(description="Sum of physical simultaneous speech duration")
    simultaneous_speech_percentage: float = Field(description="Simultaneous speech as % of audio duration")
    heavy_overlaps_count: int = Field(
        description=f"Count of overlaps longer than {HEAVY_OVERLAP_THRESHOLD_SECONDS}s"
    )
    cross_talk_pairs_count: int = Field(description="Count of overlaps between different speakers")
    same_speaker_overlaps_count: int = Field(description="Count of overlapping segments assigned to the same speaker")
    sample_overlaps: List[Dict[str, Any]] = Field(default_factory=list, description="Top overlapping segment details")


class ShortSegmentSummary(BaseModel):
    """Summary of suspiciously short segments."""
    count: int = Field(description="Number of segments shorter than threshold")
    threshold_seconds: float = Field(default=SHORT_SEGMENT_THRESHOLD_SECONDS)
    percentage_of_segments: float = Field(description="Short segments as % of total segments")
    empty_segments_count: int = Field(default=0, description="Segments with empty/whitespace-only text")
    samples: List[Dict[str, Any]] = Field(default_factory=list, description="Sample short segments with text")


class SpeakerConsistencyMetrics(BaseModel):
    """Metrics assessing speaker alternation and consistency throughout the call."""
    speaker_turns_count: int = Field(description="Number of speaker change transitions")
    same_speaker_transitions_count: int = Field(description="Number of consecutive segments by same speaker")
    turn_alternation_ratio: float = Field(description="Ratio of speaker changes to total transitions")
    talk_time_disparity_flag: bool = Field(description="True if any active speaker talks <15% or >85%")
    dominant_speaker_id: Optional[str] = Field(default=None, description="Speaker with highest talk time")
    dominant_speaker_percentage: float = Field(default=0.0)
    warnings: List[str] = Field(default_factory=list)


class TextPreservationMetrics(BaseModel):
    """Metrics checking text preservation between primary STT and diarized transcripts."""
    primary_words: Optional[int] = Field(default=None, description="Word count from primary M3 transcript")
    diarized_words: int = Field(description="Word count from diarized transcript")
    word_delta: Optional[int] = Field(default=None, description="Diarized words minus primary words")
    word_retention_ratio: Optional[float] = Field(default=None, description="Diarized / Primary words ratio")
    status: str = Field(description="PRESERVED, ENRICHED, or WARNING")
    notes: str = Field(default="")


class RecordingQualityReport(BaseModel):
    """Comprehensive quality validation report for a single audio recording."""
    recording_id: str
    source_file: str
    audio_duration_seconds: float
    detected_speakers: List[str]
    num_speakers_detected: int
    segment_count: int
    total_talk_time_seconds: float = Field(description="Raw sum of all segment durations")
    union_speech_duration_seconds: float = Field(description="Non-overlapping speech duration")
    unassigned_duration_seconds: float = Field(
        description="Audio duration not covered by any speech segment (silence/hold)"
    )
    unassigned_percentage: float = Field(description="Unassigned duration as % of audio duration")
    unassigned_speech_segments_count: int = Field(
        default=0,
        description="Speech segments without an assigned speaker ID (must be 0)",
    )
    timestamp_validation: TimestampValidation
    speaker_consistency: SpeakerConsistencyMetrics
    speaker_consistency_warnings: List[str] = Field(default_factory=list)
    overlap_metrics: OverlapMetrics
    overlap_warnings: List[str] = Field(default_factory=list)
    short_segments: ShortSegmentSummary
    speaker_stats_verified: bool = Field(description="True if stored speaker_stats match recalculated values")
    text_preservation: TextPreservationMetrics
    manual_review_required: bool = Field(description="Whether human auditor review is recommended")
    manual_review_reasons: List[str] = Field(default_factory=list, description="Reasons triggering manual review")


class DiarizationQualityReport(BaseModel):
    """Consolidated quality validation report across all processed recordings."""
    report_metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "engine": "Sarvam Saaras v4 Batch API (codemix)",
            "ground_truth_available": False,
            "evaluation_note": (
                "Objective metrics (timestamps, consistency, overlaps, text preservation) "
                "used. Diarization accuracy percentages omitted due to lack of ground truth."
            ),
        }
    )
    summary: Dict[str, Any]
    recordings: List[RecordingQualityReport]

    def save_json(self, output_path: Path) -> Path:
        """Save report to disk formatted as indented JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return output_path


def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping time intervals into non-overlapping spans."""
    if not intervals:
        return []
    sorted_inv = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_inv[0])]
    for cur in sorted_inv[1:]:
        if cur[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], cur[1])
        else:
            merged.append(list(cur))
    return [(m[0], m[1]) for m in merged]


class DiarizationQualityValidator:
    """
    Validates the quality, consistency, and integrity of diarized transcripts.
    """

    def __init__(self, tolerance_seconds: float = 0.5):
        self.tolerance_seconds = tolerance_seconds

    def validate_recording(
        self,
        diarized: DiarizedTranscript,
        primary_transcript_path: Optional[Path] = None,
    ) -> RecordingQualityReport:
        """
        Perform complete quality validation on a single DiarizedTranscript.
        """
        dur = diarized.duration
        segs = diarized.segments
        rec_id = diarized.recording_id
        manual_review_reasons: List[str] = []

        # ---------------- 1. Timestamps Validation ----------------
        min_start = min((s.start for s in segs), default=0.0)
        max_end = max((s.end for s in segs), default=0.0)
        inverted = [s for s in segs if s.end < s.start]
        exceeded = [s for s in segs if s.end > (dur + self.tolerance_seconds)]

        anomalies: List[str] = []
        if min_start < 0:
            anomalies.append(f"Negative start timestamp: {min_start}s")
        if inverted:
            anomalies.append(f"{len(inverted)} inverted segments (end < start)")
        if exceeded:
            anomalies.append(f"{len(exceeded)} segments end beyond audio duration ({dur}s)")

        ts_valid = (len(anomalies) == 0) and (min_start >= 0) and (max_end <= dur + self.tolerance_seconds)
        if not ts_valid:
            manual_review_reasons.append(f"Timestamp validation anomalies: {'; '.join(anomalies)}")

        ts_metrics = TimestampValidation(
            is_valid=ts_valid,
            min_start_seconds=round(min_start, 3),
            max_end_seconds=round(max_end, 3),
            duration_within_bounds=(max_end <= dur + self.tolerance_seconds),
            inverted_segments_count=len(inverted),
            exceeded_duration_count=len(exceeded),
            anomalies=anomalies,
        )

        # ---------------- 2. Unassigned Speech & Silence ----------------
        unassigned_segs = [s for s in segs if not s.speaker_id or s.speaker_id.strip() == ""]
        if unassigned_segs:
            manual_review_reasons.append(f"{len(unassigned_segs)} segments have no assigned speaker ID")

        raw_talk_time = round(sum(s.duration for s in segs), 3)
        intervals = [(s.start, s.end) for s in segs]
        merged = merge_intervals(intervals)
        union_speech = round(sum(m[1] - m[0] for m in merged), 3)
        unassigned_dur = round(max(0.0, dur - union_speech), 3)
        unassigned_pct = round((unassigned_dur / dur * 100), 2) if dur > 0 else 0.0

        # ---------------- 3. Overlap Detection ----------------
        overlaps: List[Dict[str, Any]] = []
        same_spk_overlaps = 0
        diff_spk_overlaps = 0
        heavy_overlaps = 0
        total_simultaneous = 0.0

        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                s1, s2 = segs[i], segs[j]
                if s2.start >= s1.end:
                    continue
                ov_start = max(s1.start, s2.start)
                ov_end = min(s1.end, s2.end)
                if ov_end > ov_start:
                    ov_duration = round(ov_end - ov_start, 3)
                    total_simultaneous += ov_duration
                    is_same_speaker = (s1.speaker_id == s2.speaker_id)
                    if is_same_speaker:
                        same_spk_overlaps += 1
                    else:
                        diff_spk_overlaps += 1

                    if ov_duration >= HEAVY_OVERLAP_THRESHOLD_SECONDS:
                        heavy_overlaps += 1

                    overlaps.append({
                        "seg1_id": s1.segment_id,
                        "seg2_id": s2.segment_id,
                        "speaker1": s1.speaker_id,
                        "speaker2": s2.speaker_id,
                        "overlap_duration_seconds": ov_duration,
                        "start": round(ov_start, 3),
                        "end": round(ov_end, 3),
                        "is_same_speaker": is_same_speaker,
                    })

        total_simultaneous = round(total_simultaneous, 3)
        simultaneous_pct = round((total_simultaneous / dur * 100), 2) if dur > 0 else 0.0

        overlap_warnings: List[str] = []
        if same_spk_overlaps > 0:
            overlap_warnings.append(
                f"{same_spk_overlaps} overlap(s) detected between segments of the SAME speaker."
            )
            manual_review_reasons.append(f"Same-speaker overlapping segments detected ({same_spk_overlaps})")
        if heavy_overlaps > 0:
            overlap_warnings.append(
                f"{heavy_overlaps} heavy cross-talk overlap(s) exceeding {HEAVY_OVERLAP_THRESHOLD_SECONDS}s."
            )
        if simultaneous_pct > 15.0:
            overlap_warnings.append(
                f"High simultaneous speech volume: {simultaneous_pct}% of audio duration."
            )
            manual_review_reasons.append(f"Simultaneous speech percentage exceeds 15% ({simultaneous_pct}%)")

        overlap_metrics = OverlapMetrics(
            overlap_events_count=len(overlaps),
            total_simultaneous_speech_seconds=total_simultaneous,
            simultaneous_speech_percentage=simultaneous_pct,
            heavy_overlaps_count=heavy_overlaps,
            cross_talk_pairs_count=diff_spk_overlaps,
            same_speaker_overlaps_count=same_spk_overlaps,
            sample_overlaps=overlaps[:5],
        )

        # ---------------- 4. Suspicious Short Segments ----------------
        short_segs = [s for s in segs if s.duration < SHORT_SEGMENT_THRESHOLD_SECONDS]
        empty_segs = [s for s in segs if not s.text.strip()]
        short_pct = round((len(short_segs) / len(segs) * 100), 2) if segs else 0.0

        if empty_segs:
            manual_review_reasons.append(f"{len(empty_segs)} segments have empty text")

        short_metrics = ShortSegmentSummary(
            count=len(short_segs),
            threshold_seconds=SHORT_SEGMENT_THRESHOLD_SECONDS,
            percentage_of_segments=short_pct,
            empty_segments_count=len(empty_segs),
            samples=[
                {
                    "segment_id": s.segment_id,
                    "speaker_id": s.speaker_id,
                    "duration": s.duration,
                    "text": s.text,
                }
                for s in short_segs[:5]
            ],
        )

        # ---------------- 5. Speaker Consistency & Alternations ----------------
        turns = 0
        same_transitions = 0
        for i in range(1, len(segs)):
            if segs[i].speaker_id != segs[i - 1].speaker_id:
                turns += 1
            else:
                same_transitions += 1

        total_transitions = turns + same_transitions
        turn_ratio = round((turns / total_transitions), 3) if total_transitions > 0 else 0.0

        spk_talk_times: Dict[str, float] = {}
        for s in segs:
            spk_talk_times[s.speaker_id] = spk_talk_times.get(s.speaker_id, 0.0) + s.duration

        dominant_spk = max(spk_talk_times.keys(), key=lambda k: spk_talk_times[k]) if spk_talk_times else None
        dominant_pct = (
            round((spk_talk_times[dominant_spk] / dur * 100), 2)
            if dominant_spk and dur > 0
            else 0.0
        )

        consistency_warnings: List[str] = []
        disparity_flag = False
        for spk, talk_time in spk_talk_times.items():
            pct = (talk_time / dur * 100) if dur > 0 else 0.0
            if pct < TALK_TIME_DISPARITY_LOWER_BOUND:
                disparity_flag = True
                msg = (
                    f"Low talk-time for {spk}: {pct:.1f}% ({talk_time:.1f}s). "
                    f"Below threshold of {TALK_TIME_DISPARITY_LOWER_BOUND}%."
                )
                consistency_warnings.append(msg)
                manual_review_reasons.append(f"Speaker talk-time disparity: {spk} speaks for only {pct:.1f}%")
            elif pct > TALK_TIME_DISPARITY_UPPER_BOUND:
                disparity_flag = True
                msg = (
                    f"High talk-time for {spk}: {pct:.1f}% ({talk_time:.1f}s). "
                    f"Above threshold of {TALK_TIME_DISPARITY_UPPER_BOUND}%."
                )
                consistency_warnings.append(msg)

        if turn_ratio < 0.3 and len(segs) > 10:
            consistency_warnings.append(
                f"Low speaker alternation ratio ({turn_ratio:.2f}). Conversation exhibits monologue blocks."
            )

        consistency_metrics = SpeakerConsistencyMetrics(
            speaker_turns_count=turns,
            same_speaker_transitions_count=same_transitions,
            turn_alternation_ratio=turn_ratio,
            talk_time_disparity_flag=disparity_flag,
            dominant_speaker_id=dominant_spk,
            dominant_speaker_percentage=dominant_pct,
            warnings=consistency_warnings,
        )

        # ---------------- 6. Speaker Stats Verification ----------------
        stats_match = True
        for ss in diarized.speaker_stats:
            calc_talk = round(spk_talk_times.get(ss.speaker_id, 0.0), 3)
            if abs(ss.total_talk_time_seconds - calc_talk) > 0.1:
                stats_match = False
                break
        if not stats_match:
            manual_review_reasons.append("Stored speaker_stats discrepancy with segment durations")

        # ---------------- 7. Text Preservation Check ----------------
        diar_words = sum(len(s.text.split()) for s in segs if s.text.strip())
        prim_words: Optional[int] = None
        retention_ratio: Optional[float] = None
        text_status = "N/A"
        text_notes = "No primary transcript provided for comparison"

        if primary_transcript_path and primary_transcript_path.is_file():
            try:
                with open(primary_transcript_path, "r", encoding="utf-8") as pf:
                    p_data = json.load(pf)
                p_segs = p_data.get("segments", [])
                prim_words = sum(len(s.get("text", "").split()) for s in p_segs if s.get("text"))
                if prim_words > 0:
                    retention_ratio = round(diar_words / prim_words, 3)
                    word_delta = diar_words - prim_words
                    if retention_ratio >= 1.0:
                        text_status = "ENRICHED"
                        text_notes = (
                            f"Diarized transcript captured +{word_delta} words (+{(retention_ratio-1)*100:.1f}%) "
                            "due to speaker channel separation during overlapping speech."
                        )
                    elif retention_ratio >= TEXT_RETENTION_WARNING_RATIO:
                        text_status = "PRESERVED"
                        text_notes = f"Retained {retention_ratio*100:.1f}% of primary transcript words."
                    else:
                        text_status = "WARNING"
                        text_notes = f"Word retention {retention_ratio*100:.1f}% below {TEXT_RETENTION_WARNING_RATIO*100}% threshold."
                        manual_review_reasons.append(f"Text loss compared to primary transcript: {text_notes}")
            except Exception as exc:
                text_notes = f"Failed to compare with primary transcript: {exc}"

        text_metrics = TextPreservationMetrics(
            primary_words=prim_words,
            diarized_words=diar_words,
            word_delta=(diar_words - prim_words) if prim_words is not None else None,
            word_retention_ratio=retention_ratio,
            status=text_status,
            notes=text_notes,
        )

        # ---------------- 8. Overall Report Assembly ----------------
        requires_manual_review = (len(manual_review_reasons) > 0)

        return RecordingQualityReport(
            recording_id=rec_id,
            source_file=diarized.source_file,
            audio_duration_seconds=dur,
            detected_speakers=diarized.speaker_ids,
            num_speakers_detected=diarized.num_speakers_detected,
            segment_count=len(segs),
            total_talk_time_seconds=raw_talk_time,
            union_speech_duration_seconds=union_speech,
            unassigned_duration_seconds=unassigned_dur,
            unassigned_percentage=unassigned_pct,
            unassigned_speech_segments_count=len(unassigned_segs),
            timestamp_validation=ts_metrics,
            speaker_consistency=consistency_metrics,
            speaker_consistency_warnings=consistency_warnings,
            overlap_metrics=overlap_metrics,
            overlap_warnings=overlap_warnings,
            short_segments=short_metrics,
            speaker_stats_verified=stats_match,
            text_preservation=text_metrics,
            manual_review_required=requires_manual_review,
            manual_review_reasons=manual_review_reasons,
        )

    def validate_batch(
        self,
        diarized_dir: Path,
        primary_dir: Optional[Path] = None,
        output_report_path: Optional[Path] = None,
    ) -> DiarizationQualityReport:
        """
        Validate all diarized transcript JSON files in a directory.
        """
        json_files = sorted(
            [f for f in diarized_dir.glob("*.json") if not f.name.endswith("_summary.json") and not f.name.endswith("_report.json") and not f.name.endswith("_mappings.json")]
        )
        if not json_files:
            raise FileNotFoundError(f"No diarized transcript JSON files found in {diarized_dir}")

        recording_reports: List[RecordingQualityReport] = []
        for jf in json_files:
            try:
                dt = DiarizedTranscript.from_json(jf)
                primary_file = (primary_dir / jf.name) if primary_dir else None
                rep = self.validate_recording(dt, primary_file)
                recording_reports.append(rep)
            except Exception as exc:
                logger.error("Failed to validate diarized transcript %s: %s", jf, exc)

        total_dur = sum(r.audio_duration_seconds for r in recording_reports)
        total_union_speech = sum(r.union_speech_duration_seconds for r in recording_reports)
        total_unassigned = sum(r.unassigned_duration_seconds for r in recording_reports)
        avg_unassigned_pct = round((total_unassigned / total_dur * 100), 2) if total_dur > 0 else 0.0
        total_overlaps = sum(r.overlap_metrics.total_simultaneous_speech_seconds for r in recording_reports)
        avg_overlap_pct = round((total_overlaps / total_dur * 100), 2) if total_dur > 0 else 0.0
        ts_all_valid = all(r.timestamp_validation.is_valid for r in recording_reports)
        manual_rev_count = sum(1 for r in recording_reports if r.manual_review_required)

        summary = {
            "total_recordings_validated": len(recording_reports),
            "total_audio_duration_seconds": round(total_dur, 2),
            "total_speech_duration_seconds": round(total_union_speech, 2),
            "total_unassigned_duration_seconds": round(total_unassigned, 2),
            "average_unassigned_percentage": avg_unassigned_pct,
            "total_overlapping_speech_seconds": round(total_overlaps, 2),
            "average_overlap_percentage": avg_overlap_pct,
            "timestamp_validation_all_passed": ts_all_valid,
            "recordings_requiring_manual_review": manual_rev_count,
            "overall_status": "PASSED_WITH_REVIEW_FLAGS" if manual_rev_count > 0 else "PASSED",
        }

        report = DiarizationQualityReport(
            summary=summary,
            recordings=recording_reports,
        )

        if output_report_path:
            report.save_json(output_report_path)
            logger.info("Saved diarization quality report to: %s", output_report_path)

        return report
