"""
Transcript and Speaker Diarization Alignment Engine.

Aligns speech-to-text transcript segments (from Whisper Large-v3) with
speaker diarization intervals (from pyannote Community-1) based on temporal overlap.
Outputs a fully validated canonical DiarizedTranscript schema.
"""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from app.core.config import Settings, get_settings
from app.diarization.diarization_schema import (
    DiarizedSegment,
    DiarizedTranscript,
    SpeakerStats,
)
from app.diarization.pyannote_diarizer import SpeakerTurn
from app.transcription.transcript_schema import Transcript, TranscriptSegment

logger = logging.getLogger("call_audit.diarization.alignment")


class AlignmentError(Exception):
    """Raised when transcript alignment fails."""
    pass


def calculate_overlap(
    seg_start: float,
    seg_end: float,
    turn_start: float,
    turn_end: float,
) -> float:
    """Calculate the overlap duration between two time intervals in seconds."""
    overlap = min(seg_end, turn_end) - max(seg_start, turn_start)
    return max(0.0, overlap)


def compute_speaker_overlap_map(
    segment: TranscriptSegment,
    speaker_turns: List[SpeakerTurn],
) -> Dict[str, float]:
    """
    Compute total overlapping seconds for each speaker with a given transcript segment.
    """
    overlaps: Dict[str, float] = {}
    for turn in speaker_turns:
        ov = calculate_overlap(segment.start, segment.end, turn.start, turn.end)
        if ov > 0.0:
            overlaps[turn.speaker_id] = overlaps.get(turn.speaker_id, 0.0) + ov
    return overlaps


def find_nearest_speaker(
    segment: TranscriptSegment,
    speaker_turns: List[SpeakerTurn],
    max_distance_seconds: float = 2.0,
) -> Optional[str]:
    """
    Find the speaker turn closest to the segment when no direct temporal overlap exists.
    """
    if not speaker_turns:
        return None

    seg_mid = (segment.start + segment.end) / 2.0
    best_turn: Optional[SpeakerTurn] = None
    min_dist = float("inf")

    for turn in speaker_turns:
        turn_mid = (turn.start + turn.end) / 2.0
        dist = abs(seg_mid - turn_mid)
        if dist < min_dist:
            min_dist = dist
            best_turn = turn

    if best_turn is not None and min_dist <= max_distance_seconds:
        return best_turn.speaker_id
    return None


def calculate_speaker_stats(
    segments: List[DiarizedSegment],
    total_duration: float,
) -> List[SpeakerStats]:
    """
    Calculate talk-time and segment statistics for each unique speaker.
    """
    speakers = sorted(list(set(s.speaker_id for s in segments)))
    stats: List[SpeakerStats] = []

    for spk_id in speakers:
        spk_segs = [s for s in segments if s.speaker_id == spk_id]
        if not spk_segs:
            continue

        tot_segments = len(spk_segs)
        tot_words = sum(len(s.text.split()) for s in spk_segs if s.text.strip())
        tot_talk_time = round(sum(s.duration for s in spk_segs), 3)
        pct = (
            round((tot_talk_time / total_duration) * 100.0, 2)
            if total_duration > 0
            else 0.0
        )
        first_start = min(s.start for s in spk_segs)
        last_end = max(s.end for s in spk_segs)

        stats.append(
            SpeakerStats(
                speaker_id=spk_id,
                role="UNKNOWN",
                total_segments=tot_segments,
                total_words=tot_words,
                total_talk_time_seconds=tot_talk_time,
                talk_time_percentage=pct,
                first_segment_start=first_start,
                last_segment_end=last_end,
            )
        )

    return stats


def align_transcript_and_diarization(
    transcript: Transcript,
    speaker_turns: List[SpeakerTurn],
    recording_id: Optional[str] = None,
    min_overlap_ratio: float = 0.25,
    processing_time_seconds: float = 0.0,
    model_name: str = "pyannote/speaker-diarization-community-1",
    settings: Optional[Settings] = None,
) -> DiarizedTranscript:
    """
    Align Whisper transcript segments with pyannote speaker intervals.

    Args:
        transcript: Whisper Transcript schema object.
        speaker_turns: List of pyannote SpeakerTurn intervals.
        recording_id: Recording ID identifier.
        min_overlap_ratio: Minimum ratio of segment duration that must overlap with a speaker.
        processing_time_seconds: Total diarization + alignment time.
        model_name: Identifier of diarization model.
        settings: Application settings.

    Returns:
        Canonical DiarizedTranscript object ready for M4 role enrichment and M4.5 classification.
    """
    rec_id = recording_id or transcript.recording_id
    duration = transcript.duration
    warnings: List[str] = []

    # Case 1: Check if transcript has no timestamps / only 1 unsegmented chunk
    has_meaningful_timestamps = (
        len(transcript.segments) > 1
        or (len(transcript.segments) == 1 and transcript.segments[0].end > transcript.segments[0].start)
    )

    if not has_meaningful_timestamps or not transcript.segments:
        logger.warning(
            "Whisper transcript for %s has no segment timestamps. Cannot reliably align with speaker turns.",
            rec_id,
        )
        warnings.append(
            "Whisper transcript contains unsegmented text without fine timestamps. "
            "Segments marked as UNKNOWN for manual review."
        )

    # Case 2: No speaker turns detected by pyannote
    if not speaker_turns:
        logger.warning("No speaker turns provided for %s. Attributing all segments to SPEAKER_00.", rec_id)
        warnings.append("No speaker turns detected. Defaulted to SPEAKER_00.")
        aligned_segs = [
            DiarizedSegment(
                segment_id=idx,
                speaker_id="SPEAKER_00",
                speaker_role="UNKNOWN",
                start=seg.start,
                end=seg.end,
                text=seg.text,
            )
            for idx, seg in enumerate(transcript.segments)
        ]
        stats = calculate_speaker_stats(aligned_segs, duration)
        return DiarizedTranscript(
            recording_id=rec_id,
            source_file=transcript.source_file,
            duration=duration,
            num_speakers_detected=1,
            num_speakers_requested=None,
            model=model_name,
            mode=transcript.transcription_mode or "codemix",
            processing_time_seconds=processing_time_seconds,
            api_cost_inr=0.0,
            language=transcript.language,
            segments=aligned_segs,
            speaker_stats=stats,
            full_transcript=transcript.full_text,
            provider="pyannote",
            status="SUCCESS",
            warnings=warnings,
        )

    # Standard Case: Align each transcript segment to dominant speaker turn
    aligned_segments: List[DiarizedSegment] = []
    unique_speakers = set()

    for idx, seg in enumerate(transcript.segments):
        seg_dur = max(seg.end - seg.start, 0.001)
        overlap_map = compute_speaker_overlap_map(seg, speaker_turns)

        assigned_speaker = "UNKNOWN"
        if overlap_map:
            # Find speaker with maximum overlap
            best_speaker = max(overlap_map, key=overlap_map.get)
            best_overlap = overlap_map[best_speaker]
            overlap_ratio = best_overlap / seg_dur

            # Check if ambiguous or overlapping between multiple speakers
            speakers_with_overlap = [spk for spk, dur in overlap_map.items() if dur >= 0.2 * seg_dur]
            if len(speakers_with_overlap) > 1:
                logger.debug(
                    "Segment %d (%s) has multi-speaker overlap across speakers: %s",
                    idx, seg.text[:30], overlap_map
                )
                warnings.append(
                    f"Segment {idx} ({seg.start:.1f}s-{seg.end:.1f}s) has multi-speaker overlap; "
                    f"attributed to majority speaker {best_speaker}."
                )

            if overlap_ratio >= min_overlap_ratio or best_overlap >= 0.3:
                assigned_speaker = best_speaker
            else:
                # Low overlap ratio; try nearest turn fallback
                nearest = find_nearest_speaker(seg, speaker_turns, max_distance_seconds=1.5)
                assigned_speaker = nearest or best_speaker
        else:
            # No direct temporal overlap, check nearest speaker turn
            nearest = find_nearest_speaker(seg, speaker_turns, max_distance_seconds=2.0)
            if nearest:
                assigned_speaker = nearest
            else:
                assigned_speaker = "UNKNOWN"
                warnings.append(f"Segment {idx} ({seg.start:.1f}s-{seg.end:.1f}s) is outside speaker turns; marked UNKNOWN.")

        unique_speakers.add(assigned_speaker)
        aligned_segments.append(
            DiarizedSegment(
                segment_id=idx,
                speaker_id=assigned_speaker,
                speaker_role="UNKNOWN",
                start=seg.start,
                end=seg.end,
                text=seg.text,
            )
        )

    # Filter out UNKNOWN from count of detected speakers if known speakers exist
    detected_count = len([s for s in unique_speakers if s != "UNKNOWN"]) or 1
    stats = calculate_speaker_stats(aligned_segments, duration)

    diarized_transcript = DiarizedTranscript(
        recording_id=rec_id,
        source_file=transcript.source_file,
        duration=duration,
        num_speakers_detected=detected_count,
        num_speakers_requested=None,
        model=model_name,
        mode=transcript.transcription_mode or "codemix",
        processing_time_seconds=processing_time_seconds,
        api_cost_inr=0.0,
        language=transcript.language,
        segments=aligned_segments,
        speaker_stats=stats,
        full_transcript=transcript.full_text,
        provider="pyannote",
        status="SUCCESS",
        warnings=warnings,
    )

    return diarized_transcript


def save_onprem_diarized_transcript(
    diarized_transcript: DiarizedTranscript,
    output_dir: Optional[Path] = None,
    settings: Optional[Settings] = None,
) -> Tuple[Path, Path]:
    """
    Save DiarizedTranscript to output/diarized_transcripts/onprem_pyannote/<id>.json and .txt.

    Returns:
        Tuple of (json_path, txt_path).
    """
    settings = settings or get_settings()
    out_dir = output_dir or settings.onprem_pyannote_diarized_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{diarized_transcript.recording_id}.json"
    txt_path = out_dir / f"{diarized_transcript.recording_id}.txt"

    # Write JSON atomically
    tmp_json = json_path.with_suffix(".tmp")
    with open(tmp_json, "w", encoding="utf-8") as f:
        f.write(diarized_transcript.model_dump_json(indent=2))
    tmp_json.replace(json_path)

    # Write human-readable TXT atomically
    lines = [
        f"RECORDING: {diarized_transcript.recording_id}",
        f"SOURCE FILE: {diarized_transcript.source_file}",
        f"DURATION: {diarized_transcript.duration:.1f}s",
        f"SPEAKERS DETECTED: {diarized_transcript.num_speakers_detected}",
        f"MODEL: {diarized_transcript.model}",
        f"PROCESSING TIME: {diarized_transcript.processing_time_seconds:.2f}s",
        "=" * 60,
        "",
    ]

    for seg in diarized_transcript.segments:
        start_m = int(seg.start // 60)
        start_s = seg.start % 60
        end_m = int(seg.end // 60)
        end_s = seg.end % 60
        time_str = f"[{start_m:02d}:{start_s:05.2f} -> {end_m:02d}:{end_s:05.2f}]"
        lines.append(f"{time_str} {seg.speaker_id}: {seg.text}")

    tmp_txt = txt_path.with_suffix(".tmp")
    with open(tmp_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    tmp_txt.replace(txt_path)

    logger.info("Saved aligned diarized transcript to %s and %s", json_path, txt_path)
    return json_path, txt_path
