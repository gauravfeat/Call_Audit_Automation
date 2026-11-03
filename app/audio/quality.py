"""
Audio Quality Analysis Module.

Performs non-destructive quality assessments on audio recordings using FFmpeg filters:
- Average RMS volume (mean volume in dBFS)
- Peak amplitude (maximum volume in dBFS)
- Clipping detection (samples reaching 0 dBFS ceiling)
- Silence detection and silence percentage calculation
- Detection of unusually low-volume recordings or extended dead air

Quality issues are flagged as actionable warnings rather than hard failures.
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from app.audio.exceptions import AudioQualityAnalysisError
from app.audio.ffmpeg_utils import get_ffmpeg_path

logger = logging.getLogger("call_audit.audio.quality")

# Quality threshold constants
SILENCE_NOISE_THRESHOLD_DB: float = -35.0  # dB threshold below which is classified as silence
SILENCE_MIN_DURATION_SEC: float = 1.5      # Minimum consecutive seconds to register a silence interval
LOW_VOLUME_MEAN_THRESHOLD_DB: float = -32.0 # Mean volume below this indicates faint recording
LOW_VOLUME_PEAK_THRESHOLD_DB: float = -18.0 # Peak volume below this indicates faint recording
CLIPPING_PEAK_THRESHOLD_DB: float = -0.1    # Peak within 0.1 dBFS of 0 considered clipped
HIGH_SILENCE_PERCENTAGE_THRESHOLD: float = 35.0 # Greater than 35% silence triggers a warning
LONG_SILENCE_WARNING_SEC: float = 6.0       # Consecutive silence >= 6s triggers a warning


class SilenceInterval(BaseModel):
    """Represents a discrete detected silence period in the recording."""
    start_seconds: float = Field(description="Start time of the silent portion in seconds")
    end_seconds: float = Field(description="End time of the silent portion in seconds")
    duration_seconds: float = Field(description="Duration of the silent portion in seconds")


class AudioQualityMetrics(BaseModel):
    """
    Audio quality assessment metrics and descriptive compliance warnings.
    """

    mean_volume_db: float = Field(description="Average RMS volume in dBFS (decibels relative to full scale)")
    peak_amplitude_db: float = Field(description="Peak volume in dBFS (maximum audio amplitude)")
    is_clipped: bool = Field(description="True if audio clipping / waveform saturation is detected")
    clipping_samples_count: int = Field(default=0, description="Count of audio samples saturated at 0 dBFS")
    total_silence_seconds: float = Field(description="Cumulative seconds of detected silence")
    silence_percentage: float = Field(description="Percentage of total recording that is silent")
    longest_silence_seconds: float = Field(description="Duration of the single longest silent period in seconds")
    silent_segments_count: int = Field(description="Number of detected silent segments")
    is_low_volume: bool = Field(description="True if audio volume is abnormally low")
    warnings: List[str] = Field(default_factory=list, description="Descriptive quality warnings for auditor review")


def analyze_audio_quality(
    file_path: Union[str, Path],
    known_duration: Optional[float] = None,
) -> AudioQualityMetrics:
    """
    Run audio quality analysis on an audio file using FFmpeg volumedetect and silencedetect filters.

    Args:
        file_path: Path to the audio file.
        known_duration: Optional total audio duration in seconds.

    Returns:
        AudioQualityMetrics populated with volume, silence, clipping stats and warnings.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        AudioQualityAnalysisError: If FFmpeg fails to execute or parse filter output.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    ffmpeg_bin = get_ffmpeg_path()

    filter_spec = (
        f"volumedetect,"
        f"silencedetect=noise={SILENCE_NOISE_THRESHOLD_DB}dB:d={SILENCE_MIN_DURATION_SEC}"
    )

    cmd = [
        str(ffmpeg_bin),
        "-i",
        str(path),
        "-af",
        filter_spec,
        "-f",
        "null",
        "-",
    ]

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        stderr_output = completed.stderr
    except subprocess.CalledProcessError as exc:
        raise AudioQualityAnalysisError(
            f"Quality analysis failed for '{path.name}': {exc.stderr or exc}"
        ) from exc
    except Exception as exc:
        raise AudioQualityAnalysisError(
            f"Unexpected error analyzing quality for '{path.name}': {exc}"
        ) from exc

    return parse_ffmpeg_quality_output(stderr_output, known_duration=known_duration)


def parse_ffmpeg_quality_output(
    stderr_text: str,
    known_duration: Optional[float] = None,
) -> AudioQualityMetrics:
    """
    Parse the combined stderr output from FFmpeg's volumedetect and silencedetect filters.

    Args:
        stderr_text: FFmpeg log output from stderr.
        known_duration: Optional known total duration in seconds.

    Returns:
        Structured AudioQualityMetrics instance with warnings.
    """
    # 1. Parse mean volume
    mean_match = re.search(r"mean_volume:\s*([-0-9.]+)\s*dB", stderr_text)
    mean_volume_db = float(mean_match.group(1)) if mean_match else -99.0

    # 2. Parse max/peak volume
    max_match = re.search(r"max_volume:\s*([-0-9.]+)\s*dB", stderr_text)
    peak_amplitude_db = float(max_match.group(1)) if max_match else -99.0

    # 3. Parse clipping (histogram_0db)
    hist0_match = re.search(r"histogram_0db:\s*(\d+)", stderr_text)
    clipping_samples_count = int(hist0_match.group(1)) if hist0_match else 0
    is_clipped = (peak_amplitude_db >= CLIPPING_PEAK_THRESHOLD_DB) or (clipping_samples_count > 0)

    # 4. Parse silence detection
    # Format: [silencedetect @ ...] silence_start: 38.56
    #         [silencedetect @ ...] silence_end: 42.32 | silence_duration: 3.76
    silence_starts: List[float] = [
        float(m) for m in re.findall(r"silence_start:\s*([-0-9.]+)", stderr_text)
    ]
    silence_ends_with_dur = re.findall(
        r"silence_end:\s*([-0-9.]+)\s*\|\s*silence_duration:\s*([-0-9.]+)", stderr_text
    )

    silence_segments: List[SilenceInterval] = []
    total_silence_seconds = 0.0

    for end_str, dur_str in silence_ends_with_dur:
        dur = float(dur_str)
        end = float(end_str)
        start = max(0.0, end - dur)
        silence_segments.append(
            SilenceInterval(start_seconds=round(start, 2), end_seconds=round(end, 2), duration_seconds=round(dur, 2))
        )
        total_silence_seconds += dur

    # Check for trailing silence (silence_start without a corresponding silence_end at end of file)
    if len(silence_starts) > len(silence_ends_with_dur) and known_duration is not None:
        last_start = silence_starts[-1]
        if known_duration > last_start:
            trailing_dur = known_duration - last_start
            silence_segments.append(
                SilenceInterval(
                    start_seconds=round(last_start, 2),
                    end_seconds=round(known_duration, 2),
                    duration_seconds=round(trailing_dur, 2),
                )
            )
            total_silence_seconds += trailing_dur

    total_silence_seconds = round(total_silence_seconds, 2)
    longest_silence_seconds = (
        round(max((s.duration_seconds for s in silence_segments), default=0.0), 2)
        if silence_segments
        else 0.0
    )

    # Calculate silence percentage
    if known_duration and known_duration > 0:
        silence_percentage = round((total_silence_seconds / known_duration) * 100.0, 2)
    else:
        silence_percentage = 0.0

    # Low volume evaluation
    is_low_volume = (
        (mean_volume_db < LOW_VOLUME_MEAN_THRESHOLD_DB)
        or (peak_amplitude_db < LOW_VOLUME_PEAK_THRESHOLD_DB)
    )

    # Construct descriptive warnings
    warnings: List[str] = []

    if is_clipped:
        warnings.append(
            f"Audio clipping detected (peak amplitude reached {peak_amplitude_db:.1f} dBFS with "
            f"{clipping_samples_count} saturated samples); potential distortion during loud utterances."
        )

    if is_low_volume:
        warnings.append(
            f"Unusually low volume detected (mean RMS: {mean_volume_db:.1f} dBFS, peak: {peak_amplitude_db:.1f} dBFS); "
            f"speech may be faint and affect transcription accuracy."
        )

    if silence_percentage > HIGH_SILENCE_PERCENTAGE_THRESHOLD:
        warnings.append(
            f"High silence ratio detected ({silence_percentage:.1f}% of call duration is silent)."
        )

    if longest_silence_seconds >= LONG_SILENCE_WARNING_SEC:
        warnings.append(
            f"Extended silent period detected ({longest_silence_seconds:.1f}s consecutive dead air)."
        )

    if mean_volume_db > -10.0:
        warnings.append(
            f"Unusually high mean volume ({mean_volume_db:.1f} dBFS); possible continuous loud line noise or background interference."
        )

    metrics = AudioQualityMetrics(
        mean_volume_db=round(mean_volume_db, 2),
        peak_amplitude_db=round(peak_amplitude_db, 2),
        is_clipped=is_clipped,
        clipping_samples_count=clipping_samples_count,
        total_silence_seconds=total_silence_seconds,
        silence_percentage=silence_percentage,
        longest_silence_seconds=longest_silence_seconds,
        silent_segments_count=len(silence_segments),
        is_low_volume=is_low_volume,
        warnings=warnings,
    )

    return metrics
