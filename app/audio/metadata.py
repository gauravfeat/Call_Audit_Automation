"""
Audio Metadata Extraction Module.

Extracts technical audio properties using FFprobe and parses them into
a strongly-typed Pydantic model.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, Set, Union

from pydantic import BaseModel, Field

from app.audio.exceptions import AudioMetadataError, UnsupportedAudioFormatError
from app.audio.ffmpeg_utils import get_ffprobe_path

logger = logging.getLogger("call_audit.audio.metadata")

SUPPORTED_AUDIO_EXTENSIONS: Set[str] = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


class AudioMetadata(BaseModel):
    """
    Structured container for extracted audio stream and container metadata.
    """

    filename: str = Field(description="Name of the audio file")
    file_path: str = Field(description="Absolute path to the audio file")
    file_size_bytes: int = Field(description="File size on disk in bytes")
    audio_format: str = Field(description="Audio container format (e.g., mp3, wav, mov,mp4,m4a)")
    codec_name: str = Field(description="Audio stream codec name (e.g., mp3, pcm_s16le, aac)")
    duration_seconds: float = Field(description="Audio duration in seconds")
    sample_rate: int = Field(description="Audio sampling rate in Hz (e.g., 8000, 16000, 44100)")
    channels: int = Field(description="Number of audio channels (1 = mono, 2 = stereo)")
    channel_layout: Optional[str] = Field(default=None, description="Channel layout name (mono, stereo, etc.)")
    bit_depth: Optional[int] = Field(default=None, description="Bit depth per sample if uncompressed/lossless")
    bitrate_kbps: Optional[float] = Field(default=None, description="Bitrate in kilobits per second")


def extract_metadata(file_path: Union[str, Path]) -> AudioMetadata:
    """
    Extract comprehensive audio metadata for a given file using FFprobe.

    Args:
        file_path: Path to the target audio file.

    Returns:
        Populated AudioMetadata instance.

    Raises:
        FileNotFoundError: If the input file does not exist.
        UnsupportedAudioFormatError: If the file extension is not supported.
        AudioMetadataError: If FFprobe fails to execute or parse stream data.
    """
    path = Path(file_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        raise UnsupportedAudioFormatError(
            f"Unsupported audio format '{ext}' for file '{path.name}'. "
            f"Supported extensions: {sorted(list(SUPPORTED_AUDIO_EXTENSIONS))}"
        )

    ffprobe_bin = get_ffprobe_path()

    cmd = [
        str(ffprobe_bin),
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        data = json.loads(completed.stdout)
    except subprocess.CalledProcessError as exc:
        raise AudioMetadataError(
            f"FFprobe failed to read '{path.name}': {exc.stderr or exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise AudioMetadataError(
            f"Failed to parse FFprobe JSON output for '{path.name}': {exc}"
        ) from exc
    except Exception as exc:
        raise AudioMetadataError(
            f"Unexpected error extracting metadata from '{path.name}': {exc}"
        ) from exc

    # Locate audio stream
    streams = data.get("streams", [])
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not audio_stream:
        raise AudioMetadataError(f"No audio stream found in '{path.name}'")

    format_info = data.get("format", {})

    # Extract duration (prefer stream duration, fallback to format duration)
    raw_duration = audio_stream.get("duration") or format_info.get("duration")
    try:
        duration_seconds = round(float(raw_duration), 3) if raw_duration is not None else 0.0
    except (ValueError, TypeError):
        duration_seconds = 0.0

    # Extract sample rate
    try:
        sample_rate = int(audio_stream.get("sample_rate", 0))
    except (ValueError, TypeError):
        sample_rate = 0

    # Extract channels
    try:
        channels = int(audio_stream.get("channels", 1))
    except (ValueError, TypeError):
        channels = 1

    channel_layout = audio_stream.get("channel_layout")
    if not channel_layout:
        channel_layout = "mono" if channels == 1 else "stereo" if channels == 2 else f"{channels}_channels"

    # Extract bit depth (only for PCM / lossless where bits_per_sample > 0)
    raw_bits = audio_stream.get("bits_per_sample") or audio_stream.get("bits_per_raw_sample")
    bit_depth: Optional[int] = None
    if raw_bits:
        try:
            val = int(raw_bits)
            if val > 0:
                bit_depth = val
        except (ValueError, TypeError):
            bit_depth = None

    # Extract bitrate in kbps (prefer format bit_rate, fallback to stream bit_rate)
    raw_bitrate = format_info.get("bit_rate") or audio_stream.get("bit_rate")
    bitrate_kbps: Optional[float] = None
    if raw_bitrate:
        try:
            bitrate_kbps = round(float(raw_bitrate) / 1000.0, 2)
        except (ValueError, TypeError):
            bitrate_kbps = None

    # File size
    file_size_bytes = path.stat().st_size

    metadata = AudioMetadata(
        filename=path.name,
        file_path=str(path),
        file_size_bytes=file_size_bytes,
        audio_format=format_info.get("format_name", ext.lstrip(".")),
        codec_name=audio_stream.get("codec_name", "unknown"),
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        channels=channels,
        channel_layout=channel_layout,
        bit_depth=bit_depth,
        bitrate_kbps=bitrate_kbps,
    )

    logger.debug("Extracted metadata for %s: %s", path.name, metadata.model_dump())
    return metadata
