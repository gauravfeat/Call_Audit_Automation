"""
Audio Preprocessing and Standardization Module.

Standardizes input telephony recordings into downstream-ready format:
- WAV container format
- 16,000 Hz sample rate (optimal for Whisper STT models)
- Mono channel (1 channel downmix/passthrough)
- PCM signed 16-bit little-endian encoding (pcm_s16le)

Preserves original input recordings completely untouched.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional, Union

from app.audio.exceptions import AudioPreprocessingError
from app.audio.ffmpeg_utils import get_ffmpeg_path
from app.core.config import get_settings

logger = logging.getLogger("call_audit.audio.preprocess")

TARGET_SAMPLE_RATE: int = 16000
TARGET_CHANNELS: int = 1
TARGET_CODEC: str = "pcm_s16le"


def preprocess_audio(
    input_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    output_filename: Optional[str] = None,
    overwrite: bool = True,
) -> Path:
    """
    Standardize an audio recording to 16kHz mono 16-bit PCM WAV.

    Args:
        input_path: Path to the raw source recording (MP3, WAV, M4A, etc.).
        output_dir: Destination directory. Defaults to settings.processed_audio_dir.
        output_filename: Custom output filename. Defaults to '<input_stem>.wav'.
        overwrite: Whether to overwrite target if it already exists. Defaults to True.

    Returns:
        Path to the successfully converted WAV file.

    Raises:
        FileNotFoundError: If input audio file does not exist.
        AudioPreprocessingError: If conversion fails or output is invalid.
    """
    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input audio file does not exist: {source}")

    settings = get_settings()
    out_dir = Path(output_dir).resolve() if output_dir else settings.processed_audio_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dest_filename = output_filename if output_filename else f"{source.stem}.wav"
    destination = (out_dir / dest_filename).resolve()

    # Safety check: Never overwrite original source file
    if destination == source:
        raise AudioPreprocessingError(
            f"Output destination cannot be identical to source recording: {source}"
        )

    ffmpeg_bin = get_ffmpeg_path()

    cmd = [
        str(ffmpeg_bin),
        "-y" if overwrite else "-n",
        "-i",
        str(source),
        "-vn",                       # Strip video streams if present
        "-acodec",
        TARGET_CODEC,                # pcm_s16le
        "-ac",
        str(TARGET_CHANNELS),        # 1 (mono)
        "-ar",
        str(TARGET_SAMPLE_RATE),     # 16000 Hz
        str(destination),
    ]

    logger.info("Preprocessing audio: %s -> %s", source.name, destination.name)

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=180,
        )
    except subprocess.CalledProcessError as exc:
        raise AudioPreprocessingError(
            f"FFmpeg conversion failed for '{source.name}': {exc.stderr or exc}"
        ) from exc
    except Exception as exc:
        raise AudioPreprocessingError(
            f"Unexpected error converting '{source.name}': {exc}"
        ) from exc

    # Verification: output file must exist and have non-zero size
    if not destination.is_file() or destination.stat().st_size == 0:
        raise AudioPreprocessingError(
            f"Conversion finished but output file is missing or zero bytes: {destination}"
        )

    logger.info("Successfully converted '%s' (size: %d bytes)", destination.name, destination.stat().st_size)
    return destination
