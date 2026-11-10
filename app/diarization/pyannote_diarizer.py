"""
pyannote Community-1 Speaker Diarization Adapter.

Executes local, on-premises speaker diarization using the pyannote.audio Community-1
pipeline (pyannote/speaker-diarization-community-1).

Features:
- Completely on-premises inference (no audio leaves client infrastructure)
- Automatic speaker counting or optional constrained speaker counts (num_speakers)
- Normalizes speaker IDs to standard canonical format (SPEAKER_00, SPEAKER_01, ...)
- Calculates speaker turn timestamps and durations
- Handles single-speaker, two-speaker, and multi-speaker calls
- Gracefully handles offline inference from Hugging Face local cache
- Dependency injection friendly for unit testing with mocked pipelines
"""

import logging
import os
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from app.core.config import Settings, get_settings

logger = logging.getLogger("call_audit.diarization.pyannote")


class PyannoteDiarizationError(Exception):
    """Raised when pyannote diarization fails or environment is misconfigured."""
    pass


@dataclass
class SpeakerTurn:
    """Individual speaker activity interval."""
    speaker_id: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)


def _get_audio_duration(audio_path: Path) -> float:
    """Extract audio duration in seconds from WAV header, falling back to file size."""
    if audio_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(audio_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return round(frames / rate, 2)
        except Exception:
            pass
    try:
        return max(round(audio_path.stat().st_size / 32000.0, 2), 0.0)
    except Exception:
        return 0.0


def _normalize_speaker_id(raw_speaker: str) -> str:
    """
    Normalize pyannote speaker label (e.g. 'SPEAKER_00', 'SPEAKER_01', 'spk_0')
    to standard canonical format: SPEAKER_00, SPEAKER_01.
    """
    raw = str(raw_speaker).strip()
    if raw.upper().startswith("SPEAKER_"):
        suffix = raw[8:]
        try:
            num = int(suffix)
            return f"SPEAKER_{num:02d}"
        except ValueError:
            return raw.upper()

    # Try numeric suffix
    digits = "".join(ch for ch in raw if ch.isdigit())
    if digits:
        try:
            num = int(digits)
            return f"SPEAKER_{num:02d}"
        except ValueError:
            pass

    return f"SPEAKER_{raw.upper()}"


class PyannoteDiarizer:
    """
    Adapter for running local pyannote speaker diarization.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        device: Optional[str] = None,
        settings: Optional[Settings] = None,
        pipeline: Optional[Any] = None,
    ) -> None:
        """
        Initialize the Pyannote Diarizer.

        Args:
            model_name: Hugging Face model ID or path. Defaults to settings.PYANNOTE_MODEL.
            hf_token: Hugging Face user access token. Defaults to settings.HUGGINGFACE_TOKEN.
            device: 'cpu' or 'cuda'. Defaults to settings.PYANNOTE_DEVICE ("cpu").
            settings: Optional Settings instance.
            pipeline: Optional pre-loaded pyannote Pipeline instance (for tests / injection).
        """
        self.settings = settings or get_settings()
        self.model_name = model_name or getattr(
            self.settings, "PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1"
        )
        self.hf_token = hf_token or self.settings.HUGGINGFACE_TOKEN or os.getenv("HUGGINGFACE_TOKEN")
        self.device = device or getattr(self.settings, "PYANNOTE_DEVICE", "cpu")
        self._pipeline = pipeline

    @property
    def pipeline(self) -> Any:
        """Lazily load the pyannote Pipeline."""
        if self._pipeline is None:
            self._pipeline = self._load_pipeline()
        return self._pipeline

    def _load_pipeline(self) -> Any:
        """Load pyannote.audio Pipeline with authentication and move to device."""
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as e:
            raise PyannoteDiarizationError(
                "pyannote.audio or torch is not installed in the environment. "
                "Install using: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu "
                "&& pip install pyannote.audio"
            ) from e

        logger.info("Loading pyannote diarization pipeline: %s", self.model_name)
        start_time = time.perf_counter()

        try:
            # Check Hugging Face token (pyannote.audio 4.x uses 'token', older versions used 'use_auth_token')
            token_val = self.hf_token if self.hf_token else None
            try:
                pipeline = Pipeline.from_pretrained(
                    self.model_name,
                    token=token_val,
                )
            except TypeError:
                pipeline = Pipeline.from_pretrained(
                    self.model_name,
                    use_auth_token=token_val if token_val else True,
                )
        except Exception as e:
            msg = (
                f"Failed to load pyannote model '{self.model_name}'. "
                "Ensure you have accepted the model conditions at https://huggingface.co/"
                f"{self.model_name} and set HUGGINGFACE_TOKEN in your .env file: {e}"
            )
            logger.error(msg)
            raise PyannoteDiarizationError(msg) from e

        if pipeline is None:
            raise PyannoteDiarizationError(
                f"Could not load pyannote pipeline for '{self.model_name}'. Token may be invalid or access not accepted."
            )

        # Move to device
        try:
            device_obj = torch.device(self.device)
            pipeline.to(device_obj)
        except Exception as e:
            logger.warning("Could not send pyannote pipeline to device '%s': %s. Using default device.", self.device, e)

        load_duration = time.perf_counter() - start_time
        logger.info("Pyannote pipeline loaded successfully in %.2fs on device %s", load_duration, self.device)
        return pipeline

    def diarize_file(
        self,
        audio_path: Union[str, Path],
        recording_id: Optional[str] = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[SpeakerTurn]:
        """
        Run speaker diarization on a single audio recording.

        Args:
            audio_path: Path to target 16kHz mono WAV file.
            recording_id: Optional recording ID for logging.
            num_speakers: Optional exact number of speakers (e.g. 2).
            min_speakers: Optional minimum number of speakers.
            max_speakers: Optional maximum number of speakers.

        Returns:
            List of SpeakerTurn intervals sorted by start time.
        """
        path = Path(audio_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        rec_id = recording_id or path.stem

        if path.stat().st_size == 0:
            raise PyannoteDiarizationError(f"Cannot diarize empty or zero-byte audio file: {path.name}")

        logger.info("Running pyannote diarization on '%s' (ID: %s)...", path.name, rec_id)
        start_time = time.perf_counter()

        pipeline = self.pipeline

        # Build pipeline inference options
        kwargs: Dict[str, Any] = {}
        if num_speakers is not None and num_speakers > 0:
            kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers is not None:
                kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                kwargs["max_speakers"] = max_speakers

        try:
            # Run inference
            diarization_output = pipeline(str(path), **kwargs)
        except Exception as e:
            msg = f"pyannote diarization inference failed for {rec_id}: {e}"
            logger.error(msg)
            raise PyannoteDiarizationError(msg) from e

        inference_time = time.perf_counter() - start_time
        turns = self._parse_diarization_output(diarization_output)
        logger.info(
            "Diarization completed for %s in %.2fs: %d turns, %d unique speakers detected",
            rec_id,
            inference_time,
            len(turns),
            len(set(t.speaker_id for t in turns)),
        )
        return turns

    def _parse_diarization_output(self, diarization_output: Any) -> List[SpeakerTurn]:
        """Parse pyannote Annotation / tracks into standard SpeakerTurn objects."""
        turns: List[SpeakerTurn] = []

        # Pyannote Annotation iter_tracks API
        if hasattr(diarization_output, "itertracks"):
            for turn, track, speaker in diarization_output.itertracks(yield_label=True):
                turns.append(
                    SpeakerTurn(
                        speaker_id=_normalize_speaker_id(str(speaker)),
                        start=round(float(turn.start), 3),
                        end=round(float(turn.end), 3),
                    )
                )
        elif isinstance(diarization_output, list):
            # Useful for mocked objects in tests
            for item in diarization_output:
                if isinstance(item, SpeakerTurn):
                    turns.append(item)
                elif isinstance(item, dict):
                    turns.append(
                        SpeakerTurn(
                            speaker_id=_normalize_speaker_id(str(item.get("speaker_id") or item.get("speaker"))),
                            start=round(float(item.get("start", 0.0)), 3),
                            end=round(float(item.get("end", 0.0)), 3),
                        )
                    )

        # Sort chronologically by start timestamp
        turns.sort(key=lambda t: (t.start, t.end))
        return turns

