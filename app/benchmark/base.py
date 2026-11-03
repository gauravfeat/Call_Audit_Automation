"""
Base ASR Adapter Interface.

Defines the abstract interface for all Speech-to-Text adapters (Whisper, Sarvam, etc.)
ensuring uniform execution, metric collection, and result normalization.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from app.benchmark.schema import ASRResult


class BaseASRAdapter(ABC):
    """
    Abstract base class for Speech-to-Text model adapters.
    """

    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        recording_id: str,
        input_audio_type: str = "processed_wav_16khz",
        **kwargs,
    ) -> ASRResult:
        """
        Transcribe an audio file and return a normalized ASRResult.

        Args:
            audio_path: Path to the target audio file.
            recording_id: Unique recording identifier.
            input_audio_type: Description of input format ('processed_wav_16khz' or 'original_mp3_8khz').
            **kwargs: Adapter-specific parameters.

        Returns:
            Normalized ASRResult object.
        """
        pass

    @abstractmethod
    def calculate_cost(self, duration_seconds: float) -> Optional[float]:
        """
        Calculate estimated API cost in INR based on audio duration.
        Returns None for local/free models.
        """
        pass
