"""
Whisper Model Manager Module.

Provides singleton-style caching and lifecycle management for Faster-Whisper
models to avoid reloading weights repeatedly across recordings.
"""

import logging
from typing import Dict, Optional, Tuple

from app.core.config import get_settings

logger = logging.getLogger("call_audit.transcription.model")


class ModelLoadError(Exception):
    """Raised when loading the Faster-Whisper model fails."""
    pass


class WhisperModelManager:
    """
    Manages Faster-Whisper model instances with caching.
    """

    _instances: Dict[Tuple[str, str, str], object] = {}

    @classmethod
    def get_model(
        cls,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        download_root: Optional[str] = None,
    ):
        """
        Retrieve or initialize a cached Faster-Whisper WhisperModel instance.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v3).
            device: Inference device (cpu or cuda).
            compute_type: Quantization / compute type (int8, float16, float32).
            download_root: Optional custom directory to store downloaded model weights.

        Returns:
            faster_whisper.WhisperModel instance.

        Raises:
            ModelLoadError: If loading or downloading the model fails.
        """
        settings = get_settings()
        selected_model = model_size or settings.WHISPER_MODEL
        selected_device = (device or settings.WHISPER_DEVICE).lower()
        selected_compute = compute_type or settings.WHISPER_COMPUTE_TYPE

        # Auto-adjust compute_type for CPU if float16 was accidentally specified
        if selected_device == "cpu" and selected_compute == "float16":
            logger.warning("float16 is not supported on CPU in CTranslate2. Adjusting compute_type to 'int8'.")
            selected_compute = "int8"

        cache_key = (selected_model, selected_device, selected_compute)

        if cache_key in cls._instances:
            logger.debug("Reusing cached Faster-Whisper model for key: %s", cache_key)
            return cls._instances[cache_key]

        logger.info(
            "Loading Faster-Whisper model: size='%s', device='%s', compute_type='%s'...",
            selected_model,
            selected_device,
            selected_compute,
        )

        try:
            from faster_whisper import WhisperModel

            model = WhisperModel(
                model_size_or_path=selected_model,
                device=selected_device,
                compute_type=selected_compute,
                download_root=download_root,
            )
            cls._instances[cache_key] = model
            logger.info("Faster-Whisper model '%s' loaded successfully.", selected_model)
            return model

        except ImportError as exc:
            raise ModelLoadError(
                "faster-whisper package is not installed. Run: pip install faster-whisper"
            ) from exc
        except RuntimeError as exc:
            err_str = str(exc)
            if "CUDA" in err_str or "cuDNN" in err_str:
                raise ModelLoadError(
                    f"CUDA execution failed: {err_str}\n"
                    "If running on a machine without NVIDIA CUDA/cuDNN, set WHISPER_DEVICE=cpu "
                    "and WHISPER_COMPUTE_TYPE=int8 in your .env file."
                ) from exc
            raise ModelLoadError(f"Failed to initialize Faster-Whisper model '{selected_model}': {exc}") from exc
        except Exception as exc:
            raise ModelLoadError(
                f"Unexpected error loading Faster-Whisper model '{selected_model}': {exc}"
            ) from exc

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached model instances (useful for testing or memory reclamation)."""
        cls._instances.clear()
        logger.info("Cleared Faster-Whisper model cache.")


def get_whisper_model(
    model_size: Optional[str] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
):
    """
    Convenience function to obtain the configured Faster-Whisper model instance.
    """
    return WhisperModelManager.get_model(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
    )
