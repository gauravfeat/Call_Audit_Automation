"""
Whisper Large-v3 ASR Adapter.

Provides local Faster-Whisper large-v3 transcription with automatic CUDA/CPU
device detection, anti-hallucination settings, and result normalization
into the common ASRResult schema.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

from app.benchmark.base import BaseASRAdapter
from app.benchmark.schema import ASRResult, BenchmarkSegment
from app.core.config import get_settings

logger = logging.getLogger("call_audit.benchmark.whisper_large_adapter")


def _resolve_device_and_compute() -> tuple:
    """
    Auto-detect the best available device and compute type.

    Returns: (device, compute_type, device_info)
    """
    settings = get_settings()
    requested_device = settings.WHISPER_LARGE_DEVICE.lower()
    requested_compute = settings.WHISPER_LARGE_COMPUTE_TYPE.lower()

    device_info = {}

    # Detect CUDA availability
    cuda_available = False
    try:
        import ctranslate2
        cuda_count = ctranslate2.get_cuda_device_count()
        cuda_available = cuda_count > 0
        device_info["cuda_device_count"] = cuda_count
    except Exception:
        device_info["cuda_device_count"] = 0

    # Resolve device
    if requested_device == "auto":
        device = "cuda" if cuda_available else "cpu"
    elif requested_device == "cuda" and not cuda_available:
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    else:
        device = requested_device

    # Resolve compute type
    if requested_compute == "auto":
        if device == "cuda":
            compute_type = "float16"
        else:
            compute_type = "int8"
    elif device == "cpu" and requested_compute == "float16":
        logger.warning("float16 not supported on CPU. Using int8 instead.")
        compute_type = "int8"
    else:
        compute_type = requested_compute

    device_info["selected_device"] = device
    device_info["selected_compute_type"] = compute_type

    return device, compute_type, device_info


class WhisperLargeAdapter(BaseASRAdapter):
    """
    Faster-Whisper Large-v3 local inference adapter.
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        beam_size: int = 1,
        vad_filter: bool = True,
    ):
        super().__init__(model_name="whisper-large-v3", model_version=model_size)
        self._model_size = model_size
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._model = None
        self._device_info = {}

    def _ensure_model(self):
        """Load the model on first use via WhisperModelManager."""
        if self._model is not None:
            return

        from app.transcription.model import WhisperModelManager

        device, compute_type, device_info = _resolve_device_and_compute()
        self._device_info = device_info

        logger.info(
            "Loading Whisper large-v3 model: device=%s, compute_type=%s",
            device, compute_type,
        )
        load_start = time.perf_counter()

        self._model = WhisperModelManager.get_model(
            model_size=self._model_size,
            device=device,
            compute_type=compute_type,
        )

        load_time = round(time.perf_counter() - load_start, 2)
        self._device_info["model_load_time_seconds"] = load_time
        logger.info("Whisper large-v3 model loaded in %.2fs.", load_time)

    def calculate_cost(self, duration_seconds: float) -> Optional[float]:
        """Local model — no API cost."""
        return None

    def transcribe(
        self,
        audio_path: Path,
        recording_id: str,
        input_audio_type: str = "processed_wav_16khz",
        **kwargs,
    ) -> ASRResult:
        """
        Transcribe audio using Faster-Whisper large-v3 locally.
        """
        audio_path = Path(audio_path).resolve()

        if not audio_path.is_file():
            return ASRResult(
                recording_id=recording_id,
                source_file=audio_path.name,
                input_audio_type=input_audio_type,
                model_name=self.model_name,
                model_version=self.model_version,
                transcription_mode="default",
                transcript="",
                processing_time_seconds=0.0,
                status="FAILED",
                error_details=f"Audio file not found: {audio_path}",
            )

        beam_size = kwargs.get("beam_size", self._beam_size)
        vad_filter = kwargs.get("vad_filter", self._vad_filter)
        warnings: List[str] = []

        try:
            self._ensure_model()
        except Exception as exc:
            return ASRResult(
                recording_id=recording_id,
                source_file=audio_path.name,
                input_audio_type=input_audio_type,
                model_name=self.model_name,
                model_version=self.model_version,
                transcription_mode="default",
                transcript="",
                processing_time_seconds=0.0,
                status="FAILED",
                error_details=f"Model load failed: {exc}",
                provider_metadata=self._device_info,
            )

        logger.info(
            "Whisper large-v3 inference: recording=%s, beam=%d, vad=%s, device=%s",
            recording_id, beam_size, vad_filter,
            self._device_info.get("selected_device", "unknown"),
        )

        start_time = time.perf_counter()

        try:
            segments_gen, info = self._model.transcribe(
                str(audio_path),
                beam_size=beam_size,
                temperature=0.0,
                condition_on_previous_text=False,
                repetition_penalty=1.1,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                vad_filter=vad_filter,
                vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
                word_timestamps=True,
                language=None,
                task="transcribe",
            )

            segments: List[BenchmarkSegment] = []
            full_text_parts = []
            seg_idx = 0

            for seg in segments_gen:
                text = seg.text.strip()
                if not text:
                    continue
                full_text_parts.append(text)
                segments.append(BenchmarkSegment(
                    segment_id=seg_idx,
                    start=round(float(seg.start), 3),
                    end=round(float(seg.end), 3),
                    text=text,
                    confidence=round(float(seg.avg_logprob), 4) if hasattr(seg, "avg_logprob") else None,
                ))
                seg_idx += 1

            elapsed = round(time.perf_counter() - start_time, 2)

            detected_lang = info.language or "unknown"
            lang_prob = round(float(info.language_probability or 0.0), 4)
            audio_duration = round(float(info.duration or 0.0), 2)

            provider_meta = {
                **self._device_info,
                "audio_duration_seconds": audio_duration,
                "total_segments": len(segments),
                "beam_size": beam_size,
                "vad_filter": vad_filter,
            }

            return ASRResult(
                recording_id=recording_id,
                source_file=audio_path.name,
                input_audio_type=input_audio_type,
                model_name=self.model_name,
                model_version=self.model_version,
                transcription_mode="default",
                detected_language=detected_lang,
                language_probability=lang_prob,
                transcript=" ".join(full_text_parts),
                segments=segments,
                processing_time_seconds=elapsed,
                api_cost_inr=None,
                status="SUCCESS",
                warnings=warnings,
                provider_metadata=provider_meta,
            )

        except Exception as exc:
            elapsed = round(time.perf_counter() - start_time, 2)
            logger.error("Whisper large-v3 inference failed for %s after %.2fs: %s", recording_id, elapsed, exc)
            return ASRResult(
                recording_id=recording_id,
                source_file=audio_path.name,
                input_audio_type=input_audio_type,
                model_name=self.model_name,
                model_version=self.model_version,
                transcription_mode="default",
                transcript="",
                processing_time_seconds=elapsed,
                status="FAILED",
                error_details=str(exc),
                provider_metadata=self._device_info,
            )
