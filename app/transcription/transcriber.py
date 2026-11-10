"""
Audio Transcriber Module.

Executes speech-to-text inference on standardized 16kHz mono WAV recordings.
Default & Primary Engine: Sarvam Saaras v4 (mode=codemix).
Optional Engine: Faster-Whisper (small / large-v3) for local benchmarking.

Features:
- Primary Sarvam Saaras v4 codemix pipeline with financial keyterms priming
- Automatic language detection (Hindi/English code-mixed)
- Chunk-level timestamps and segment extraction
- Caching & reuse of valid existing transcripts to prevent duplicate API charges
- Full backward-compatibility with Faster-Whisper when explicitly requested or passed as model
- Serialization into canonical Transcript Pydantic schema
"""

import logging
import time
import wave
from pathlib import Path
from typing import Any, List, Optional, Union

from app.core.config import Settings, get_settings
from app.transcription.transcript_schema import Transcript, TranscriptSegment, WordTimestamp

logger = logging.getLogger("call_audit.transcription.transcriber")


class TranscriptionError(Exception):
    """Raised when audio transcription fails."""
    pass


def _get_audio_duration(path: Path) -> float:
    """Extract audio duration in seconds from WAV header, with fallback."""
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return round(frames / rate, 2)
        except Exception:
            pass
    # Fallback from file size
    try:
        return max(round(path.stat().st_size / 32000.0, 2), 0.0)
    except Exception:
        return 0.0


class AudioTranscriber:
    """
    Executes speech-to-text transcription on standardized audio files.
    Primary STT Engine: Sarvam Saaras v4 Codemix.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        mode: Optional[str] = None,
        model: Optional[Any] = None,
        adapter: Optional[Any] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Initialize AudioTranscriber.

        Args:
            provider: STT provider ('sarvam' or 'whisper'). Defaults to settings.DEFAULT_STT_PROVIDER ("sarvam").
            mode: Transcription mode for Sarvam ('codemix' or 'verbatim'). Defaults to settings.SARVAM_MODE ("codemix").
            model: Optional pre-loaded Faster-Whisper WhisperModel instance. If provided without explicit provider, uses 'whisper'.
            adapter: Optional pre-initialized BaseASRAdapter instance (e.g. SarvamAdapter or mock).
            settings: Optional application settings. Defaults to cached settings.
        """
        self.settings = settings or get_settings()

        # If model is explicitly passed (e.g., in unit tests), default provider to whisper
        if model is not None and provider is None:
            self.provider = "whisper"
        else:
            self.provider = (provider or self.settings.DEFAULT_STT_PROVIDER).lower()

        self.mode = mode or self.settings.SARVAM_MODE or "codemix"
        self._model = model
        self._adapter = adapter

    @property
    def model(self) -> Any:
        """Get or lazily load the Faster-Whisper model when provider='whisper'."""
        if self._model is None:
            from app.transcription.model import WhisperModelManager
            self._model = WhisperModelManager.get_model()
        return self._model

    @property
    def adapter(self) -> Any:
        """Get or lazily load the SarvamAdapter when provider='sarvam'."""
        if self._adapter is None:
            if self.provider == "sarvam":
                from app.benchmark.sarvam_adapter import SarvamAdapter
                self._adapter = SarvamAdapter(
                    api_key=self.settings.SARVAM_API_KEY,
                    model=self.settings.SARVAM_MODEL,
                    mode=self.mode,
                )
            else:
                from app.benchmark.whisper_small_adapter import WhisperSmallAdapter
                self._adapter = WhisperSmallAdapter()
        return self._adapter

    def transcribe_file(
        self,
        audio_path: Union[str, Path],
        recording_id: Optional[str] = None,
        reuse_existing: bool = True,
    ) -> Transcript:
        """
        Transcribe a standardized audio file into structured text and timestamps.

        Args:
            audio_path: Path to target audio file (WAV format expected).
            recording_id: Optional custom recording ID. Defaults to file stem.
            reuse_existing: If True, reuse existing valid transcript if available to avoid duplicate charges.

        Returns:
            Validated Transcript schema object.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            TranscriptionError: If audio file is empty or decoding fails.
        """
        path = Path(audio_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if path.stat().st_size == 0:
            raise TranscriptionError(f"Cannot transcribe empty or zero-byte audio file: {path.name}")

        rec_id = recording_id or path.stem

        if self.provider == "sarvam":
            return self._transcribe_sarvam(path, rec_id, reuse_existing=reuse_existing)
        elif self.provider in ("whisper_api", "remote_whisper"):
            return self._transcribe_whisper_api(path, rec_id, reuse_existing=reuse_existing)
        else:
            return self._transcribe_whisper(path, rec_id)

    def _transcribe_whisper_api(
        self,
        path: Path,
        rec_id: str,
        reuse_existing: bool = True,
    ) -> Transcript:
        """Transcribe using Remote Whisper Large-v3 API."""
        from app.transcription.whisper_api_client import WhisperAPIClient, WhisperAPIError
        logger.info(
            "Starting Remote Whisper Large-v3 API transcription for '%s' (ID: %s)...",
            path.name, rec_id,
        )
        try:
            client = WhisperAPIClient(settings=self.settings)
            return client.transcribe_file(
                audio_path=path,
                recording_id=rec_id,
                reuse_existing=reuse_existing,
                save_output=True,
            )
        except WhisperAPIError as exc:
            raise TranscriptionError(f"Whisper API transcription failed for {rec_id}: {exc}") from exc

    def _transcribe_sarvam(
        self,
        path: Path,
        rec_id: str,
        reuse_existing: bool = True,
    ) -> Transcript:
        """Transcribe using Sarvam Saaras v4 (primary pipeline)."""
        logger.info(
            "Starting Sarvam Saaras v4 transcription for '%s' (ID: %s, mode: %s)...",
            path.name, rec_id, self.mode,
        )

        # 1. Check existing primary transcript file
        primary_file = self.settings.primary_transcripts_dir / f"{rec_id}.json"
        if reuse_existing and primary_file.is_file():
            try:
                cached = Transcript.from_json(primary_file)
                if cached.segments and cached.full_text:
                    logger.info("Reusing cached primary transcript for %s", rec_id)
                    return cached
            except Exception:
                pass

        # 2. Check existing benchmark result in stt_benchmark/sarvam_v4_codemix
        benchmark_file = (
            self.settings.stt_benchmark_dir
            / f"sarvam_v4_{self.mode}"
            / f"{rec_id}.json"
        )
        if reuse_existing and benchmark_file.is_file():
            try:
                from app.benchmark.schema import ASRResult
                asr_res = ASRResult.from_json(benchmark_file)
                if asr_res.status == "SUCCESS" and asr_res.transcript:
                    logger.info(
                        "Converting existing benchmark Sarvam result for %s to primary transcript",
                        rec_id,
                    )
                    duration = _get_audio_duration(path)
                    transcript = Transcript.from_asr_result(asr_res, duration=duration)
                    return transcript
            except Exception as exc:
                logger.debug("Could not reuse benchmark result for %s: %s", rec_id, exc)

        # 3. Fresh transcription via SarvamAdapter
        start_time = time.perf_counter()
        try:
            asr_res = self.adapter.transcribe(
                audio_path=path,
                recording_id=rec_id,
                input_audio_type="processed_wav_16khz",
                mode=self.mode,
            )
            if asr_res.status == "FAILED":
                raise TranscriptionError(asr_res.error_details or "Sarvam transcription failed.")

            duration = _get_audio_duration(path)
            transcript = Transcript.from_asr_result(asr_res, duration=duration)

            logger.info(
                "Completed Sarvam transcription for '%s' in %.2fs. Words: %d, Segments: %d, Cost: INR %.4f",
                path.name,
                transcript.processing_time_seconds,
                transcript.total_words,
                len(transcript.segments),
                transcript.api_cost_inr or 0.0,
            )
            return transcript

        except TranscriptionError:
            raise
        except Exception as exc:
            elapsed = round(time.perf_counter() - start_time, 2)
            err_msg = f"Sarvam transcription failed for '{path.name}' after {elapsed:.2f}s: {exc}"
            logger.error(err_msg)
            raise TranscriptionError(err_msg) from exc

    def _transcribe_whisper(self, path: Path, rec_id: str) -> Transcript:
        """Transcribe using Faster-Whisper (backward compatibility / testing)."""
        logger.info("Starting Faster-Whisper transcription for '%s' (ID: %s)...", path.name, rec_id)
        start_time = time.perf_counter()
        try:
            segments_generator, info = self.model.transcribe(
                str(path),
                beam_size=self.settings.WHISPER_BEAM_SIZE,
                temperature=0.0,
                condition_on_previous_text=False,
                repetition_penalty=1.1,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                vad_filter=self.settings.WHISPER_VAD_FILTER,
                vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
                word_timestamps=self.settings.WHISPER_WORD_TIMESTAMPS,
                language=None,
                task="transcribe",
            )

            segments: List[TranscriptSegment] = []
            segment_idx = 0
            for seg in segments_generator:
                text_clean = seg.text.strip()
                if not text_clean:
                    continue

                words_list: Optional[List[WordTimestamp]] = None
                if getattr(seg, "words", None):
                    words_list = []
                    for w in seg.words:
                        w_text = w.word.strip()
                        if not w_text:
                            continue
                        prob = getattr(w, "probability", None)
                        words_list.append(
                            WordTimestamp(
                                word=w_text,
                                start=round(float(w.start), 3),
                                end=round(float(w.end), 3),
                                probability=round(float(prob), 4) if prob is not None else None,
                            )
                        )

                segments.append(
                    TranscriptSegment(
                        segment_id=segment_idx,
                        start=round(float(seg.start), 3),
                        end=round(float(seg.end), 3),
                        text=text_clean,
                        words=words_list,
                    )
                )
                segment_idx += 1

            elapsed_seconds = round(time.perf_counter() - start_time, 2)
            detected_lang = info.language or "unknown"
            lang_prob = round(float(info.language_probability or 0.0), 4)
            audio_duration = round(float(info.duration or 0.0), 2)

            return Transcript(
                recording_id=rec_id,
                source_file=path.name,
                language=detected_lang,
                language_probability=lang_prob,
                duration=audio_duration,
                model=self.settings.WHISPER_MODEL,
                processing_time_seconds=elapsed_seconds,
                segments=segments,
                provider="faster-whisper",
                transcription_mode="default",
                api_cost_inr=0.0,
            )

        except Exception as exc:
            elapsed_seconds = round(time.perf_counter() - start_time, 2)
            err_msg = f"Whisper transcription failed for '{path.name}' after {elapsed_seconds:.2f}s: {exc}"
            logger.error(err_msg)
            raise TranscriptionError(err_msg) from exc

    def save_transcript(
        self,
        transcript: Transcript,
        output_dir: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Serialize and write a Transcript to disk as JSON.

        Args:
            transcript: Validated Transcript object.
            output_dir: Destination folder. Defaults to primary_transcripts_dir for Sarvam,
                        or transcripts_dir for Whisper.
            filename: Custom filename. Defaults to '<recording_id>.json'.

        Returns:
            Path to the written JSON file.
        """
        if output_dir:
            dest_dir = Path(output_dir).resolve()
        elif transcript.provider == "sarvam":
            dest_dir = self.settings.primary_transcripts_dir
        else:
            dest_dir = self.settings.transcripts_dir

        dest_filename = filename or f"{transcript.recording_id}.json"
        dest_path = dest_dir / dest_filename
        return transcript.save_json(dest_path)
