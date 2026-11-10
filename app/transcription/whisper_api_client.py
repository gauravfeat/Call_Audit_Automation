"""
Remote Whisper Large-v3 API Client and Adapter.

Connects to a remote Whisper Large-v3 API endpoint (e.g. FastAPI / vLLM on GPU VM)
to perform Speech-to-Text transcription. Serializes responses into canonical
Transcript objects and automatically saves outputs in output/transcripts/onprem_whisper/.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.core.config import Settings, get_settings
from app.transcription.transcript_schema import Transcript, TranscriptSegment, WordTimestamp

logger = logging.getLogger("call_audit.transcription.whisper_api")


class WhisperAPIError(Exception):
    """Raised when remote Whisper API transcription fails."""
    pass


class WhisperAPIClient:
    """
    Client for remote Whisper Large-v3 FastAPI transcription service.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        response_format: Optional[str] = None,
        temperature: Optional[float] = None,
        settings: Optional[Settings] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        """
        Initialize the Whisper API Client.

        Args:
            base_url: Remote API base URL (e.g. http://122.170.2.205:7091).
            model: Whisper model identifier ('large-v3').
            timeout: Request timeout in seconds.
            response_format: 'verbose_json' or 'json'.
            temperature: Sampling temperature.
            settings: Application settings instance.
            session: Optional pre-configured requests.Session (useful for dependency injection / testing).
        """
        self.settings = settings or get_settings()
        self.base_url = (base_url or self.settings.WHISPER_API_BASE_URL).rstrip("/")
        raw_model = model or getattr(self.settings, "WHISPER_MODEL", "openai/whisper-large-v3")
        if raw_model in ("large-v3", "whisper-large-v3", "openai/whisper-large-v3"):
            self.model = "openai/whisper-large-v3"
        else:
            self.model = raw_model
        self.timeout = timeout or getattr(self.settings, "WHISPER_TIMEOUT", 600)
        self.response_format = response_format or getattr(self.settings, "WHISPER_RESPONSE_FORMAT", "verbose_json")
        self.temperature = temperature if temperature is not None else getattr(self.settings, "WHISPER_TEMPERATURE", 0.0)

        if session is not None:
            self.session = session
        else:
            self.session = requests.Session()
            retry_strategy = Retry(
                total=2,
                backoff_factor=1.0,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["POST"],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)

    @property
    def endpoint_url(self) -> str:
        """Full transcription endpoint URL."""
        return f"{self.base_url}/v1/audio/transcriptions"

    def transcribe_file(
        self,
        audio_path: Union[str, Path],
        recording_id: Optional[str] = None,
        reuse_existing: bool = True,
        save_output: bool = True,
    ) -> Transcript:
        """
        Transcribe an audio file via the remote Whisper Large-v3 API.

        Args:
            audio_path: Path to the input audio file (.wav or .mp3).
            recording_id: Optional recording identifier. Defaults to audio_path.stem.
            reuse_existing: If True and valid onprem_whisper transcript exists, returns it without calling API.
            save_output: If True, saves JSON and TXT outputs into output/transcripts/onprem_whisper/.

        Returns:
            Canonical Transcript object.
        """
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        rec_id = recording_id or audio_path.stem
        out_dir = self.settings.onprem_whisper_transcripts_dir
        cached_json = out_dir / f"{rec_id}.json"

        # Check existing cached transcript
        if reuse_existing and cached_json.is_file():
            try:
                with open(cached_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("recording_id") == rec_id and (data.get("segments") or data.get("text")):
                    logger.info("Reusing cached Whisper transcript for %s from %s", rec_id, cached_json)
                    return self._build_transcript_from_saved(data, rec_id, audio_path)
            except Exception as e:
                logger.warning("Could not read cached transcript at %s: %s. Re-transcribing.", cached_json, e)

        logger.info(
            "Sending %s (%s, %.1f KB) to Whisper API at %s",
            rec_id, audio_path.name, audio_path.stat().st_size / 1024, self.endpoint_url
        )

        start_time = time.perf_counter()
        mime_type = "audio/wav" if audio_path.suffix.lower() == ".wav" else "audio/mpeg"

        try:
            with open(audio_path, "rb") as f:
                files = {"file": (audio_path.name, f, mime_type)}
                data = {
                    "model": self.model,
                    "response_format": self.response_format,
                    "temperature": str(self.temperature),
                }

                response = self.session.post(
                    self.endpoint_url,
                    files=files,
                    data=data,
                    headers={"accept": "application/json"},
                    timeout=self.timeout,
                )

        except requests.exceptions.Timeout as e:
            msg = f"Whisper API request timed out after {self.timeout}s for {rec_id}: {e}"
            logger.error(msg)
            raise WhisperAPIError(msg) from e
        except requests.exceptions.ConnectionError as e:
            msg = f"Failed to connect to Whisper API at {self.endpoint_url} for {rec_id}: {e}"
            logger.error(msg)
            raise WhisperAPIError(msg) from e
        except Exception as e:
            msg = f"Unexpected network failure calling Whisper API for {rec_id}: {e}"
            logger.error(msg)
            raise WhisperAPIError(msg) from e

        processing_time = round(time.perf_counter() - start_time, 2)

        if response.status_code != 200:
            error_body = response.text[:500]
            msg = f"Whisper API returned HTTP {response.status_code} for {rec_id}: {error_body}"
            logger.error(msg)
            raise WhisperAPIError(msg)

        try:
            raw_response = response.json()
        except Exception as e:
            msg = f"Invalid JSON response from Whisper API for {rec_id}: {e}"
            logger.error(msg)
            raise WhisperAPIError(msg) from e

        transcript_text = (raw_response.get("text") or "").strip()
        if not transcript_text:
            logger.warning("Whisper API returned empty transcript text for %s", rec_id)

        transcript = self._parse_api_response(
            raw_response=raw_response,
            recording_id=rec_id,
            source_file=audio_path.name,
            processing_time_seconds=processing_time,
        )

        if save_output:
            self.save_transcript_outputs(
                transcript=transcript,
                raw_response=raw_response,
                recording_id=rec_id,
                source_audio_path=audio_path,
                processing_time_seconds=processing_time,
            )

        return transcript

    def _parse_api_response(
        self,
        raw_response: Dict[str, Any],
        recording_id: str,
        source_file: str,
        processing_time_seconds: float,
    ) -> Transcript:
        """Parse raw API dictionary into standard Transcript model."""
        full_text = (raw_response.get("text") or "").strip()
        duration = float(raw_response.get("duration") or raw_response.get("usage", {}).get("seconds") or 0.0)
        language = str(raw_response.get("language") or "hi")

        segments: List[TranscriptSegment] = []
        raw_segments = raw_response.get("segments") or []

        if raw_segments:
            for idx, seg in enumerate(raw_segments):
                seg_text = (seg.get("text") or "").strip()
                start_val = round(float(seg.get("start", 0.0)), 3)
                end_val = round(float(seg.get("end", start_val)), 3)

                words_list: Optional[List[WordTimestamp]] = None
                if "words" in seg and isinstance(seg["words"], list):
                    words_list = []
                    for w in seg["words"]:
                        w_word = str(w.get("word") or "")
                        w_start = round(float(w.get("start", start_val)), 3)
                        w_end = round(float(w.get("end", w_start)), 3)
                        w_prob = float(w["probability"]) if "probability" in w else None
                        words_list.append(WordTimestamp(word=w_word, start=w_start, end=w_end, probability=w_prob))

                segments.append(
                    TranscriptSegment(
                        segment_id=idx,
                        start=start_val,
                        end=end_val,
                        text=seg_text,
                        words=words_list,
                    )
                )
        elif full_text:
            # Fallback if API returned json instead of verbose_json
            segments.append(
                TranscriptSegment(
                    segment_id=0,
                    start=0.0,
                    end=duration if duration > 0 else 0.0,
                    text=full_text,
                )
            )

        return Transcript(
            recording_id=recording_id,
            source_file=source_file,
            language=language,
            language_probability=1.0,
            duration=duration,
            model=f"whisper-large-v3-api:{self.model}",
            processing_time_seconds=processing_time_seconds,
            segments=segments,
            provider="whisper_api",
            transcription_mode=self.response_format,
            provider_metadata={
                "base_url": self.base_url,
                "model": self.model,
                "response_format": self.response_format,
                "temperature": self.temperature,
                "usage": raw_response.get("usage"),
            },
        )

    def _build_transcript_from_saved(
        self,
        saved_data: Dict[str, Any],
        recording_id: str,
        audio_path: Path,
    ) -> Transcript:
        """Reconstruct Transcript object from previously saved onprem_whisper JSON."""
        # If saved file is canonical Transcript schema
        if "segments" in saved_data and "processing_time_seconds" in saved_data:
            try:
                return Transcript.model_validate(saved_data)
            except Exception:
                pass

        # Otherwise rebuild from stored metadata + raw response
        raw_resp = saved_data.get("raw_api_response") or saved_data
        proc_time = float(saved_data.get("processing_time_seconds") or 0.0)
        return self._parse_api_response(
            raw_response=raw_resp,
            recording_id=recording_id,
            source_file=audio_path.name,
            processing_time_seconds=proc_time,
        )

    def save_transcript_outputs(
        self,
        transcript: Transcript,
        raw_response: Dict[str, Any],
        recording_id: str,
        source_audio_path: Path,
        processing_time_seconds: float,
        output_dir: Optional[Path] = None,
    ) -> Tuple[Path, Path]:
        """
        Save JSON and TXT outputs atomically to output/transcripts/onprem_whisper/.

        Returns:
            Tuple of (json_path, txt_path).
        """
        out_dir = output_dir or self.settings.onprem_whisper_transcripts_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / f"{recording_id}.json"
        txt_path = out_dir / f"{recording_id}.txt"

        # Prepare comprehensive JSON payload
        full_json_payload = {
            "recording_id": recording_id,
            "source_file": source_audio_path.name,
            "provider": "whisper_api",
            "model": self.model,
            "api_base_url": self.base_url,
            "language": transcript.language,
            "duration": transcript.duration,
            "processing_time_seconds": processing_time_seconds,
            "status": "SUCCESS",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "text": transcript.full_text,
            "segments": [s.model_dump() for s in transcript.segments],
            "raw_api_response": raw_response,
        }

        # Write JSON atomically
        tmp_json = json_path.with_suffix(".tmp")
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(full_json_payload, f, ensure_ascii=False, indent=2)
        tmp_json.replace(json_path)

        # Write TXT atomically
        lines = [
            f"RECORDING: {recording_id}",
            f"SOURCE FILE: {source_audio_path.name}",
            f"DURATION: {transcript.duration:.1f}s",
            f"LANGUAGE: {transcript.language}",
            f"PROVIDER: whisper_api ({self.model})",
            f"PROCESSING TIME: {processing_time_seconds:.2f}s",
            "=" * 60,
            "",
        ]

        if transcript.segments:
            for s in transcript.segments:
                start_m = int(s.start // 60)
                start_s = s.start % 60
                end_m = int(s.end // 60)
                end_s = s.end % 60
                time_str = f"[{start_m:02d}:{start_s:05.2f} -> {end_m:02d}:{end_s:05.2f}]"
                lines.append(f"{time_str} {s.text}")
        else:
            lines.append(transcript.full_text)

        tmp_txt = txt_path.with_suffix(".tmp")
        with open(tmp_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        tmp_txt.replace(txt_path)

        logger.info("Saved Whisper API outputs to %s and %s", json_path, txt_path)
        return json_path, txt_path

