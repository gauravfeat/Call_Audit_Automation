"""
Application Configuration Module.

Loads environment variables using Pydantic Settings and defines project paths
using pathlib to maintain operating system independence.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    """

    # Application metadata
    APP_NAME: str = Field(default="Call Audit Automation", description="Application service name")
    APP_ENV: str = Field(default="development", description="Environment: development, staging, production")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")

    # Server configuration
    HOST: str = Field(default="127.0.0.1", description="Server bind host")
    PORT: int = Field(default=8000, description="Server bind port")
    RELOAD: bool = Field(default=True, description="Enable auto-reload for local development")

    # API credentials (optional for initial setup; required for AI pipeline stages)
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="Google Gemini API key")
    GEMINI_MODEL: str = Field(default="gemini-2.5-flash", description="Google Gemini model identifier")
    ROLE_CLASSIFICATION_CONFIDENCE_THRESHOLD: float = Field(
        default=0.85, description="Confidence threshold for automatic role classification"
    )
    HUGGINGFACE_TOKEN: Optional[str] = Field(default=None, description="Hugging Face User Access Token")

    # Optional explicit FFmpeg/FFprobe binary paths (auto-detected if None)
    FFMPEG_PATH: Optional[str] = Field(default=None, description="Explicit path to ffmpeg executable")
    FFPROBE_PATH: Optional[str] = Field(default=None, description="Explicit path to ffprobe executable")

    # Faster-Whisper Speech-to-Text configuration (Baseline Small)
    WHISPER_MODEL: str = Field(default="small", description="Whisper model size: tiny, base, small, medium, large-v3")
    WHISPER_DEVICE: str = Field(default="cpu", description="Inference device: cpu or cuda")
    WHISPER_COMPUTE_TYPE: str = Field(default="int8", description="Compute type: int8, float16, float32, bfloat16")
    WHISPER_BEAM_SIZE: int = Field(default=5, description="Beam search width for decoding")
    WHISPER_VAD_FILTER: bool = Field(default=True, description="Enable Silero VAD filter to skip silence")
    WHISPER_WORD_TIMESTAMPS: bool = Field(default=True, description="Extract word-level timestamps")

    # Sarvam AI Saaras v4 Speech-to-Text configuration (Primary STT Provider)
    DEFAULT_STT_PROVIDER: str = Field(default="sarvam", description="Default STT provider: sarvam or whisper")
    SARVAM_API_KEY: Optional[str] = Field(default=None, description="Sarvam AI API subscription key")
    SARVAM_MODEL: str = Field(default="saaras:v4", description="Sarvam model (saaras:v4)")
    SARVAM_MODE: str = Field(default="codemix", description="Sarvam mode: codemix or verbatim")

    # Faster-Whisper Large-v3 configuration
    WHISPER_LARGE_MODEL: str = Field(default="large-v3", description="Whisper Large model identifier")
    WHISPER_LARGE_DEVICE: str = Field(default="auto", description="Whisper Large device: auto, cpu, cuda")
    WHISPER_LARGE_COMPUTE_TYPE: str = Field(default="auto", description="Whisper Large compute type: auto, int8, float16")

    # Base and subproject paths (derived dynamically via pathlib)
    BASE_DIR: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent,
        description="Project root directory",
    )

    @property
    def bin_dir(self) -> Path:
        """Directory for local helper binaries (e.g., ffmpeg, ffprobe)."""
        return self.BASE_DIR / "bin"

    @property
    def input_dir(self) -> Path:
        """Directory for incoming audio recordings and client Excel reports."""
        return self.BASE_DIR / "input"

    @property
    def recordings_dir(self) -> Path:
        """Directory for mock and local audio files."""
        return self.input_dir / "recordings"

    @property
    def excel_dir(self) -> Path:
        """Directory for raw client observation Excel sheets."""
        return self.input_dir / "excel"

    @property
    def output_dir(self) -> Path:
        """Root output directory for audit run artifacts."""
        return self.BASE_DIR / "output"

    @property
    def processed_audio_dir(self) -> Path:
        """Directory for standardized preprocessed audio recordings (16kHz mono WAV)."""
        return self.output_dir / "processed_audio"

    @property
    def audio_reports_dir(self) -> Path:
        """Directory for audio quality reports and summaries."""
        return self.output_dir / "audio_reports"

    @property
    def transcripts_dir(self) -> Path:
        """Directory for STT and diarized transcript outputs."""
        return self.output_dir / "transcripts"

    @property
    def primary_transcripts_dir(self) -> Path:
        """Directory for primary STT transcripts (Sarvam Saaras v4 Codemix)."""
        return self.transcripts_dir / "sarvam_v4_codemix"

    @property
    def stt_benchmark_dir(self) -> Path:
        """Directory for multi-model STT benchmarking outputs."""
        return self.output_dir / "stt_benchmark"

    @property
    def diarized_transcripts_dir(self) -> Path:
        """Directory for speaker-diarized transcript outputs."""
        return self.output_dir / "diarized_transcripts"

    @property
    def role_enriched_transcripts_dir(self) -> Path:
        """Directory for role-enriched diarized transcript outputs."""
        return self.diarized_transcripts_dir / "role_enriched"

    @property
    def ai_role_classifications_dir(self) -> Path:
        """Directory for AI-predicted role classification outputs."""
        return self.diarized_transcripts_dir / "ai_role_classifications"

    @property
    def validation_results_dir(self) -> Path:
        """Directory for KPI validation results and false-positive flags."""
        return self.output_dir / "validation_results"

    @property
    def reports_dir(self) -> Path:
        """Directory for final consolidated audit reports."""
        return self.output_dir / "reports"

    @property
    def config_dir(self) -> Path:
        """Directory for system configuration and KPI condition definitions."""
        return self.BASE_DIR / "config"

    @property
    def kpi_conditions_path(self) -> Path:
        """Path to KPI conditions JSON specification."""
        return self.config_dir / "kpi_conditions.json"

    @property
    def financial_keyterms_path(self) -> Path:
        """Path to financial keyterms vocabulary JSON specification."""
        return self.config_dir / "financial_keyterms.json"

    @property
    def logs_dir(self) -> Path:
        """Directory for application log files."""
        return self.BASE_DIR / "logs"

    def ensure_directories(self) -> None:
        """
        Ensure all critical working directories exist on filesystem.
        """
        for directory in [
            self.recordings_dir,
            self.excel_dir,
            self.processed_audio_dir,
            self.audio_reports_dir,
            self.transcripts_dir,
            self.primary_transcripts_dir,
            self.stt_benchmark_dir,
            self.diarized_transcripts_dir,
            self.role_enriched_transcripts_dir,
            self.ai_role_classifications_dir,
            self.validation_results_dir,
            self.reports_dir,
            self.config_dir,
            self.logs_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings instance.
    """
    settings = Settings()
    settings.ensure_directories()
    return settings

# Convenient global settings instance
settings: Settings = get_settings()

__all__ = ["Settings", "get_settings", "settings"]
