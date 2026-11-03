"""
Custom exceptions for Call Audit Automation Audio Processing Pipeline.
"""


class AudioProcessingError(Exception):
    """Base exception for all audio processing pipeline failures."""
    pass


class FFmpegNotFoundError(AudioProcessingError):
    """
    Raised when FFmpeg or FFprobe binaries cannot be located on the system.
    Provides actionable guidance for installation and configuration.
    """
    def __init__(self, message: str = "FFmpeg/FFprobe binaries not found.") -> None:
        detailed_msg = (
            f"{message}\n\n"
            "Action Required:\n"
            "1. Install FFmpeg on Windows using winget:\n"
            "   winget install \"Gyan.FFmpeg\"\n"
            "2. Or download essentials build from: https://www.gyan.dev/ffmpeg/builds/\n"
            "   and place ffmpeg.exe and ffprobe.exe in the project 'bin/' directory or in your system PATH.\n"
            "3. Or specify custom paths in .env via FFMPEG_PATH and FFPROBE_PATH."
        )
        super().__init__(detailed_msg)


class UnsupportedAudioFormatError(AudioProcessingError):
    """Raised when an audio file has an unsupported format or extension."""
    pass


class AudioMetadataError(AudioProcessingError):
    """Raised when extracting audio metadata fails."""
    pass


class AudioQualityAnalysisError(AudioProcessingError):
    """Raised when audio quality analysis fails."""
    pass


class AudioPreprocessingError(AudioProcessingError):
    """Raised when audio format conversion or preprocessing fails."""
    pass
