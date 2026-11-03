"""
FFmpeg and FFprobe binary locator and verification utilities.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from app.audio.exceptions import FFmpegNotFoundError
from app.core.config import get_settings

logger = logging.getLogger("call_audit.audio.ffmpeg")


def find_binary(binary_name: str, configured_path: Optional[str] = None) -> Path:
    """
    Locate an executable binary (ffmpeg or ffprobe).

    Lookup priority:
    1. Explicitly configured path from environment or settings.
    2. Project local 'bin/' directory (<project_root>/bin/<name>.exe).
    3. System PATH.

    Args:
        binary_name: Name of the binary without extension ('ffmpeg' or 'ffprobe').
        configured_path: Optional explicit path string from settings.

    Returns:
        Path to the resolved executable.

    Raises:
        FFmpegNotFoundError: If the binary could not be found or executed.
    """
    # 1. Check explicit configuration
    if configured_path:
        path = Path(configured_path).resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
        # Also check with .exe on Windows
        exe_path = path.with_suffix(".exe") if not path.suffix else path
        if exe_path.is_file():
            return exe_path

    # 2. Check project local bin directory
    settings = get_settings()
    local_bin = settings.bin_dir
    for candidate in [
        local_bin / f"{binary_name}.exe",
        local_bin / binary_name,
    ]:
        if candidate.is_file():
            return candidate

    # 3. Check system PATH
    which_path = shutil.which(binary_name)
    if which_path:
        return Path(which_path).resolve()

    # Not found
    raise FFmpegNotFoundError(f"Could not locate '{binary_name}' binary.")


def get_ffmpeg_path() -> Path:
    """
    Get the absolute path to the FFmpeg executable.
    """
    settings = get_settings()
    return find_binary("ffmpeg", settings.FFMPEG_PATH)


def get_ffprobe_path() -> Path:
    """
    Get the absolute path to the FFprobe executable.
    """
    settings = get_settings()
    return find_binary("ffprobe", settings.FFPROBE_PATH)


def verify_ffmpeg_installation() -> dict[str, str]:
    """
    Verify that both FFmpeg and FFprobe are accessible and can execute.

    Returns:
        Dictionary mapping binary names to resolved paths and version summaries.

    Raises:
        FFmpegNotFoundError: If either binary is missing or non-functional.
    """
    ffmpeg_path = get_ffmpeg_path()
    ffprobe_path = get_ffprobe_path()

    results: dict[str, str] = {}

    for name, path in [("ffmpeg", ffmpeg_path), ("ffprobe", ffprobe_path)]:
        try:
            completed = subprocess.run(
                [str(path), "-version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            first_line = completed.stdout.splitlines()[0] if completed.stdout else "version unknown"
            results[name] = f"{path} ({first_line})"
            logger.info("Found %s: %s", name, results[name])
        except (subprocess.SubprocessError, OSError) as exc:
            raise FFmpegNotFoundError(
                f"Binary '{name}' found at '{path}' but failed to execute: {exc}"
            ) from exc

    return results
