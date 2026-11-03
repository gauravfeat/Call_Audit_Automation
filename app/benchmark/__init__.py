"""
STT Benchmark Module.

Provides a common ASR interface, adapters for Sarvam Saaras v4 and
Faster-Whisper (Small and Large-v3), plus evaluation and reporting tools.
"""

from app.benchmark.schema import ASRResult, BenchmarkSegment
from app.benchmark.base import BaseASRAdapter
from app.benchmark.repetition import detect_repetition, RepetitionMetrics
from app.benchmark.evaluator import BenchmarkRunner, evaluate_single_result

__all__ = [
    "ASRResult",
    "BenchmarkSegment",
    "BaseASRAdapter",
    "detect_repetition",
    "RepetitionMetrics",
    "BenchmarkRunner",
    "evaluate_single_result",
]
