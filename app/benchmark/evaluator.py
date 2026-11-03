"""
STT Benchmark Evaluator and Orchestrator.

Orchestrates benchmark runs across all configured ASR models, collects
normalized results, runs repetition detection and financial term analysis,
and generates consolidated comparison reports.
"""

import csv
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.benchmark.repetition import detect_repetition
from app.benchmark.schema import ASRResult
from app.core.config import get_settings

logger = logging.getLogger("call_audit.benchmark.evaluator")


def _load_financial_keyterms() -> List[str]:
    """Load financial keyterms from config for terminology coverage scoring."""
    settings = get_settings()
    path = settings.financial_keyterms_path
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("keyterms", [])
    except Exception:
        return []


def score_financial_terms(transcript: str, keyterms: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Count how many financial keyterms appear in a transcript.
    Case-insensitive matching.
    """
    if keyterms is None:
        keyterms = _load_financial_keyterms()

    text_lower = transcript.lower()
    found = {}
    for term in keyterms:
        term_lower = term.lower()
        count = text_lower.count(term_lower)
        if count > 0:
            found[term] = count

    return {
        "total_keyterms": len(keyterms),
        "matched_keyterms": len(found),
        "matched_terms": found,
        "coverage_ratio": round(len(found) / max(len(keyterms), 1), 4),
    }


def evaluate_single_result(result: ASRResult) -> Dict[str, Any]:
    """
    Evaluate a single ASRResult for quality metrics.
    Returns evaluation metrics dict (does NOT modify the result).
    """
    evaluation = {
        "recording_id": result.recording_id,
        "model_name": result.model_name,
        "transcription_mode": result.transcription_mode,
        "status": result.status,
        "word_count": result.total_words,
        "segment_count": len(result.segments),
        "has_timestamps": any(s.start is not None for s in result.segments) if result.segments else False,
        "processing_time_seconds": result.processing_time_seconds,
        "api_cost_inr": result.api_cost_inr,
        "detected_language": result.detected_language,
        "language_probability": result.language_probability,
    }

    # Repetition analysis
    if result.transcript:
        rep = detect_repetition(result.transcript)
        evaluation["repetition"] = {
            "is_repetitive": rep.is_repetitive,
            "unique_to_total_ratio": rep.unique_to_total_ratio,
            "longest_streak": rep.longest_consecutive_word_streak,
            "flagged_words_count": len(rep.flagged_repeated_words),
            "repeated_phrases_count": len(rep.repeated_phrases),
            "warnings": rep.warnings,
        }
    else:
        evaluation["repetition"] = {"is_repetitive": False, "warnings": ["Empty transcript"]}

    # Financial term coverage
    if result.transcript:
        evaluation["financial_terms"] = score_financial_terms(result.transcript)
    else:
        evaluation["financial_terms"] = {"total_keyterms": 0, "matched_keyterms": 0, "coverage_ratio": 0.0}

    # Warnings and errors
    evaluation["warnings"] = result.warnings
    evaluation["error_details"] = result.error_details

    return evaluation


class BenchmarkRunner:
    """
    Orchestrates STT benchmark runs across multiple models and recordings.
    """

    def __init__(self):
        self.settings = get_settings()
        self.results: List[ASRResult] = []
        self.evaluations: List[Dict[str, Any]] = []

    def discover_recordings(self, audio_type: str = "wav") -> List[Path]:
        """
        Discover audio recordings for benchmarking.

        Args:
            audio_type: "wav" for processed 16kHz WAVs, "mp3" for original 8kHz MP3s.
        """
        if audio_type == "wav":
            source_dir = self.settings.processed_audio_dir
            pattern = "*.wav"
        else:
            source_dir = self.settings.recordings_dir
            pattern = "*.mp3"

        files = sorted(source_dir.glob(pattern))
        files = [f for f in files if f.name != ".gitkeep"]
        logger.info("Discovered %d %s recordings in %s", len(files), audio_type.upper(), source_dir)
        return files

    def run_model(
        self,
        model_key: str,
        audio_files: List[Path],
        audio_type: str = "wav",
        reuse: bool = False,
        **adapter_kwargs,
    ) -> List[ASRResult]:
        """
        Run a single model across all recordings.

        Args:
            model_key: One of "baseline", "whisper-large", "sarvam-codemix", "sarvam-verbatim".
            audio_files: List of audio file paths to transcribe.
            audio_type: "wav" or "mp3".
            reuse: If True, skip files with existing benchmark results.
            **adapter_kwargs: Extra kwargs forwarded to the adapter.
        """
        input_type = "processed_wav_16khz" if audio_type == "wav" else "original_mp3_8khz"
        adapter = self._create_adapter(model_key, **adapter_kwargs)
        output_subdir = self._get_output_subdir(model_key)
        output_subdir.mkdir(parents=True, exist_ok=True)

        results = []
        for audio_path in audio_files:
            recording_id = audio_path.stem
            result_path = output_subdir / f"{recording_id}.json"

            # Reuse existing result if requested
            if reuse and result_path.exists():
                try:
                    existing = ASRResult.from_json(result_path)
                    if existing.status == "SUCCESS":
                        logger.info("Reusing cached result for %s/%s", model_key, recording_id)
                        results.append(existing)
                        continue
                except Exception:
                    pass  # Re-run if cached file is corrupt

            logger.info("Transcribing %s with %s...", recording_id, model_key)
            result = adapter.transcribe(
                audio_path=audio_path,
                recording_id=recording_id,
                input_audio_type=input_type,
                **adapter_kwargs,
            )

            # Save individual result
            result.save_json(result_path)
            results.append(result)
            logger.info(
                "Completed %s/%s: status=%s, words=%d, time=%.2fs",
                model_key, recording_id, result.status,
                result.total_words, result.processing_time_seconds,
            )

        self.results.extend(results)
        return results

    def _create_adapter(self, model_key: str, **kwargs):
        """Instantiate the appropriate adapter for a model key."""
        if model_key == "baseline":
            from app.benchmark.whisper_small_adapter import WhisperSmallAdapter
            return WhisperSmallAdapter()

        elif model_key == "whisper-large":
            from app.benchmark.whisper_large_adapter import WhisperLargeAdapter
            return WhisperLargeAdapter(
                beam_size=kwargs.pop("beam_size", 1),
                vad_filter=kwargs.pop("vad_filter", True),
            )

        elif model_key.startswith("sarvam"):
            from app.benchmark.sarvam_adapter import SarvamAdapter
            mode = "verbatim" if "verbatim" in model_key else "codemix"
            api_key = kwargs.pop("api_key", None) or self.settings.SARVAM_API_KEY
            model = kwargs.pop("model", None) or self.settings.SARVAM_MODEL or "saaras:v4"
            return SarvamAdapter(
                api_key=api_key,
                model=model,
                mode=mode,
            )

        else:
            raise ValueError(f"Unknown model key: {model_key}")

    def _get_output_subdir(self, model_key: str) -> Path:
        """Map model key to output subdirectory."""
        mapping = {
            "baseline": "whisper_small_baseline",
            "whisper-large": "whisper_large_v3",
            "sarvam-codemix": "sarvam_v4_codemix",
            "sarvam-verbatim": "sarvam_v4_verbatim",
        }
        subdir_name = mapping.get(model_key, model_key.replace("-", "_"))
        return self.settings.stt_benchmark_dir / subdir_name

    def load_all_cached_results(self) -> List[ASRResult]:
        """
        Discover and load all cached ASRResult JSON files from all benchmark subdirectories.
        """
        benchmark_dir = self.settings.stt_benchmark_dir
        self.results = []
        for subdir in sorted(benchmark_dir.iterdir()):
            if subdir.is_dir() and not subdir.name.startswith("."):
                for json_file in sorted(subdir.glob("*.json")):
                    if json_file.name in ("benchmark_summary.json", "benchmark_evaluations.json"):
                        continue
                    try:
                        res = ASRResult.from_json(json_file)
                        self.results.append(res)
                    except Exception as exc:
                        logger.warning("Failed to load result from %s: %s", json_file, exc)
        logger.info("Loaded %d cached benchmark results across all models.", len(self.results))
        return self.results

    def evaluate_all(self) -> List[Dict[str, Any]]:
        """Run evaluation metrics on all collected results."""
        self.evaluations = [evaluate_single_result(r) for r in self.results]
        return self.evaluations

    def generate_summary(self) -> Dict[str, Any]:
        """Generate consolidated benchmark summary."""
        if not self.evaluations:
            self.evaluate_all()

        summary = {
            "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_runs": len(self.results),
            "models_tested": list(set(r.model_name for r in self.results)),
            "recordings_tested": list(set(r.recording_id for r in self.results)),
            "results_by_model": {},
        }

        # Group by model+mode
        groups: Dict[str, List[int]] = {}
        for i, r in enumerate(self.results):
            key = f"{r.model_name}_{r.transcription_mode}"
            groups.setdefault(key, []).append(i)

        for group_key, indices in groups.items():
            group_results = [self.results[i] for i in indices]
            group_evals = [self.evaluations[i] for i in indices]

            successful = [r for r in group_results if r.status == "SUCCESS"]
            failed = [r for r in group_results if r.status == "FAILED"]

            avg_time = (
                sum(r.processing_time_seconds for r in successful) / len(successful)
                if successful else 0.0
            )
            total_cost = sum(r.api_cost_inr or 0.0 for r in successful)
            avg_words = (
                sum(r.total_words for r in successful) / len(successful)
                if successful else 0
            )

            # Repetition summary
            repetitive_count = sum(
                1 for e in group_evals
                if e.get("repetition", {}).get("is_repetitive", False)
            )

            # Financial term coverage
            avg_coverage = 0.0
            coverages = [
                e.get("financial_terms", {}).get("coverage_ratio", 0.0) for e in group_evals
            ]
            if coverages:
                avg_coverage = round(sum(coverages) / len(coverages), 4)

            summary["results_by_model"][group_key] = {
                "model_name": group_results[0].model_name,
                "transcription_mode": group_results[0].transcription_mode,
                "total_recordings": len(group_results),
                "successful": len(successful),
                "failed": len(failed),
                "avg_processing_time_seconds": round(avg_time, 2),
                "total_api_cost_inr": round(total_cost, 4),
                "avg_word_count": round(avg_words, 1),
                "repetitive_transcripts": repetitive_count,
                "avg_financial_term_coverage": avg_coverage,
                "has_timestamps": any(
                    e.get("has_timestamps", False) for e in group_evals
                ),
            }

        return summary

    def save_reports(self) -> Dict[str, Path]:
        """Save benchmark summary JSON and CSV reports."""
        benchmark_dir = self.settings.stt_benchmark_dir
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        summary = self.generate_summary()
        saved_files = {}

        # JSON summary
        json_path = benchmark_dir / "benchmark_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        saved_files["summary_json"] = json_path
        logger.info("Saved benchmark summary: %s", json_path)

        # Detailed evaluations JSON
        evals_path = benchmark_dir / "benchmark_evaluations.json"
        with open(evals_path, "w", encoding="utf-8") as f:
            json.dump(self.evaluations, f, indent=2, ensure_ascii=False)
        saved_files["evaluations_json"] = evals_path

        # CSV summary
        csv_path = benchmark_dir / "benchmark_summary.csv"
        self._write_csv(csv_path)
        saved_files["summary_csv"] = csv_path
        logger.info("Saved benchmark CSV: %s", csv_path)

        return saved_files

    def _write_csv(self, csv_path: Path) -> None:
        """Write per-recording CSV comparison table."""
        if not self.results:
            return

        fieldnames = [
            "recording_id", "model_name", "transcription_mode", "status",
            "word_count", "segment_count", "has_timestamps",
            "processing_time_seconds", "api_cost_inr",
            "detected_language", "language_probability",
            "is_repetitive", "unique_to_total_ratio",
            "financial_terms_matched", "financial_term_coverage",
            "warnings_count", "error_details",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i, result in enumerate(self.results):
                ev = self.evaluations[i] if i < len(self.evaluations) else {}
                rep = ev.get("repetition", {})
                fin = ev.get("financial_terms", {})

                writer.writerow({
                    "recording_id": result.recording_id,
                    "model_name": result.model_name,
                    "transcription_mode": result.transcription_mode,
                    "status": result.status,
                    "word_count": result.total_words,
                    "segment_count": len(result.segments),
                    "has_timestamps": ev.get("has_timestamps", False),
                    "processing_time_seconds": result.processing_time_seconds,
                    "api_cost_inr": result.api_cost_inr or "",
                    "detected_language": result.detected_language or "",
                    "language_probability": result.language_probability or "",
                    "is_repetitive": rep.get("is_repetitive", False),
                    "unique_to_total_ratio": rep.get("unique_to_total_ratio", ""),
                    "financial_terms_matched": fin.get("matched_keyterms", 0),
                    "financial_term_coverage": fin.get("coverage_ratio", 0.0),
                    "warnings_count": len(result.warnings),
                    "error_details": result.error_details or "",
                })
