"""
Unit Tests for STT Benchmark Module.

Tests adapter creation, schema validation, repetition detection, financial
term scoring, and evaluator summary generation.
All external APIs and model downloads are fully mocked.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.benchmark.schema import ASRResult, BenchmarkSegment
from app.benchmark.repetition import detect_repetition, RepetitionMetrics
from app.benchmark.base import BaseASRAdapter
from app.benchmark.wer import normalize_text_for_evaluation, calculate_wer_cer


# ======================== Schema Tests ========================

class TestBenchmarkSegment:
    def test_create_segment(self):
        seg = BenchmarkSegment(segment_id=0, start=0.0, end=1.5, text="Hello world")
        assert seg.segment_id == 0
        assert seg.text == "Hello world"
        assert seg.start == 0.0
        assert seg.end == 1.5

    def test_segment_optional_fields(self):
        seg = BenchmarkSegment(segment_id=1, text="Test")
        assert seg.start is None
        assert seg.end is None
        assert seg.speaker is None
        assert seg.confidence is None


class TestASRResult:
    def _make_result(self, **overrides):
        defaults = {
            "recording_id": "test_001",
            "source_file": "test_001.wav",
            "input_audio_type": "processed_wav_16khz",
            "model_name": "test-model",
            "model_version": "v1",
            "transcription_mode": "default",
            "transcript": "hello world from test",
            "processing_time_seconds": 1.5,
            "status": "SUCCESS",
        }
        defaults.update(overrides)
        return ASRResult(**defaults)

    def test_create_result(self):
        result = self._make_result()
        assert result.status == "SUCCESS"
        assert result.total_words == 4

    def test_total_words_empty(self):
        result = self._make_result(transcript="")
        assert result.total_words == 0

    def test_save_and_load_json(self):
        result = self._make_result(
            segments=[
                BenchmarkSegment(segment_id=0, start=0.0, end=1.0, text="hello"),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            result.save_json(path)
            assert path.exists()

            loaded = ASRResult.from_json(path)
            assert loaded.recording_id == "test_001"
            assert loaded.model_name == "test-model"
            assert len(loaded.segments) == 1
            assert loaded.segments[0].text == "hello"

    def test_failed_result(self):
        result = self._make_result(
            status="FAILED",
            transcript="",
            error_details="Connection timeout",
        )
        assert result.status == "FAILED"
        assert result.error_details == "Connection timeout"


# ======================== Repetition Detection Tests ========================

class TestRepetitionDetection:
    def test_clean_transcript(self):
        text = "Hello sir, how are you? I am fine. Thank you for calling."
        metrics = detect_repetition(text)
        assert isinstance(metrics, RepetitionMetrics)
        assert metrics.is_repetitive is False
        assert metrics.longest_consecutive_word_streak <= 2

    def test_consecutive_word_streak(self):
        text = "hello hello hello hello hello world"
        metrics = detect_repetition(text)
        assert metrics.is_repetitive is True
        assert metrics.longest_consecutive_word_streak >= 5
        assert len(metrics.flagged_repeated_words) >= 1

    def test_repeated_phrases(self):
        text = " ".join(["aapka kya naam hai"] * 10 + ["shukriya"])
        metrics = detect_repetition(text)
        assert metrics.is_repetitive is True
        assert len(metrics.repeated_phrases) >= 1

    def test_low_diversity(self):
        text = " ".join(["abc"] * 100)
        metrics = detect_repetition(text)
        assert metrics.is_repetitive is True
        assert metrics.unique_to_total_ratio < 0.25

    def test_empty_text(self):
        metrics = detect_repetition("")
        assert metrics.is_repetitive is False
        assert metrics.total_words_count == 0


# ======================== WER/CER Tests ========================

class TestTextNormalization:
    def test_basic_normalization(self):
        assert normalize_text_for_evaluation("Hello, World!") == "hello world"

    def test_hindi_danda_removal(self):
        result = normalize_text_for_evaluation("नमस्ते। कैसे हो?")
        assert "।" not in result
        assert "?" not in result

    def test_empty_string(self):
        assert normalize_text_for_evaluation("") == ""

    def test_whitespace_collapse(self):
        assert normalize_text_for_evaluation("  hello   world  ") == "hello world"


class TestWERCalculation:
    def test_identical_strings(self):
        result = calculate_wer_cer("hello world", "hello world")
        assert result["wer"] == 0.0
        assert result["cer"] == 0.0

    def test_completely_different(self):
        result = calculate_wer_cer("hello", "goodbye")
        assert result["wer"] > 0.0

    def test_empty_reference(self):
        result = calculate_wer_cer("", "some text")
        assert result["wer"] == 1.0

    def test_empty_hypothesis(self):
        result = calculate_wer_cer("some text", "")
        assert result["wer"] > 0.0


# ======================== Base Adapter Tests ========================

class TestBaseASRAdapter:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseASRAdapter(model_name="test", model_version="v1")

    def test_concrete_adapter(self):
        class DummyAdapter(BaseASRAdapter):
            def transcribe(self, audio_path, recording_id, input_audio_type="test", **kwargs):
                return ASRResult(
                    recording_id=recording_id,
                    source_file=str(audio_path),
                    input_audio_type=input_audio_type,
                    model_name=self.model_name,
                    model_version=self.model_version,
                    transcription_mode="default",
                    transcript="test transcript",
                    processing_time_seconds=0.5,
                    status="SUCCESS",
                )

            def calculate_cost(self, duration_seconds):
                return None

        adapter = DummyAdapter(model_name="dummy", model_version="v1")
        assert adapter.model_name == "dummy"
        assert adapter.calculate_cost(60.0) is None

        result = adapter.transcribe(Path("fake.wav"), "rec_001")
        assert result.status == "SUCCESS"


# ======================== Sarvam Adapter Tests (Mocked) ========================

class TestSarvamAdapter:
    def test_missing_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            from app.benchmark.sarvam_adapter import SarvamAdapter
            with pytest.raises(ValueError, match="API key is required"):
                SarvamAdapter(api_key=None)

    def test_cost_calculation(self):
        from app.benchmark.sarvam_adapter import SarvamAdapter
        adapter = SarvamAdapter(api_key="fake-key-12345")
        # 10 minutes = ₹1.50
        cost = adapter.calculate_cost(600.0)
        assert cost == pytest.approx(1.5, abs=0.01)

    @patch("app.benchmark.sarvam_adapter.SarvamAdapter.client", new_callable=lambda: property(lambda self: MagicMock()))
    def test_transcribe_file_not_found(self, mock_client):
        from app.benchmark.sarvam_adapter import SarvamAdapter
        adapter = SarvamAdapter(api_key="fake-key-12345")
        result = adapter.transcribe(Path("nonexistent.wav"), "rec_001")
        assert result.status == "FAILED"
        assert "not found" in result.error_details


# ======================== Whisper Small Adapter Tests ========================

class TestWhisperSmallAdapter:
    def test_load_cached_transcript(self):
        from app.benchmark.whisper_small_adapter import WhisperSmallAdapter

        # Create a mock cached JSON
        mock_transcript = {
            "recording_id": "test_001",
            "source_file": "test_001.wav",
            "language": "hi",
            "language_probability": 0.95,
            "duration": 120.0,
            "model": "small",
            "processing_time_seconds": 45.0,
            "segments": [
                {"segment_id": 0, "start": 0.0, "end": 3.0, "text": "नमस्ते"},
                {"segment_id": 1, "start": 3.5, "end": 6.0, "text": "कैसे हो"},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "test_001.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(mock_transcript, f)

            adapter = WhisperSmallAdapter()
            result = adapter._load_cached_transcript(json_path, "test_001", "processed_wav_16khz")

            assert result.status == "SUCCESS"
            assert result.model_name == "whisper-small"
            assert result.detected_language == "hi"
            assert len(result.segments) == 2
            assert "नमस्ते" in result.transcript


# ======================== Evaluator Tests ========================

class TestEvaluator:
    def test_score_financial_terms(self):
        from app.benchmark.evaluator import score_financial_terms

        text = "I want to open a trading account and invest in mutual fund and SIP."
        keyterms = ["Mutual Fund", "SIP", "Trading Account", "Demat", "Equity"]
        scores = score_financial_terms(text, keyterms)

        assert scores["total_keyterms"] == 5
        assert scores["matched_keyterms"] == 3
        assert "Mutual Fund" in scores["matched_terms"]
        assert "SIP" in scores["matched_terms"]
        assert "Trading Account" in scores["matched_terms"]
        assert scores["coverage_ratio"] == 0.6

    def test_evaluate_single_result(self):
        from app.benchmark.evaluator import evaluate_single_result

        result = ASRResult(
            recording_id="test_001",
            source_file="test_001.wav",
            input_audio_type="processed_wav_16khz",
            model_name="test-model",
            model_version="v1",
            transcription_mode="default",
            transcript="Hello this is a test transcript for evaluation",
            processing_time_seconds=2.5,
            status="SUCCESS",
        )
        ev = evaluate_single_result(result)

        assert ev["recording_id"] == "test_001"
        assert ev["model_name"] == "test-model"
        assert ev["word_count"] > 0
        assert "repetition" in ev
        assert "financial_terms" in ev
        assert ev["repetition"]["is_repetitive"] is False

    def test_benchmark_runner_summary(self):
        from app.benchmark.evaluator import BenchmarkRunner

        runner = BenchmarkRunner()
        runner.results = [
            ASRResult(
                recording_id="rec_001",
                source_file="rec_001.wav",
                input_audio_type="processed_wav_16khz",
                model_name="model-a",
                model_version="v1",
                transcription_mode="default",
                transcript="Hello world test",
                processing_time_seconds=3.0,
                status="SUCCESS",
            ),
            ASRResult(
                recording_id="rec_001",
                source_file="rec_001.wav",
                input_audio_type="processed_wav_16khz",
                model_name="model-b",
                model_version="v2",
                transcription_mode="codemix",
                transcript="Hello duniya test",
                processing_time_seconds=5.0,
                status="SUCCESS",
                api_cost_inr=0.15,
            ),
        ]
        runner.evaluate_all()
        summary = runner.generate_summary()

        assert summary["total_runs"] == 2
        assert "model-a_default" in summary["results_by_model"]
        assert "model-b_codemix" in summary["results_by_model"]

    def test_benchmark_runner_save_reports(self):
        from app.benchmark.evaluator import BenchmarkRunner

        runner = BenchmarkRunner()
        runner.results = [
            ASRResult(
                recording_id="rec_001",
                source_file="rec_001.wav",
                input_audio_type="processed_wav_16khz",
                model_name="test-model",
                model_version="v1",
                transcription_mode="default",
                transcript="sample transcript",
                processing_time_seconds=1.0,
                status="SUCCESS",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            # Mock settings to use temp dir
            with patch("app.benchmark.evaluator.get_settings") as mock_settings:
                mock_s = MagicMock()
                mock_s.stt_benchmark_dir = Path(tmp) / "benchmark"
                mock_s.financial_keyterms_path = Path(tmp) / "nonexistent.json"
                mock_settings.return_value = mock_s

                runner.evaluate_all()
                saved = runner.save_reports()

                assert "summary_json" in saved
                assert saved["summary_json"].exists()
                assert "summary_csv" in saved
                assert saved["summary_csv"].exists()

                # Validate JSON content
                with open(saved["summary_json"], "r") as f:
                    data = json.load(f)
                assert data["total_runs"] == 1


# ======================== Integration-like Test (all mocked) ========================

class TestEndToEndMocked:
    """Simulate a full benchmark flow with mocked adapters."""

    def test_full_flow(self):
        from app.benchmark.evaluator import BenchmarkRunner

        runner = BenchmarkRunner()

        # Simulate results from 3 different models
        models_data = [
            ("whisper-small", "small", "default", "Hello this is test one", 30.0, None),
            ("sarvam-saaras-v4", "saaras:v4", "codemix", "Hello yeh test ek hai", 8.5, 0.075),
            ("whisper-large-v3", "large-v3", "default", "Hello this is test number one", 120.0, None),
        ]

        for model_name, version, mode, text, proc_time, cost in models_data:
            runner.results.append(ASRResult(
                recording_id="rec_001",
                source_file="rec_001.wav",
                input_audio_type="processed_wav_16khz",
                model_name=model_name,
                model_version=version,
                transcription_mode=mode,
                transcript=text,
                processing_time_seconds=proc_time,
                api_cost_inr=cost,
                status="SUCCESS",
                segments=[
                    BenchmarkSegment(segment_id=0, start=0.0, end=5.0, text=text),
                ],
            ))

        evals = runner.evaluate_all()
        assert len(evals) == 3

        summary = runner.generate_summary()
        assert summary["total_runs"] == 3
        assert len(summary["models_tested"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
