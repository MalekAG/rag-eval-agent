"""
Unit tests for RAG evaluation system.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "execution"))

from config import EvalConfig, load_config, validate_config, WeightsConfig
from metrics import MetricsCalculator, calculate_composite_score, MetricScores
from report import ReportGenerator, TestResult, EvaluationReport
from adapters.http_adapter import HTTPAdapter


class TestConfig:
    """Tests for configuration loading and validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EvalConfig()
        assert config.adapter == "http"
        assert config.concurrency == 1
        assert config.weights.faithfulness == 40.0

    def test_weights_normalization(self):
        """Test that weights normalize to 1.0."""
        weights = WeightsConfig(
            faithfulness=40,
            answer_relevance=20,
            context_precision=20,
            context_recall=20,
        )
        normalized = weights.normalized()
        assert abs(sum(normalized.values()) - 1.0) < 0.001

    def test_validate_config_invalid_adapter(self):
        """Test validation fails for invalid adapter."""
        config = EvalConfig(adapter="invalid")
        with pytest.raises(ValueError, match="Invalid adapter"):
            validate_config(config)

    def test_validate_config_missing_endpoint(self):
        """Test validation fails when HTTP adapter has no endpoint."""
        config = EvalConfig(adapter="http", endpoint=None)
        with pytest.raises(ValueError, match="requires an endpoint"):
            validate_config(config)

    def test_validate_config_valid(self):
        """Test validation passes for valid config."""
        config = EvalConfig(adapter="http", endpoint="http://localhost:8000")
        warnings = validate_config(config)
        assert isinstance(warnings, list)


class TestHTTPAdapter:
    """Tests for HTTP adapter."""

    def test_adapter_initialization(self):
        """Test adapter initializes correctly."""
        adapter = HTTPAdapter(
            endpoint="http://localhost:8000/query",
            headers={"Authorization": "Bearer test"},
            timeout=30,
            allow_internal=True,  # Allow localhost for testing
        )
        assert adapter.endpoint == "http://localhost:8000/query"
        assert "Authorization" in adapter.headers
        assert adapter.timeout == 30

    @patch("adapters.http_adapter.requests.post")
    def test_query_success(self, mock_post):
        """Test successful query."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "answer": "The refund policy is 30 days.",
            "contexts": ["Policy doc content"],
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        adapter = HTTPAdapter(endpoint="http://localhost:8000/query", allow_internal=True)
        answer, contexts = adapter.query("What is the refund policy?")

        assert answer == "The refund policy is 30 days."
        assert contexts == ["Policy doc content"]

    @patch("adapters.http_adapter.requests.post")
    def test_query_timeout(self, mock_post):
        """Test query timeout handling."""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout()

        adapter = HTTPAdapter(endpoint="http://localhost:8000/query", allow_internal=True)
        with pytest.raises(TimeoutError):
            adapter.query("What is the refund policy?")


class TestMetrics:
    """Tests for metrics calculation."""

    def test_composite_score_calculation(self):
        """Test composite score calculation."""
        scores = MetricScores(
            faithfulness=0.9,
            answer_relevance=0.8,
            context_precision=0.7,
            context_recall=0.6,
            semantic_similarity=0.85,
            judge_reasoning="Good",
            unsupported_claims=[],
        )
        weights = {
            "faithfulness": 0.4,
            "answer_relevance": 0.2,
            "context_precision": 0.2,
            "context_recall": 0.2,
        }
        composite = calculate_composite_score(scores, weights)

        expected = 0.9 * 0.4 + 0.8 * 0.2 + 0.7 * 0.2 + 0.6 * 0.2
        assert abs(composite - expected) < 0.001

    def test_composite_score_missing_context_metrics(self):
        """Test composite calculation when context metrics are None."""
        scores = MetricScores(
            faithfulness=0.9,
            answer_relevance=0.8,
            context_precision=None,
            context_recall=None,
            semantic_similarity=0.85,
            judge_reasoning="Good",
            unsupported_claims=[],
        )
        weights = {
            "faithfulness": 0.4,
            "answer_relevance": 0.2,
            "context_precision": 0.2,
            "context_recall": 0.2,
        }
        composite = calculate_composite_score(scores, weights)

        # Should only use available metrics, renormalized
        expected = (0.9 * 0.4 + 0.8 * 0.2) / (0.4 + 0.2)
        assert abs(composite - expected) < 0.001


class TestReport:
    """Tests for report generation."""

    def test_report_generator_initialization(self, tmp_path, monkeypatch):
        """Test report generator creates output directory."""
        # Allow tmp_path for testing by monkeypatching ALLOWED_OUTPUT_DIRS
        import report
        monkeypatch.setattr(report, "ALLOWED_OUTPUT_DIRS", [tmp_path, tmp_path / "results"])

        output_dir = tmp_path / "results"
        gen = ReportGenerator(str(output_dir))
        assert output_dir.exists()

    def test_markdown_report_generation(self, tmp_path, monkeypatch):
        """Test markdown report generation."""
        # Allow tmp_path for testing
        import report
        monkeypatch.setattr(report, "ALLOWED_OUTPUT_DIRS", [tmp_path])

        gen = ReportGenerator(str(tmp_path))

        results = [
            TestResult(
                id="tc_001",
                question="Test question?",
                ground_truth="Expected answer",
                answer="Generated answer",
                contexts=["context1"],
                faithfulness=0.9,
                answer_relevance=0.8,
                context_precision=0.7,
                context_recall=0.6,
                semantic_similarity=0.85,
                composite_score=0.78,
                latency_ms=150,
                passed=True,
                critical=False,
                tags=["test"],
                judge_reasoning="Good answer",
                unsupported_claims=[],
            ),
        ]

        summary = gen.generate_summary(
            results=results,
            adapter_name="http",
            endpoint="http://localhost:8000",
            duration_seconds=10.5,
            estimated_cost=0.10,
            actual_cost=0.08,
            thresholds={"composite": 0.7},
        )

        report = EvaluationReport(
            summary=summary,
            results=results,
            thresholds={"composite": 0.7},
            weights={"faithfulness": 0.4, "answer_relevance": 0.2},
        )

        md = gen.generate_markdown(report)
        assert "# RAG Evaluation Report" in md
        assert "http" in md
        assert "0.78" in md or "0.9" in md  # Score present


class TestDatasetLoading:
    """Tests for dataset loading and validation."""

    def test_load_valid_dataset(self, tmp_path, sample_dataset, monkeypatch):
        """Test loading a valid dataset."""
        # Allow tmp_path for testing
        import evaluator
        monkeypatch.setattr(evaluator, "ALLOWED_DATASET_DIRS", [tmp_path])

        dataset_path = tmp_path / "test_dataset.json"
        dataset_path.write_text(json.dumps(sample_dataset))

        # Import here to avoid path issues
        from evaluator import load_dataset
        loaded = load_dataset(str(dataset_path))

        assert len(loaded["test_cases"]) == 2
        assert loaded["test_cases"][0]["critical"] is True

    def test_load_missing_dataset(self, tmp_path, monkeypatch):
        """Test error on missing dataset."""
        # Allow tmp_path for testing
        import evaluator
        monkeypatch.setattr(evaluator, "ALLOWED_DATASET_DIRS", [tmp_path])

        from evaluator import load_dataset
        with pytest.raises(FileNotFoundError):
            load_dataset(str(tmp_path / "nonexistent.json"))

    def test_load_invalid_dataset(self, tmp_path, monkeypatch):
        """Test error on dataset without test_cases."""
        # Allow tmp_path for testing
        import evaluator
        monkeypatch.setattr(evaluator, "ALLOWED_DATASET_DIRS", [tmp_path])

        dataset_path = tmp_path / "invalid.json"
        dataset_path.write_text(json.dumps({"data": []}))

        from evaluator import load_dataset
        with pytest.raises(ValueError, match="test_cases"):
            load_dataset(str(dataset_path))


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_dry_run(self, tmp_path, sample_dataset, monkeypatch):
        """Test dry run mode validates without executing."""
        # Allow tmp_path for testing
        import evaluator
        monkeypatch.setattr(evaluator, "ALLOWED_DATASET_DIRS", [tmp_path])

        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text(json.dumps(sample_dataset))

        from evaluator import main

        # Mock sys.argv
        monkeypatch.setattr(
            "sys.argv",
            [
                "evaluator.py",
                "--adapter", "http",
                "--endpoint", "http://localhost:8000",
                "--dataset", str(dataset_path),
                "--dry-run",
            ],
        )

        exit_code = main()
        assert exit_code == 0
