"""
Configuration loading and validation for RAG evaluation.

Supports YAML config files with environment variable substitution.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class HTTPConfig:
    """HTTP adapter configuration."""
    headers: dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    slow_threshold: int = 5


@dataclass
class RetryConfig:
    """Retry behavior configuration."""
    max_attempts: int = 3
    backoff: str = "exponential"  # "exponential" or "fixed"
    base_delay: float = 1.0


@dataclass
class WeightsConfig:
    """Metric weights configuration (percentages, normalized to 1.0)."""
    faithfulness: float = 40.0
    answer_relevance: float = 20.0
    context_precision: float = 20.0
    context_recall: float = 20.0

    def normalized(self) -> dict[str, float]:
        """Return weights normalized to sum to 1.0."""
        total = self.faithfulness + self.answer_relevance + self.context_precision + self.context_recall
        if total == 0:
            total = 1
        return {
            "faithfulness": self.faithfulness / total,
            "answer_relevance": self.answer_relevance / total,
            "context_precision": self.context_precision / total,
            "context_recall": self.context_recall / total,
        }


@dataclass
class ThresholdsConfig:
    """Score thresholds for CI integration."""
    composite: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None


@dataclass
class ComparisonConfig:
    """Answer comparison configuration."""
    semantic_similarity_threshold: float = 0.85
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass
class OutputConfig:
    """Output configuration."""
    directory: str = "results"
    formats: list[str] = field(default_factory=lambda: ["markdown", "json"])


@dataclass
class EvalConfig:
    """Complete evaluation configuration."""
    adapter: str = "http"
    endpoint: Optional[str] = None
    dataset: Optional[str] = None
    concurrency: int = 1
    http: HTTPConfig = field(default_factory=HTTPConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # CLI overrides (populated from argparse)
    verbose: bool = False
    quiet: bool = False
    dry_run: bool = False


def substitute_env_vars(value: str) -> str:
    """Replace ${VAR} patterns with environment variable values."""
    pattern = r'\$\{([^}]+)\}'

    def replace(match):
        var_name = match.group(1)
        return os.getenv(var_name, match.group(0))

    return re.sub(pattern, replace, value)


def process_dict_env_vars(d: dict) -> dict:
    """Recursively substitute environment variables in a dict."""
    result = {}
    for key, value in d.items():
        if isinstance(value, str):
            result[key] = substitute_env_vars(value)
        elif isinstance(value, dict):
            result[key] = process_dict_env_vars(value)
        elif isinstance(value, list):
            result[key] = [
                substitute_env_vars(v) if isinstance(v, str) else v
                for v in value
            ]
        else:
            result[key] = value
    return result


def load_config(config_path: Optional[str] = None) -> EvalConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, returns defaults.

    Returns:
        EvalConfig instance
    """
    if config_path is None:
        return EvalConfig()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        raw_config = yaml.safe_load(f) or {}

    # Substitute environment variables
    config_data = process_dict_env_vars(raw_config)

    # Build config object
    config = EvalConfig()

    # Top-level fields
    config.adapter = config_data.get("adapter", config.adapter)
    config.endpoint = config_data.get("endpoint", config.endpoint)
    config.dataset = config_data.get("dataset", config.dataset)
    config.concurrency = config_data.get("concurrency", config.concurrency)

    # HTTP config
    if "http" in config_data:
        http = config_data["http"]
        config.http = HTTPConfig(
            headers=http.get("headers", {}),
            timeout=http.get("timeout", 30),
            slow_threshold=http.get("slow_threshold", 5),
        )

    # Retry config
    if "retry" in config_data:
        retry = config_data["retry"]
        config.retry = RetryConfig(
            max_attempts=retry.get("max_attempts", 3),
            backoff=retry.get("backoff", "exponential"),
            base_delay=retry.get("base_delay", 1.0),
        )

    # Weights config
    if "weights" in config_data:
        weights = config_data["weights"]
        config.weights = WeightsConfig(
            faithfulness=weights.get("faithfulness", 40.0),
            answer_relevance=weights.get("answer_relevance", 20.0),
            context_precision=weights.get("context_precision", 20.0),
            context_recall=weights.get("context_recall", 20.0),
        )

    # Thresholds config
    if "thresholds" in config_data:
        thresholds = config_data["thresholds"]
        config.thresholds = ThresholdsConfig(
            composite=thresholds.get("composite"),
            faithfulness=thresholds.get("faithfulness"),
            answer_relevance=thresholds.get("answer_relevance"),
            context_precision=thresholds.get("context_precision"),
            context_recall=thresholds.get("context_recall"),
        )

    # Comparison config
    if "comparison" in config_data:
        comparison = config_data["comparison"]
        config.comparison = ComparisonConfig(
            semantic_similarity_threshold=comparison.get(
                "semantic_similarity_threshold", 0.85
            ),
            embedding_model=comparison.get("embedding_model", "all-MiniLM-L6-v2"),
        )

    # Output config
    if "output" in config_data:
        output = config_data["output"]
        config.output = OutputConfig(
            directory=output.get("directory", "results"),
            formats=output.get("formats", ["markdown", "json"]),
        )

    return config


def validate_config(config: EvalConfig) -> list[str]:
    """
    Validate configuration and return list of warnings.

    Args:
        config: Configuration to validate

    Returns:
        List of warning messages (empty if no issues)

    Raises:
        ValueError: If configuration is invalid
    """
    warnings = []

    # Validate adapter
    valid_adapters = ["http", "langchain", "llamaindex"]
    if config.adapter not in valid_adapters:
        raise ValueError(f"Invalid adapter: {config.adapter}. Must be one of {valid_adapters}")

    # Validate endpoint for HTTP adapter
    if config.adapter == "http" and not config.endpoint:
        raise ValueError("HTTP adapter requires an endpoint")

    # Validate weights sum
    weights_sum = (
        config.weights.faithfulness
        + config.weights.answer_relevance
        + config.weights.context_precision
        + config.weights.context_recall
    )
    if weights_sum <= 0:
        raise ValueError("Metric weights must sum to a positive number")

    # Validate thresholds
    for name, value in [
        ("composite", config.thresholds.composite),
        ("faithfulness", config.thresholds.faithfulness),
        ("answer_relevance", config.thresholds.answer_relevance),
        ("context_precision", config.thresholds.context_precision),
        ("context_recall", config.thresholds.context_recall),
    ]:
        if value is not None and (value < 0 or value > 1):
            raise ValueError(f"Threshold {name} must be between 0 and 1, got {value}")

    # Validate concurrency
    if config.concurrency < 1:
        raise ValueError("Concurrency must be at least 1")

    # Warning for high concurrency
    if config.concurrency > 10:
        warnings.append(f"High concurrency ({config.concurrency}) may overwhelm target RAG system")

    return warnings
