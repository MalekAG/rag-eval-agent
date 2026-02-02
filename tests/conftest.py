"""
Shared test fixtures for RAG evaluation tests.
"""

import pytest
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path

# Add execution directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "execution"))


@pytest.fixture
def sample_test_case():
    """Sample test case for evaluation."""
    return {
        "id": "test_001",
        "question": "What is the refund policy?",
        "ground_truth": "Full refund within 30 days of purchase.",
        "expected_contexts": ["policy.md"],
        "critical": False,
        "tags": ["billing"],
    }


@pytest.fixture
def sample_dataset():
    """Sample dataset with multiple test cases."""
    return {
        "metadata": {
            "name": "Test Dataset",
            "created": "2026-02-02",
            "version": "1.0",
        },
        "test_cases": [
            {
                "id": "tc_001",
                "question": "What is the refund policy?",
                "ground_truth": "Full refund within 30 days.",
                "critical": True,
                "tags": ["billing"],
            },
            {
                "id": "tc_002",
                "question": "How do I reset my password?",
                "ground_truth": "Click Forgot Password on the login page.",
                "critical": False,
                "tags": ["auth"],
            },
        ],
    }


@pytest.fixture
def mock_adapter():
    """Mock RAG adapter for testing."""
    adapter = MagicMock()
    adapter.name = "mock"
    adapter.description = "Mock adapter for testing"
    adapter.health_check.return_value = True
    adapter.query.return_value = (
        "Full refund within 30 days of purchase.",
        ["Our refund policy allows full refunds within 30 days."],
    )
    return adapter


@pytest.fixture
def mock_anthropic():
    """Mock Anthropic client for testing."""
    with patch("metrics.anthropic.Anthropic") as mock:
        client = MagicMock()
        mock.return_value = client

        # Mock response
        response = MagicMock()
        response.content = [MagicMock(text="Score: 0.85\nReasoning: Good answer.")]
        response.usage.input_tokens = 1000
        response.usage.output_tokens = 200
        client.messages.create.return_value = response

        yield client


@pytest.fixture
def mock_embedding_model():
    """Mock sentence transformer model."""
    with patch("metrics.SentenceTransformer") as mock:
        model = MagicMock()
        mock.return_value = model

        # Mock embeddings (normalized vectors)
        import numpy as np
        model.encode.return_value = np.array([
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
        ])

        yield model
