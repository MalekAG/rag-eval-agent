"""
Abstract base adapter for RAG systems.

All adapters must conform to this interface to ensure consistent evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class QueryResult:
    """Result from a RAG query."""
    answer: str
    contexts: Optional[list[str]]
    latency_ms: float
    metadata: Optional[dict] = None


class RAGAdapter(ABC):
    """
    Abstract base class for RAG system adapters.

    All adapters must implement the query method with this exact signature.
    """

    @abstractmethod
    def query(self, question: str) -> tuple[str, Optional[list[str]]]:
        """
        Query the RAG system with a question.

        Args:
            question: The question to ask the RAG system.

        Returns:
            A tuple of (answer, retrieved_contexts):
            - answer: The generated response string
            - retrieved_contexts: List of context strings used to generate answer,
                                  or None if contexts are unavailable

        Raises:
            ConnectionError: If the RAG system is unreachable
            TimeoutError: If the query times out
            ValueError: If the response is malformed
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the RAG system is reachable and responding.

        Returns:
            True if healthy, False otherwise
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the adapter name for reporting."""
        pass

    @property
    def description(self) -> str:
        """Optional description of the RAG system being evaluated."""
        return ""
