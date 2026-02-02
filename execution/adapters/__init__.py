"""RAG system adapters."""

from .base import RAGAdapter, QueryResult
from .http_adapter import HTTPAdapter

__all__ = ["RAGAdapter", "QueryResult", "HTTPAdapter"]
