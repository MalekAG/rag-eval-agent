"""
HTTP adapter for generic RAG API endpoints.

Supports configurable authentication and response parsing.
"""

import os
from typing import Optional
import requests
from dotenv import load_dotenv

from .base import RAGAdapter

load_dotenv()


class HTTPAdapter(RAGAdapter):
    """
    Adapter for RAG systems exposed via HTTP API.

    Expected API response format:
    {
        "answer": "The generated response",
        "contexts": ["context1", "context2", ...]  // optional
    }

    Alternative field names can be configured via response_mapping.
    """

    def __init__(
        self,
        endpoint: str,
        headers: Optional[dict[str, str]] = None,
        timeout: int = 30,
        response_mapping: Optional[dict[str, str]] = None,
        request_format: Optional[dict] = None,
        health_endpoint: Optional[str] = None,
    ):
        """
        Initialize HTTP adapter.

        Args:
            endpoint: The URL to query
            headers: HTTP headers (auth, content-type, etc.)
            timeout: Request timeout in seconds
            response_mapping: Map from standard fields to actual API field names
                             e.g., {"answer": "response", "contexts": "sources"}
            request_format: Custom request body template. Use {question} placeholder.
                           Default: {"question": "{question}"}
            health_endpoint: Optional health check URL. If None, tries common patterns.
        """
        self.endpoint = endpoint
        self.timeout = timeout
        self.response_mapping = response_mapping or {}
        self.request_format = request_format or {"question": "{question}"}
        self.health_endpoint = health_endpoint

        # Build headers with auth support
        self.headers = {"Content-Type": "application/json"}

        # Priority 1: Explicit headers parameter
        if headers:
            self.headers.update(headers)

        # Priority 3: Environment variable (only if not already set)
        if "Authorization" not in self.headers:
            auth_header = os.getenv("RAG_AUTH_HEADER")
            if auth_header:
                self.headers["Authorization"] = auth_header

    def query(self, question: str) -> tuple[str, Optional[list[str]]]:
        """
        Query the RAG endpoint.

        Args:
            question: The question to ask

        Returns:
            Tuple of (answer, contexts)
        """
        # Build request body
        body = {}
        for key, value in self.request_format.items():
            if isinstance(value, str) and "{question}" in value:
                body[key] = value.format(question=question)
            else:
                body[key] = value

        # Make request
        try:
            response = requests.post(
                self.endpoint,
                json=body,
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to {self.endpoint}: {e}")
        except requests.exceptions.HTTPError as e:
            raise ValueError(f"HTTP error: {e}")

        # Parse response
        try:
            data = response.json()
        except ValueError:
            raise ValueError(f"Invalid JSON response: {response.text[:200]}")

        # Extract answer using mapping or default field names
        answer_field = self.response_mapping.get("answer", "answer")
        contexts_field = self.response_mapping.get("contexts", "contexts")

        # Try common alternative field names
        answer = None
        for field in [answer_field, "answer", "response", "result", "output", "text"]:
            if field in data:
                answer = data[field]
                break

        if answer is None:
            raise ValueError(f"Could not find answer in response. Fields: {list(data.keys())}")

        # Extract contexts (optional)
        contexts = None
        for field in [contexts_field, "contexts", "context", "sources", "documents", "retrieved"]:
            if field in data:
                contexts = data[field]
                if isinstance(contexts, str):
                    contexts = [contexts]
                break

        return answer, contexts

    def health_check(self) -> bool:
        """Check if endpoint is reachable."""
        # If explicit health endpoint configured, use it
        if self.health_endpoint:
            try:
                response = requests.get(self.health_endpoint, timeout=5)
                return response.status_code == 200
            except Exception:
                return False

        # Try multiple common health check patterns
        base_url = self.endpoint.rsplit("/", 1)[0]
        health_patterns = [
            f"{base_url}/health",
            f"{base_url}/healthz",
            f"{base_url}/ping",
            f"{base_url}/status",
        ]

        for health_url in health_patterns:
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    return True
            except Exception:
                continue

        # Fallback: HEAD request to the endpoint itself
        try:
            response = requests.head(self.endpoint, timeout=5)
            return response.status_code < 500
        except Exception:
            return False

    @property
    def name(self) -> str:
        return "http"

    @property
    def description(self) -> str:
        return f"HTTP endpoint: {self.endpoint}"
