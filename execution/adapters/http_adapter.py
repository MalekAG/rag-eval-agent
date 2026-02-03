"""
HTTP adapter for generic RAG API endpoints.

Supports configurable authentication and response parsing.
"""

import ipaddress
import os
import socket
from typing import Optional
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

from .base import RAGAdapter

load_dotenv()


# Blocked IP ranges for SSRF protection
BLOCKED_IP_RANGES = [
    ipaddress.ip_network("127.0.0.0/8"),      # Loopback
    ipaddress.ip_network("10.0.0.0/8"),       # Private Class A
    ipaddress.ip_network("172.16.0.0/12"),    # Private Class B
    ipaddress.ip_network("192.168.0.0/16"),   # Private Class C
    ipaddress.ip_network("169.254.0.0/16"),   # Link-local (includes AWS metadata)
    ipaddress.ip_network("0.0.0.0/8"),        # "This" network
    ipaddress.ip_network("224.0.0.0/4"),      # Multicast
    ipaddress.ip_network("240.0.0.0/4"),      # Reserved
]


def validate_url(url: str, allow_internal: bool = False) -> None:
    """
    Validate URL to prevent SSRF attacks.

    Args:
        url: The URL to validate
        allow_internal: If True, allows internal/private network addresses.
                       Should only be True for testing or controlled environments.

    Raises:
        ValueError: If URL is invalid or points to blocked resources
    """
    parsed = urlparse(url)

    # Check scheme
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme '{parsed.scheme}'. Only http and https are allowed.")

    # Check hostname exists
    if not parsed.hostname:
        raise ValueError("URL must have a valid hostname")

    # Skip IP range checks if internal addresses are allowed
    if allow_internal:
        return

    # Resolve hostname to IP and check against blocked ranges
    try:
        ip_str = socket.gethostbyname(parsed.hostname)
        ip = ipaddress.ip_address(ip_str)

        for blocked_range in BLOCKED_IP_RANGES:
            if ip in blocked_range:
                raise ValueError(
                    f"URL hostname resolves to blocked IP range ({ip_str}). "
                    "Internal/private network addresses are not allowed."
                )
    except socket.gaierror:
        # DNS resolution failed - allow it to fail later during actual request
        # This handles cases where the hostname might be valid but not resolvable yet
        pass


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
        allow_internal: bool = False,
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
            allow_internal: If True, allows internal/private network addresses.
                           Should only be True for testing or controlled environments.
        """
        # Validate endpoint URL to prevent SSRF
        validate_url(endpoint, allow_internal=allow_internal)

        self.endpoint = endpoint
        self.timeout = timeout
        self.response_mapping = response_mapping or {}
        self.request_format = request_format or {"question": "{question}"}

        # Validate health endpoint if provided
        if health_endpoint:
            validate_url(health_endpoint, allow_internal=allow_internal)
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
