"""
Transport protocol for FraudLens SDK.

Author: Yobie Benjamin
Date: 2026-02-28
"""

from typing import Any, AsyncIterator, Protocol

from fraudlens.client.models import AnalysisResult, StreamEvent


class Transport(Protocol):
    """Protocol for transport layer between SDK and backend."""
    
    async def send_request(
        self,
        endpoint: str,
        data: dict[str, Any],
        timeout: float | None = None
    ) -> dict[str, Any]:
        """
        Send a request and get response.
        
        Args:
            endpoint: API endpoint (e.g., "/analyze/email")
            data: Request payload
            timeout: Request timeout in seconds
        
        Returns:
            Response data
        
        Raises:
            TransportError: If communication fails
            TimeoutError: If request times out
            AuthenticationError: If authentication fails
        """
        ...
    
    async def stream_request(
        self,
        endpoint: str,
        data: dict[str, Any],
        timeout: float | None = None
    ) -> AsyncIterator[StreamEvent]:
        """
        Send a streaming request and yield events.
        
        Args:
            endpoint: API endpoint
            data: Request payload
            timeout: Request timeout
        
        Yields:
            Stream events as they arrive
        
        Raises:
            TransportError: If communication fails
            TimeoutError: If request times out
        """
        ...
    
    async def close(self) -> None:
        """Close transport and cleanup resources."""
        ...
