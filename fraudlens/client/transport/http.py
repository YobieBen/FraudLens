"""
HTTP transport for remote FraudLens API calls.

Author: Yobie Benjamin  
Date: 2026-02-28
"""

import asyncio
from typing import Any, AsyncIterator

import aiohttp

from fraudlens.client.exceptions import (
    APIError,
    AuthenticationError,
    RateLimitError,
    ResourceNotFoundError,
    TimeoutError,
    TransportError,
    ValidationError,
)
from fraudlens.client.models import StreamEvent


class HTTPTransport:
    """
    Transport for HTTP calls to FraudLens API.
    
    This transport sends requests to a remote FraudLens API server.
    Ideal for SaaS deployments where the SDK is used as a client.
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        """
        Initialize HTTP transport.
        
        Args:
            base_url: Base URL of FraudLens API (e.g., "https://api.fraudlens.io")
            api_key: API key for authentication
            timeout: Default request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._session: aiohttp.ClientSession | None = None
        self._closed = False
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout_config = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout_config,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "FraudLens-SDK/0.1.0"
                }
            )
        
        return self._session
    
    async def send_request(
        self,
        endpoint: str,
        data: dict[str, Any],
        timeout: float | None = None
    ) -> dict[str, Any]:
        """
        Send an HTTP request to the API.
        
        Args:
            endpoint: API endpoint (e.g., "/v1/analyze/email")
            data: Request payload
            timeout: Request timeout in seconds (overrides default)
        
        Returns:
            Response data
        
        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit exceeded
            ValidationError: If request data is invalid
            ResourceNotFoundError: If resource not found
            APIError: For other API errors
            TransportError: If communication fails
            TimeoutError: If request times out
        """
        if self._closed:
            raise TransportError("Transport is closed")
        
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with session.post(url, json=data) as response:
                # Handle different status codes
                if response.status == 200:
                    return await response.json()
                
                elif response.status == 401:
                    raise AuthenticationError("Invalid API key")
                
                elif response.status == 429:
                    retry_after = response.headers.get("Retry-After")
                    raise RateLimitError(
                        "Rate limit exceeded",
                        retry_after=int(retry_after) if retry_after else None
                    )
                
                elif response.status == 400:
                    error_data = await response.json()
                    raise ValidationError(
                        error_data.get("message", "Invalid request"),
                        details=error_data
                    )
                
                elif response.status == 404:
                    raise ResourceNotFoundError("Resource not found")
                
                elif response.status >= 500:
                    error_data = await response.json()
                    raise APIError(
                        error_data.get("message", "Server error"),
                        status_code=response.status,
                        details=error_data
                    )
                
                else:
                    error_data = await response.json()
                    raise APIError(
                        error_data.get("message", "Unknown error"),
                        status_code=response.status,
                        details=error_data
                    )
        
        except aiohttp.ClientError as e:
            raise TransportError(f"HTTP request failed: {str(e)}") from e
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request timed out after {timeout or self.timeout}s")
    
    async def stream_request(
        self,
        endpoint: str,
        data: dict[str, Any],
        timeout: float | None = None
    ) -> AsyncIterator[StreamEvent]:
        """
        Send a streaming HTTP request (Server-Sent Events).
        
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
        if self._closed:
            raise TransportError("Transport is closed")
        
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with session.post(
                url,
                json=data,
                headers={"Accept": "text/event-stream"}
            ) as response:
                if response.status != 200:
                    error_data = await response.json()
                    raise APIError(
                        error_data.get("message", "Stream failed"),
                        status_code=response.status
                    )
                
                # Parse Server-Sent Events
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    
                    # SSE format: "data: {json}\n\n"
                    if line.startswith("data: "):
                        json_data = line[6:]  # Remove "data: " prefix
                        
                        try:
                            import json
                            event_data = json.loads(json_data)
                            
                            # Convert to StreamEvent
                            yield StreamEvent(**event_data)
                        
                        except json.JSONDecodeError:
                            # Skip malformed events
                            continue
        
        except aiohttp.ClientError as e:
            raise TransportError(f"Stream failed: {str(e)}") from e
        except asyncio.TimeoutError:
            raise TimeoutError(f"Stream timed out after {timeout or self.timeout}s")
    
    async def close(self) -> None:
        """Close transport and cleanup resources."""
        self._closed = True
        
        if self._session and not self._session.closed:
            await self._session.close()
