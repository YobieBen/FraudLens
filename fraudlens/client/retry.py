"""
Retry logic with exponential backoff for FraudLens SDK.

Author: Yobie Benjamin
Date: 2026-02-28
"""

import asyncio
import random
from typing import Any, Awaitable, Callable, TypeVar

from fraudlens.client.exceptions import RateLimitError, TimeoutError, TransportError

T = TypeVar("T")


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given retry attempt."""
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            # Add jitter to prevent thundering herd
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay


def is_retryable_error(error: Exception) -> bool:
    """Check if error should trigger a retry."""
    if isinstance(error, RateLimitError):
        return True
    
    if isinstance(error, TimeoutError):
        return True
    
    if isinstance(error, TransportError):
        return True
    
    # Don't retry validation errors, auth errors, etc.
    return False


async def retry_async(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    config: RetryConfig | None = None,
    **kwargs: Any
) -> T:
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        *args: Positional arguments for func
        config: Retry configuration
        **kwargs: Keyword arguments for func
    
    Returns:
        Result of func
    
    Raises:
        Last exception if all retries exhausted
    
    Example:
        ```python
        result = await retry_async(
            api_call,
            endpoint="/analyze",
            data={"text": "..."},
            config=RetryConfig(max_retries=5)
        )
        ```
    """
    config = config or RetryConfig()
    last_error: Exception | None = None
    
    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_error = e
            
            # Check if we should retry
            if not is_retryable_error(e):
                raise
            
            # Check if we've exhausted retries
            if attempt >= config.max_retries:
                raise
            
            # Calculate delay
            if isinstance(e, RateLimitError) and e.retry_after:
                delay = e.retry_after
            else:
                delay = config.calculate_delay(attempt)
            
            # Wait before retrying
            await asyncio.sleep(delay)
    
    # This should never be reached, but just in case
    if last_error:
        raise last_error
    raise RuntimeError("Retry logic error")


def retry_sync(
    func: Callable[..., T],
    *args: Any,
    config: RetryConfig | None = None,
    **kwargs: Any
) -> T:
    """
    Retry a sync function with exponential backoff.
    
    Args:
        func: Function to retry
        *args: Positional arguments for func
        config: Retry configuration
        **kwargs: Keyword arguments for func
    
    Returns:
        Result of func
    
    Raises:
        Last exception if all retries exhausted
    """
    import time
    
    config = config or RetryConfig()
    last_error: Exception | None = None
    
    for attempt in range(config.max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            
            if not is_retryable_error(e):
                raise
            
            if attempt >= config.max_retries:
                raise
            
            if isinstance(e, RateLimitError) and e.retry_after:
                delay = e.retry_after
            else:
                delay = config.calculate_delay(attempt)
            
            time.sleep(delay)
    
    if last_error:
        raise last_error
    raise RuntimeError("Retry logic error")
