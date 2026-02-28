"""
Middleware implementations for event processing.

Author: Yobie Benjamin
Date: 2026-02-28
"""

import time
from typing import Protocol

from loguru import logger

from fraudlens.events.types import Event


class EventMiddleware(Protocol):
    """Protocol for event middleware."""
    
    async def __call__(self, event: Event) -> Event | None:
        """
        Process event.
        
        Args:
            event: Event to process
        
        Returns:
            Processed event or None to stop propagation
        """
        ...


class LoggingMiddleware:
    """Middleware that logs all events."""
    
    __name__ = "logging_middleware"
    
    def __init__(self, log_level: str = "DEBUG"):
        """
        Initialize logging middleware.
        
        Args:
            log_level: Log level for events
        """
        self.log_level = log_level
    
    async def __call__(self, event: Event) -> Event:
        """Log event and pass through."""
        logger.log(
            self.log_level,
            f"Event: {event.__class__.__name__} from {event.source} "
            f"(priority={event.priority.name}, id={event.id})"
        )
        return event


class MetricsMiddleware:
    """Middleware that collects metrics on events."""
    
    __name__ = "metrics_middleware"
    
    def __init__(self):
        """Initialize metrics middleware."""
        self._event_counts: dict[str, int] = {}
        self._processing_times: dict[str, list[float]] = {}
    
    async def __call__(self, event: Event) -> Event:
        """Collect metrics and pass through."""
        event_type = event.__class__.__name__
        
        # Count events
        self._event_counts[event_type] = self._event_counts.get(event_type, 0) + 1
        
        return event
    
    def get_metrics(self) -> dict:
        """Get collected metrics."""
        return {
            "event_counts": self._event_counts.copy(),
            "total_events": sum(self._event_counts.values()),
        }
    
    def reset(self) -> None:
        """Reset metrics."""
        self._event_counts.clear()
        self._processing_times.clear()


class FilterMiddleware:
    """Middleware that filters events based on criteria."""
    
    __name__ = "filter_middleware"
    
    def __init__(self, filter_func: callable):
        """
        Initialize filter middleware.
        
        Args:
            filter_func: Function that returns True to allow event
        """
        self.filter_func = filter_func
    
    async def __call__(self, event: Event) -> Event | None:
        """Filter event based on criteria."""
        if self.filter_func(event):
            return event
        
        logger.debug(f"Event filtered: {event.__class__.__name__}")
        return None


class TimingMiddleware:
    """Middleware that adds timing information to events."""
    
    def __init__(self):
        """Initialize timing middleware."""
        self._start_times: dict[str, float] = {}
    
    async def __call__(self, event: Event) -> Event:
        """Add timing metadata to event."""
        current_time = time.time()
        event_id = str(event.id)
        
        # Add current timestamp to metadata
        if "middleware_timestamps" not in event.metadata:
            event.metadata["middleware_timestamps"] = []
        
        event.metadata["middleware_timestamps"].append({
            "middleware": "timing",
            "timestamp": current_time
        })
        
        return event
