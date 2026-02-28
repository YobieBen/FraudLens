"""
Event-driven architecture for FraudLens.

Provides a lightweight event bus for inter-component communication,
enabling loose coupling and extensibility.

Author: Yobie Benjamin
Date: 2026-02-28
"""

from fraudlens.events.bus import EventBus, get_event_bus
from fraudlens.events.middleware import EventMiddleware, LoggingMiddleware, MetricsMiddleware
from fraudlens.events.types import (
    AnalysisCompleteEvent,
    DetectionStartedEvent,
    ErrorEvent,
    Event,
    EventPriority,
    ThresholdExceededEvent,
)

__all__ = [
    # Core
    "Event",
    "EventBus",
    "get_event_bus",
    "EventPriority",
    # Event types
    "DetectionStartedEvent",
    "AnalysisCompleteEvent",
    "ThresholdExceededEvent",
    "ErrorEvent",
    # Middleware
    "EventMiddleware",
    "LoggingMiddleware",
    "MetricsMiddleware",
]
