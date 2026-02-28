"""
Event type definitions for FraudLens.

Author: Yobie Benjamin
Date: 2026-02-28
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EventPriority(int, Enum):
    """Event priority levels."""
    
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class Event(BaseModel):
    """
    Base event class.
    
    All events must inherit from this class.
    """
    
    id: UUID = Field(default_factory=uuid4)
    """Unique event identifier."""
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    """When the event occurred."""
    
    source: str
    """Component or service that generated the event."""
    
    priority: EventPriority = EventPriority.NORMAL
    """Event priority."""
    
    metadata: dict[str, Any] = Field(default_factory=dict)
    """Additional event metadata."""
    
    class Config:
        """Pydantic config."""
        frozen = True  # Events are immutable


class DetectionStartedEvent(Event):
    """Event emitted when fraud detection starts."""
    
    input_id: str
    """Identifier for the input being analyzed."""
    
    modality: str
    """Type of input (text, image, etc.)."""
    
    request_context: dict[str, Any] = Field(default_factory=dict)
    """Context about the detection request."""


class AnalysisCompleteEvent(Event):
    """Event emitted when analysis completes."""
    
    input_id: str
    """Identifier for the analyzed input."""
    
    fraud_score: float
    """Overall fraud score (0.0 to 1.0)."""
    
    fraud_types: list[str]
    """Detected fraud types."""
    
    processing_time_ms: float
    """Processing time in milliseconds."""
    
    result_summary: dict[str, Any] = Field(default_factory=dict)
    """Summary of analysis results."""


class ThresholdExceededEvent(Event):
    """Event emitted when fraud score exceeds threshold."""
    
    input_id: str
    """Identifier for the input."""
    
    fraud_score: float
    """Fraud score that exceeded threshold."""
    
    threshold: float
    """Threshold that was exceeded."""
    
    fraud_types: list[str]
    """Types of fraud detected."""
    
    recommended_actions: list[str] = Field(default_factory=list)
    """Recommended actions to take."""


class ErrorEvent(Event):
    """Event emitted when an error occurs."""
    
    error_type: str
    """Type or category of error."""
    
    error_message: str
    """Error message."""
    
    stack_trace: str | None = None
    """Stack trace if available."""
    
    component: str
    """Component where error occurred."""
    
    recoverable: bool = False
    """Whether the error is recoverable."""


class ModelLoadedEvent(Event):
    """Event emitted when a model is loaded."""
    
    model_id: str
    """Model identifier."""
    
    model_type: str
    """Type of model (text, image, etc.)."""
    
    load_time_ms: float
    """Time taken to load model."""
    
    memory_mb: float
    """Memory used by model."""


class CacheHitEvent(Event):
    """Event emitted on cache hit."""
    
    cache_key: str
    """Cache key that was hit."""
    
    cache_backend: str
    """Cache backend used."""


class CacheMissEvent(Event):
    """Event emitted on cache miss."""
    
    cache_key: str
    """Cache key that was missed."""
    
    cache_backend: str
    """Cache backend used."""


class PluginExecutedEvent(Event):
    """Event emitted when a plugin executes."""
    
    plugin_id: str
    """Plugin identifier."""
    
    execution_time_ms: float
    """Plugin execution time."""
    
    success: bool
    """Whether plugin executed successfully."""
    
    plugin_result: dict[str, Any] = Field(default_factory=dict)
    """Result from plugin execution."""
