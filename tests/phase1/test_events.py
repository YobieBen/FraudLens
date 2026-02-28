"""
Tests for Phase 1 event system.

Author: Yobie Benjamin
Date: 2026-02-28
"""

import asyncio

import pytest

from fraudlens.events import (
    AnalysisCompleteEvent,
    DetectionStartedEvent,
    Event,
    EventBus,
    EventPriority,
    LoggingMiddleware,
    MetricsMiddleware,
)


class TestEventTypes:
    """Test event type definitions."""
    
    def test_event_creation(self):
        """Test basic event creation."""
        event = DetectionStartedEvent(
            source="test",
            input_id="test-123",
            modality="text"
        )
        
        assert event.source == "test"
        assert event.input_id == "test-123"
        assert event.modality == "text"
        assert event.priority == EventPriority.NORMAL
    
    def test_event_immutability(self):
        """Test that events are immutable."""
        event = DetectionStartedEvent(
            source="test",
            input_id="test-123",
            modality="text"
        )
        
        with pytest.raises(Exception):  # Pydantic ValidationError
            event.source = "modified"
    
    def test_event_with_priority(self):
        """Test event with custom priority."""
        event = AnalysisCompleteEvent(
            source="test",
            input_id="test-123",
            fraud_score=0.8,
            fraud_types=["phishing"],
            processing_time_ms=100.0,
            priority=EventPriority.HIGH
        )
        
        assert event.priority == EventPriority.HIGH


class TestEventBus:
    """Test event bus functionality."""
    
    @pytest.fixture
    def bus(self):
        """Create fresh event bus for each test."""
        return EventBus()
    
    @pytest.mark.asyncio
    async def test_subscribe_and_emit(self, bus):
        """Test subscribing to and emitting events."""
        received_events = []
        
        async def handler(event: DetectionStartedEvent):
            received_events.append(event)
        
        bus.subscribe(DetectionStartedEvent, handler)
        
        event = DetectionStartedEvent(
            source="test",
            input_id="test-123",
            modality="text"
        )
        
        await bus.emit(event)
        
        assert len(received_events) == 1
        assert received_events[0].input_id == "test-123"
    
    @pytest.mark.asyncio
    async def test_multiple_handlers(self, bus):
        """Test multiple handlers for same event."""
        call_count = 0
        
        async def handler1(event: DetectionStartedEvent):
            nonlocal call_count
            call_count += 1
        
        async def handler2(event: DetectionStartedEvent):
            nonlocal call_count
            call_count += 1
        
        bus.subscribe(DetectionStartedEvent, handler1)
        bus.subscribe(DetectionStartedEvent, handler2)
        
        event = DetectionStartedEvent(
            source="test",
            input_id="test-123",
            modality="text"
        )
        
        await bus.emit(event)
        
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self, bus):
        """Test that handlers execute in priority order."""
        execution_order = []
        
        async def low_priority_handler(event: Event):
            execution_order.append("low")
        
        async def high_priority_handler(event: Event):
            execution_order.append("high")
        
        # Subscribe with different priorities
        bus.subscribe(DetectionStartedEvent, low_priority_handler, priority=0)
        bus.subscribe(DetectionStartedEvent, high_priority_handler, priority=10)
        
        event = DetectionStartedEvent(
            source="test",
            input_id="test-123",
            modality="text"
        )
        
        await bus.emit(event)
        await asyncio.sleep(0.1)  # Give time for async execution
        
        # High priority should execute first
        assert execution_order[0] == "high"
    
    @pytest.mark.asyncio
    async def test_wildcard_subscription(self, bus):
        """Test wildcard subscription to all events."""
        received_events = []
        
        async def wildcard_handler(event: Event):
            received_events.append(event)
        
        bus.subscribe_all(wildcard_handler)
        
        event1 = DetectionStartedEvent(
            source="test",
            input_id="test-123",
            modality="text"
        )
        event2 = AnalysisCompleteEvent(
            source="test",
            input_id="test-123",
            fraud_score=0.5,
            fraud_types=[],
            processing_time_ms=50.0
        )
        
        await bus.emit(event1)
        await bus.emit(event2)
        
        assert len(received_events) == 2
    
    @pytest.mark.asyncio
    async def test_unsubscribe(self, bus):
        """Test unsubscribing from events."""
        call_count = 0
        
        async def handler(event: Event):
            nonlocal call_count
            call_count += 1
        
        bus.subscribe(DetectionStartedEvent, handler)
        
        event = DetectionStartedEvent(
            source="test",
            input_id="test-123",
            modality="text"
        )
        
        await bus.emit(event)
        assert call_count == 1
        
        bus.unsubscribe(DetectionStartedEvent, handler)
        
        await bus.emit(event)
        assert call_count == 1  # Should not increase
    
    @pytest.mark.asyncio
    async def test_error_handling(self, bus):
        """Test that errors in handlers don't stop other handlers."""
        call_count = 0
        
        async def failing_handler(event: Event):
            raise ValueError("Test error")
        
        async def successful_handler(event: Event):
            nonlocal call_count
            call_count += 1
        
        bus.subscribe(DetectionStartedEvent, failing_handler)
        bus.subscribe(DetectionStartedEvent, successful_handler)
        
        event = DetectionStartedEvent(
            source="test",
            input_id="test-123",
            modality="text"
        )
        
        await bus.emit(event)
        
        # Successful handler should still execute
        assert call_count == 1


class TestEventMiddleware:
    """Test event middleware."""
    
    @pytest.fixture
    def bus(self):
        """Create fresh event bus for each test."""
        return EventBus()
    
    @pytest.mark.asyncio
    async def test_logging_middleware(self, bus):
        """Test logging middleware."""
        middleware = LoggingMiddleware(log_level="DEBUG")
        bus.add_middleware(middleware)
        
        event = DetectionStartedEvent(
            source="test",
            input_id="test-123",
            modality="text"
        )
        
        # Should not raise errors
        await bus.emit(event)
    
    @pytest.mark.asyncio
    async def test_metrics_middleware(self, bus):
        """Test metrics middleware."""
        middleware = MetricsMiddleware()
        bus.add_middleware(middleware)
        
        event1 = DetectionStartedEvent(
            source="test",
            input_id="test-123",
            modality="text"
        )
        event2 = AnalysisCompleteEvent(
            source="test",
            input_id="test-456",
            fraud_score=0.7,
            fraud_types=["phishing"],
            processing_time_ms=150.0
        )
        
        await bus.emit(event1)
        await bus.emit(event2)
        
        metrics = middleware.get_metrics()
        
        assert metrics["total_events"] == 2
        assert "DetectionStartedEvent" in metrics["event_counts"]
        assert "AnalysisCompleteEvent" in metrics["event_counts"]
    
    @pytest.mark.asyncio
    async def test_middleware_filtering(self, bus):
        """Test middleware that filters events."""
        from fraudlens.events.middleware import FilterMiddleware
        
        # Only allow high priority events
        middleware = FilterMiddleware(
            lambda event: event.priority == EventPriority.HIGH
        )
        bus.add_middleware(middleware)
        
        received = []
        
        async def handler(event: Event):
            received.append(event)
        
        bus.subscribe_all(handler)
        
        normal_event = DetectionStartedEvent(
            source="test",
            input_id="test-123",
            modality="text",
            priority=EventPriority.NORMAL
        )
        
        high_event = DetectionStartedEvent(
            source="test",
            input_id="test-456",
            modality="text",
            priority=EventPriority.HIGH
        )
        
        await bus.emit(normal_event)
        await bus.emit(high_event)
        
        # Only high priority event should pass
        assert len(received) == 1
        assert received[0].priority == EventPriority.HIGH


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
