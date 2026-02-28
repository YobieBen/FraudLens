"""
Phase 1 integration tests - E2E tests for all components working together.

Author: Yobie Benjamin
Date: 2026-02-28
"""

import pytest

from fraudlens.compat import ConfigAdapter
from fraudlens.config import FraudLensSettings, get_settings
from fraudlens.core.container import Container
from fraudlens.events import (
    AnalysisCompleteEvent,
    DetectionStartedEvent,
    EventBus,
    EventPriority,
    MetricsMiddleware,
)


class TestPhase1Integration:
    """Integration tests for all Phase 1 components."""
    
    @pytest.mark.asyncio
    async def test_config_with_events(self):
        """Test configuration system integrated with event system."""
        # Create settings
        settings = FraudLensSettings(environment="dev", debug=True)
        
        # Create event bus
        bus = EventBus()
        
        # Subscribe to events
        received_events = []
        
        async def config_changed_handler(event):
            received_events.append(event)
        
        bus.subscribe(DetectionStartedEvent, config_changed_handler)
        
        # Emit event with config context
        event = DetectionStartedEvent(
            source="test",
            input_id="test-123",
            modality="text",
            metadata={"environment": settings.environment}
        )
        
        await bus.emit(event)
        
        assert len(received_events) == 1
        assert received_events[0].metadata["environment"] == "dev"
    
    @pytest.mark.asyncio
    async def test_container_with_events_and_config(self):
        """Test DI container with event bus and config."""
        # Create container
        container = Container()
        
        # Register components
        settings = FraudLensSettings()
        bus = EventBus()
        
        container.register_singleton(FraudLensSettings, instance=settings)
        container.register_singleton(EventBus, instance=bus)
        
        # Resolve and verify
        resolved_settings = container.resolve(FraudLensSettings)
        resolved_bus = container.resolve(EventBus)
        
        assert resolved_settings is settings
        assert resolved_bus is bus
        
        # Test event emission
        event = DetectionStartedEvent(
            source="container_test",
            input_id="test-456",
            modality="image"
        )
        
        await resolved_bus.emit(event)
    
    @pytest.mark.asyncio
    async def test_config_adapter_compatibility(self):
        """Test backward compatibility adapter with new config."""
        # Create new-style settings with proper CacheSettings
        from fraudlens.config.settings import CacheSettings
        
        settings = FraudLensSettings(
            environment="staging",
            cache=CacheSettings(max_size=2000, ttl_seconds=7200)
        )
        
        # Wrap in adapter
        adapter = ConfigAdapter(settings)
        
        # Test old-style access
        assert adapter.get("cache.max_size") == 2000
        assert adapter.get("cache.ttl_seconds") == 7200
        assert adapter.get("processors.text.enabled") == True
        
        # Test set
        adapter.set("cache.max_size", 3000)
        assert adapter.get("cache.max_size") == 3000
    
    @pytest.mark.asyncio
    async def test_event_metrics_with_multiple_event_types(self):
        """Test metrics middleware tracking multiple event types."""
        bus = EventBus()
        metrics = MetricsMiddleware()
        bus.add_middleware(metrics)
        
        # Subscribe handlers
        async def handler(event):
            pass
        
        bus.subscribe_all(handler)
        
        # Emit various events
        events = [
            DetectionStartedEvent(
                source="test",
                input_id=f"test-{i}",
                modality="text"
            )
            for i in range(5)
        ] + [
            AnalysisCompleteEvent(
                source="test",
                input_id=f"test-{i}",
                fraud_score=0.5,
                fraud_types=[],
                processing_time_ms=100.0
            )
            for i in range(3)
        ]
        
        for event in events:
            await bus.emit(event)
        
        # Check metrics
        stats = metrics.get_metrics()
        
        assert stats["total_events"] == 8
        assert stats["event_counts"]["DetectionStartedEvent"] == 5
        assert stats["event_counts"]["AnalysisCompleteEvent"] == 3
    
    @pytest.mark.asyncio
    async def test_full_stack_simulation(self):
        """Simulate a full fraud detection workflow using Phase 1 components."""
        # Setup - Create all components
        settings = FraudLensSettings(environment="dev")
        bus = EventBus()
        container = Container()
        
        # Register in container
        container.register_singleton(FraudLensSettings, instance=settings)
        container.register_singleton(EventBus, instance=bus)
        
        # Add metrics middleware
        metrics = MetricsMiddleware()
        bus.add_middleware(metrics)
        
        # Setup event handlers to simulate workflow
        workflow_state = {
            "detection_started": False,
            "analysis_complete": False,
        }
        
        async def on_detection_started(event: DetectionStartedEvent):
            workflow_state["detection_started"] = True
        
        async def on_analysis_complete(event: AnalysisCompleteEvent):
            workflow_state["analysis_complete"] = True
        
        bus.subscribe(DetectionStartedEvent, on_detection_started)
        bus.subscribe(AnalysisCompleteEvent, on_analysis_complete)
        
        # Simulate workflow
        # 1. Detection starts
        start_event = DetectionStartedEvent(
            source="fraud_detector",
            input_id="workflow-123",
            modality="text",
            priority=EventPriority.HIGH
        )
        await bus.emit(start_event)
        
        # 2. Analysis completes
        complete_event = AnalysisCompleteEvent(
            source="fraud_detector",
            input_id="workflow-123",
            fraud_score=0.85,
            fraud_types=["phishing", "social_engineering"],
            processing_time_ms=250.0,
            priority=EventPriority.HIGH
        )
        await bus.emit(complete_event)
        
        # Verify workflow executed
        assert workflow_state["detection_started"] == True
        assert workflow_state["analysis_complete"] == True
        
        # Verify metrics
        stats = metrics.get_metrics()
        assert stats["total_events"] == 2
        
        # Verify components are still accessible
        resolved_settings = container.resolve(FraudLensSettings)
        assert resolved_settings.environment == "dev"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
