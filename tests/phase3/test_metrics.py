"""Tests for metrics collector."""

import pytest
from prometheus_client import CollectorRegistry

from fraudlens.observability.metrics import (
    MetricsCollector,
    get_metrics_collector,
    reset_metrics,
)


class TestMetricsCollector:
    """Test MetricsCollector."""
    
    def test_collector_initialization(self):
        """Test collector can be initialized."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)
        
        assert collector.registry == registry
        assert collector.fraud_detections_total is not None
        assert collector.llm_tokens_total is not None
    
    def test_record_fraud_detection(self):
        """Test recording fraud detection metrics."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)
        
        collector.record_fraud_detection(
            agent_id="test_agent",
            fraud_detected=True,
            fraud_score=0.85,
            confidence=0.9,
            fraud_types=["phishing"],
            latency_seconds=0.234
        )
        
        # Verify metrics were recorded
        metrics = collector.export_metrics()
        assert b"fraudlens_fraud_detections_total" in metrics
        assert b"test_agent" in metrics
    
    def test_record_agent_execution(self):
        """Test recording agent execution metrics."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)
        
        collector.record_agent_execution(
            agent_id="test_agent",
            agent_type="phishing",
            duration_seconds=0.5
        )
        
        metrics = collector.export_metrics()
        assert b"fraudlens_agent_execution_seconds" in metrics
    
    def test_record_llm_request(self):
        """Test recording LLM request metrics."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)
        
        collector.record_llm_request(
            provider="anthropic",
            model="claude-3-5-sonnet",
            duration_seconds=1.5,
            tokens_used=1000,
            cost_usd=0.015
        )
        
        metrics = collector.export_metrics()
        assert b"fraudlens_llm_api_duration_seconds" in metrics
        assert b"fraudlens_llm_tokens_total" in metrics
        assert b"fraudlens_llm_cost_usd_total" in metrics
    
    def test_record_consensus(self):
        """Test recording consensus metrics."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)
        
        collector.record_consensus(
            variance=0.05,
            agreement_factor=0.95
        )
        
        metrics = collector.export_metrics()
        assert b"fraudlens_consensus_variance" in metrics
        assert b"fraudlens_consensus_agreement_factor" in metrics
    
    def test_time_operation_context_manager(self):
        """Test time_operation context manager."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)
        
        with collector.time_operation(
            collector.agent_execution_seconds,
            agent_id="test",
            agent_type="test"
        ):
            pass  # Do some work
        
        metrics = collector.export_metrics()
        assert b"fraudlens_agent_execution_seconds" in metrics
    
    def test_metrics_export(self):
        """Test metrics can be exported."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)
        
        metrics = collector.export_metrics()
        assert isinstance(metrics, bytes)
        assert len(metrics) > 0
    
    def test_content_type(self):
        """Test content type for metrics endpoint."""
        collector = MetricsCollector()
        content_type = collector.get_content_type()
        
        assert "text/plain" in content_type or "text" in content_type


class TestGlobalMetrics:
    """Test global metrics collector."""
    
    def test_get_metrics_collector(self):
        """Test getting global metrics collector."""
        reset_metrics()
        
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        # Should return same instance
        assert collector1 is collector2
    
    def test_reset_metrics(self):
        """Test resetting global metrics."""
        collector1 = get_metrics_collector()
        reset_metrics()
        collector2 = get_metrics_collector()
        
        # Should return new instance after reset
        assert collector1 is not collector2
