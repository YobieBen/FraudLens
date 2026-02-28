"""
Metrics collection and monitoring for FraudLens.

Provides Prometheus-compatible metrics for tracking fraud detection
performance, accuracy, and system health.

Author: Yobie Benjamin
Date: 2026-02-28
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    Summary,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


class MetricType(str, Enum):
    """Types of metrics."""
    
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class MetricLabels:
    """Standard labels for fraud detection metrics."""
    
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None
    fraud_type: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    tool_name: Optional[str] = None
    result: Optional[str] = None  # "success", "error", "timeout"
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class MetricsCollector:
    """
    Central metrics collector for FraudLens.
    
    Provides Prometheus-compatible metrics with thread-safe updates.
    All metrics are automatically registered with the global registry.
    
    Example:
        ```python
        metrics = get_metrics_collector()
        
        # Increment counter
        metrics.fraud_detections_total.labels(
            fraud_type="phishing",
            result="success"
        ).inc()
        
        # Record histogram
        with metrics.agent_execution_seconds.labels(
            agent_id="phishing_agent"
        ).time():
            result = await agent.analyze(data)
        ```
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics collector.
        
        Args:
            registry: Prometheus registry (None = global registry)
        """
        self.registry = registry
        self._lock = Lock()
        
        # Fraud Detection Metrics
        self.fraud_detections_total = Counter(
            "fraudlens_fraud_detections_total",
            "Total number of fraud detection requests",
            ["agent_id", "fraud_type", "result"],
            registry=registry
        )
        
        self.fraud_score_distribution = Histogram(
            "fraudlens_fraud_score",
            "Distribution of fraud scores",
            ["agent_id"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=registry
        )
        
        self.fraud_confidence_distribution = Histogram(
            "fraudlens_confidence_score",
            "Distribution of confidence scores",
            ["agent_id"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=registry
        )
        
        self.true_positives = Counter(
            "fraudlens_true_positives_total",
            "True positive fraud detections",
            ["agent_id", "fraud_type"],
            registry=registry
        )
        
        self.false_positives = Counter(
            "fraudlens_false_positives_total",
            "False positive fraud detections",
            ["agent_id", "fraud_type"],
            registry=registry
        )
        
        self.true_negatives = Counter(
            "fraudlens_true_negatives_total",
            "True negative fraud detections",
            ["agent_id"],
            registry=registry
        )
        
        self.false_negatives = Counter(
            "fraudlens_false_negatives_total",
            "False negative fraud detections",
            ["agent_id", "fraud_type"],
            registry=registry
        )
        
        # Performance Metrics
        self.request_latency_seconds = Histogram(
            "fraudlens_request_duration_seconds",
            "Request latency in seconds",
            ["agent_id", "result"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=registry
        )
        
        self.agent_execution_seconds = Histogram(
            "fraudlens_agent_execution_seconds",
            "Agent execution time in seconds",
            ["agent_id", "agent_type"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=registry
        )
        
        self.llm_api_latency_seconds = Histogram(
            "fraudlens_llm_api_duration_seconds",
            "LLM API call latency in seconds",
            ["provider", "model", "result"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=registry
        )
        
        self.tool_execution_seconds = Histogram(
            "fraudlens_tool_execution_seconds",
            "Tool execution time in seconds",
            ["tool_name", "result"],
            buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=registry
        )
        
        # Throughput Metrics
        self.requests_per_second = Summary(
            "fraudlens_requests_per_second",
            "Request throughput",
            ["agent_id"],
            registry=registry
        )
        
        # Agent Metrics
        self.active_agents = Gauge(
            "fraudlens_active_agents",
            "Number of active agents",
            ["agent_type"],
            registry=registry
        )
        
        self.agent_errors_total = Counter(
            "fraudlens_agent_errors_total",
            "Total agent errors",
            ["agent_id", "error_type"],
            registry=registry
        )
        
        # Consensus Metrics
        self.consensus_variance = Histogram(
            "fraudlens_consensus_variance",
            "Variance in agent consensus",
            buckets=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
            registry=registry
        )
        
        self.consensus_agreement_factor = Histogram(
            "fraudlens_consensus_agreement_factor",
            "Agreement factor in consensus",
            buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
            registry=registry
        )
        
        # LLM Provider Metrics
        self.llm_tokens_total = Counter(
            "fraudlens_llm_tokens_total",
            "Total LLM tokens used",
            ["provider", "model", "type"],  # type: "input" or "output"
            registry=registry
        )
        
        self.llm_requests_total = Counter(
            "fraudlens_llm_requests_total",
            "Total LLM requests",
            ["provider", "model", "result"],
            registry=registry
        )
        
        self.llm_cost_usd_total = Counter(
            "fraudlens_llm_cost_usd_total",
            "Total LLM cost in USD",
            ["provider", "model"],
            registry=registry
        )
        
        self.llm_retries_total = Counter(
            "fraudlens_llm_retries_total",
            "Total LLM retry attempts",
            ["provider", "model", "reason"],
            registry=registry
        )
        
        # Circuit Breaker Metrics
        self.circuit_breaker_state = Gauge(
            "fraudlens_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=open, 2=half_open)",
            ["provider"],
            registry=registry
        )
        
        self.circuit_breaker_failures = Counter(
            "fraudlens_circuit_breaker_failures_total",
            "Circuit breaker failures",
            ["provider"],
            registry=registry
        )
        
        # Event Bus Metrics
        self.events_published_total = Counter(
            "fraudlens_events_published_total",
            "Total events published",
            ["event_type"],
            registry=registry
        )
        
        self.events_processed_total = Counter(
            "fraudlens_events_processed_total",
            "Total events processed",
            ["event_type", "handler"],
            registry=registry
        )
        
        # System Info
        self.build_info = Info(
            "fraudlens_build",
            "FraudLens build information",
            registry=registry
        )
    
    def record_fraud_detection(
        self,
        agent_id: str,
        fraud_detected: bool,
        fraud_score: float,
        confidence: float,
        fraud_types: List[str],
        latency_seconds: float,
        result: str = "success"
    ) -> None:
        """
        Record a fraud detection event.
        
        Args:
            agent_id: Agent identifier
            fraud_detected: Whether fraud was detected
            fraud_score: Fraud score (0-1)
            confidence: Confidence score (0-1)
            fraud_types: List of fraud types detected
            latency_seconds: Request latency
            result: Result status ("success", "error", etc.)
        """
        with self._lock:
            # Record detection
            for fraud_type in fraud_types or ["none"]:
                self.fraud_detections_total.labels(
                    agent_id=agent_id,
                    fraud_type=fraud_type,
                    result=result
                ).inc()
            
            # Record scores
            self.fraud_score_distribution.labels(agent_id=agent_id).observe(fraud_score)
            self.fraud_confidence_distribution.labels(agent_id=agent_id).observe(confidence)
            
            # Record latency
            self.request_latency_seconds.labels(
                agent_id=agent_id,
                result=result
            ).observe(latency_seconds)
    
    def record_agent_execution(
        self,
        agent_id: str,
        agent_type: str,
        duration_seconds: float,
        result: str = "success"
    ) -> None:
        """Record agent execution metrics."""
        self.agent_execution_seconds.labels(
            agent_id=agent_id,
            agent_type=agent_type
        ).observe(duration_seconds)
    
    def record_llm_request(
        self,
        provider: str,
        model: str,
        duration_seconds: float,
        tokens_used: int,
        cost_usd: float,
        result: str = "success"
    ) -> None:
        """Record LLM API request metrics."""
        with self._lock:
            self.llm_api_latency_seconds.labels(
                provider=provider,
                model=model,
                result=result
            ).observe(duration_seconds)
            
            self.llm_requests_total.labels(
                provider=provider,
                model=model,
                result=result
            ).inc()
            
            self.llm_tokens_total.labels(
                provider=provider,
                model=model,
                type="total"
            ).inc(tokens_used)
            
            self.llm_cost_usd_total.labels(
                provider=provider,
                model=model
            ).inc(cost_usd)
    
    def record_consensus(
        self,
        variance: float,
        agreement_factor: float
    ) -> None:
        """Record consensus building metrics."""
        self.consensus_variance.observe(variance)
        self.consensus_agreement_factor.observe(agreement_factor)
    
    @contextmanager
    def time_operation(self, metric: Histogram, **labels):
        """
        Context manager to time an operation.
        
        Example:
            with metrics.time_operation(metrics.agent_execution_seconds, agent_id="test"):
                perform_work()
        """
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            metric.labels(**labels).observe(duration)
    
    def export_metrics(self) -> bytes:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Metrics in Prometheus text format
        """
        return generate_latest(self.registry)
    
    def get_content_type(self) -> str:
        """Get content type for metrics endpoint."""
        return CONTENT_TYPE_LATEST


# Global metrics collector instance
_global_metrics: Optional[MetricsCollector] = None
_metrics_lock = Lock()


def get_metrics_collector() -> MetricsCollector:
    """
    Get the global metrics collector.
    
    Returns:
        Global MetricsCollector instance
    """
    global _global_metrics
    
    if _global_metrics is None:
        with _metrics_lock:
            if _global_metrics is None:
                _global_metrics = MetricsCollector()
    
    return _global_metrics


def reset_metrics() -> None:
    """Reset the global metrics collector (useful for testing)."""
    global _global_metrics
    
    with _metrics_lock:
        _global_metrics = None
