"""
Distributed tracing for FraudLens using OpenTelemetry.

Enables request tracing across agents, LLM calls, and tools
for debugging complex fraud detection workflows.

Author: Yobie Benjamin
Date: 2026-02-28
"""

import functools
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Status, StatusCode, Span
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


class TracingConfig:
    """Configuration for distributed tracing."""
    
    def __init__(
        self,
        service_name: str = "fraudlens",
        service_version: str = "0.1.0",
        environment: str = "development",
        endpoint: Optional[str] = None,
        console_export: bool = False,
    ):
        """
        Initialize tracing config.
        
        Args:
            service_name: Name of the service
            service_version: Version of the service
            environment: Environment (dev/staging/prod)
            endpoint: OTLP endpoint (e.g., "http://localhost:4317")
            console_export: Whether to export to console
        """
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.endpoint = endpoint
        self.console_export = console_export


def setup_tracing(config: Optional[TracingConfig] = None) -> TracerProvider:
    """
    Setup distributed tracing with OpenTelemetry.
    
    Args:
        config: Tracing configuration
    
    Returns:
        Configured TracerProvider
    
    Example:
        ```python
        from fraudlens.observability import setup_tracing
        
        setup_tracing(TracingConfig(
            service_name="fraudlens",
            environment="production",
            endpoint="http://jaeger:4317"
        ))
        ```
    """
    config = config or TracingConfig()
    
    # Create resource with service metadata
    resource = Resource.create({
        "service.name": config.service_name,
        "service.version": config.service_version,
        "deployment.environment": config.environment,
    })
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    
    # Add OTLP exporter if endpoint provided
    if config.endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=config.endpoint)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    
    # Add console exporter if enabled (useful for development)
    if config.console_export:
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(console_exporter))
    
    # Set as global tracer provider
    trace.set_tracer_provider(provider)
    
    return provider


def get_tracer(name: str = __name__) -> trace.Tracer:
    """
    Get a tracer instance.
    
    Args:
        name: Tracer name (typically module name)
    
    Returns:
        Tracer instance
    """
    return trace.get_tracer(name)


@contextmanager
def trace_operation(
    operation_name: str,
    attributes: Optional[Dict[str, Any]] = None,
    tracer: Optional[trace.Tracer] = None
):
    """
    Context manager to trace an operation.
    
    Args:
        operation_name: Name of the operation
        attributes: Additional span attributes
        tracer: Tracer to use (None = default)
    
    Example:
        ```python
        with trace_operation("analyze_fraud", {"agent_id": "phishing"}):
            result = await agent.analyze(data)
        ```
    """
    tracer = tracer or get_tracer()
    
    with tracer.start_as_current_span(operation_name) as span:
        if attributes:
            span.set_attributes(attributes)
        
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def trace_function(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Decorator to automatically trace a function.
    
    Args:
        name: Span name (defaults to function name)
        attributes: Additional span attributes
    
    Example:
        ```python
        @trace_function(attributes={"component": "agent"})
        async def analyze_fraud(data):
            return await process(data)
        ```
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with trace_operation(span_name, attributes):
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with trace_operation(span_name, attributes):
                return func(*args, **kwargs)
        
        # Return appropriate wrapper
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class TracingContext:
    """
    Helper class for managing tracing context.
    
    Provides utilities for extracting and injecting trace context
    for distributed tracing across service boundaries.
    """
    
    def __init__(self):
        """Initialize tracing context."""
        self.propagator = TraceContextTextMapPropagator()
    
    def extract_context(self, carrier: Dict[str, str]) -> Any:
        """
        Extract trace context from carrier (e.g., HTTP headers).
        
        Args:
            carrier: Dictionary containing trace headers
        
        Returns:
            Trace context
        """
        return self.propagator.extract(carrier)
    
    def inject_context(self, carrier: Dict[str, str]) -> None:
        """
        Inject current trace context into carrier.
        
        Args:
            carrier: Dictionary to inject headers into
        """
        self.propagator.inject(carrier)
    
    def get_current_trace_id(self) -> Optional[str]:
        """Get current trace ID as hex string."""
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return format(span.get_span_context().trace_id, '032x')
        return None
    
    def get_current_span_id(self) -> Optional[str]:
        """Get current span ID as hex string."""
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return format(span.get_span_context().span_id, '016x')
        return None


class SpanHelper:
    """
    Helper methods for working with spans.
    
    Provides convenience methods for adding common attributes
    and events to spans.
    """
    
    @staticmethod
    def add_agent_attributes(span: Span, agent_id: str, agent_type: str) -> None:
        """Add agent-specific attributes to span."""
        span.set_attributes({
            "fraudlens.agent.id": agent_id,
            "fraudlens.agent.type": agent_type,
        })
    
    @staticmethod
    def add_fraud_detection_attributes(
        span: Span,
        fraud_detected: bool,
        fraud_score: float,
        confidence: float,
        fraud_types: list[str]
    ) -> None:
        """Add fraud detection attributes to span."""
        span.set_attributes({
            "fraudlens.fraud.detected": fraud_detected,
            "fraudlens.fraud.score": fraud_score,
            "fraudlens.fraud.confidence": confidence,
            "fraudlens.fraud.types": ",".join(fraud_types),
        })
    
    @staticmethod
    def add_llm_attributes(
        span: Span,
        provider: str,
        model: str,
        tokens: int,
        cost_usd: float
    ) -> None:
        """Add LLM request attributes to span."""
        span.set_attributes({
            "fraudlens.llm.provider": provider,
            "fraudlens.llm.model": model,
            "fraudlens.llm.tokens": tokens,
            "fraudlens.llm.cost_usd": cost_usd,
        })
    
    @staticmethod
    def add_tool_attributes(span: Span, tool_name: str, duration_ms: float) -> None:
        """Add tool execution attributes to span."""
        span.set_attributes({
            "fraudlens.tool.name": tool_name,
            "fraudlens.tool.duration_ms": duration_ms,
        })
    
    @staticmethod
    def add_consensus_attributes(
        span: Span,
        num_agents: int,
        variance: float,
        agreement_factor: float
    ) -> None:
        """Add consensus attributes to span."""
        span.set_attributes({
            "fraudlens.consensus.num_agents": num_agents,
            "fraudlens.consensus.variance": variance,
            "fraudlens.consensus.agreement_factor": agreement_factor,
        })
    
    @staticmethod
    def record_evidence_found(span: Span, evidence_type: str, confidence: float) -> None:
        """Record evidence found event."""
        span.add_event(
            "evidence_found",
            attributes={
                "evidence.type": evidence_type,
                "evidence.confidence": confidence,
            }
        )
    
    @staticmethod
    def record_retry(span: Span, attempt: int, reason: str) -> None:
        """Record retry attempt event."""
        span.add_event(
            "retry_attempt",
            attributes={
                "retry.attempt": attempt,
                "retry.reason": reason,
            }
        )
    
    @staticmethod
    def record_circuit_breaker_event(span: Span, event: str, provider: str) -> None:
        """Record circuit breaker event."""
        span.add_event(
            "circuit_breaker_event",
            attributes={
                "circuit_breaker.event": event,
                "circuit_breaker.provider": provider,
            }
        )


# Global tracing context instance
_tracing_context: Optional[TracingContext] = None


def get_tracing_context() -> TracingContext:
    """Get global tracing context."""
    global _tracing_context
    
    if _tracing_context is None:
        _tracing_context = TracingContext()
    
    return _tracing_context
