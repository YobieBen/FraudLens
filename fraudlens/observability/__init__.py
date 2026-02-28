"""
Observability & Telemetry for FraudLens.

Provides metrics, logging, tracing, and health checks for
production monitoring and debugging.

Author: Yobie Benjamin
Date: 2026-02-28
"""

from fraudlens.observability.health import (
    HealthCheck,
    HealthCheckManager,
    HealthCheckResult,
    HealthStatus,
    get_health_manager,
    reset_health_manager,
)
from fraudlens.observability.logging import (
    AuditLogger,
    LogContext,
    get_audit_logger,
    get_log_context,
    get_logger,
    log_context,
    setup_logging,
    set_log_context,
)
from fraudlens.observability.metrics import (
    MetricsCollector,
    MetricLabels,
    get_metrics_collector,
    reset_metrics,
)
from fraudlens.observability.tracing import (
    SpanHelper,
    TracingConfig,
    TracingContext,
    get_tracer,
    get_tracing_context,
    setup_tracing,
    trace_function,
    trace_operation,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    "log_context",
    "get_log_context",
    "set_log_context",
    "LogContext",
    "AuditLogger",
    "get_audit_logger",
    # Metrics
    "MetricsCollector",
    "MetricLabels",
    "get_metrics_collector",
    "reset_metrics",
    # Tracing
    "setup_tracing",
    "get_tracer",
    "trace_operation",
    "trace_function",
    "TracingConfig",
    "TracingContext",
    "get_tracing_context",
    "SpanHelper",
    # Health
    "HealthCheck",
    "HealthCheckManager",
    "HealthCheckResult",
    "HealthStatus",
    "get_health_manager",
    "reset_health_manager",
]
