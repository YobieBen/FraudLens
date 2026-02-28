"""
Structured logging for FraudLens with JSON output and correlation.

Author: Yobie Benjamin
Date: 2026-02-28
"""

import logging
import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

import structlog


class LogLevel(str, Enum):
    """Log levels."""
    
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogContext:
    """Context for structured logging."""
    
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    environment: str = "development"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


# Context variable for log context
_log_context: ContextVar[Optional[LogContext]] = ContextVar("log_context", default=None)


def get_log_context() -> Optional[LogContext]:
    """Get current log context."""
    return _log_context.get()


def set_log_context(context: LogContext) -> None:
    """Set log context."""
    _log_context.set(context)


@contextmanager
def log_context(**kwargs):
    """
    Context manager for enriching logs.
    
    Example:
        with log_context(agent_id="phishing_agent", user_id="user123"):
            logger.info("analyzing_request")
    """
    current = get_log_context() or LogContext()
    
    # Create new context with updates
    new_context = LogContext(**{**current.to_dict(), **kwargs})
    
    token = _log_context.set(new_context)
    try:
        yield new_context
    finally:
        _log_context.reset(token)


def add_context_to_event(logger, method_name, event_dict):
    """Add context to log event."""
    context = get_log_context()
    if context:
        event_dict.update(context.to_dict())
    return event_dict


def add_timestamp(logger, method_name, event_dict):
    """Add ISO timestamp to event."""
    event_dict["timestamp"] = datetime.utcnow().isoformat()
    return event_dict


def setup_logging(
    level: str = "INFO",
    json_output: bool = True,
    environment: str = "development"
) -> structlog.BoundLogger:
    """
    Setup structured logging.
    
    Args:
        level: Log level
        json_output: Whether to output JSON
        environment: Environment name
    
    Returns:
        Configured logger
    """
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )
    
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        add_context_to_event,
        add_timestamp,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Set default context
    default_context = LogContext(environment=environment)
    set_log_context(default_context)
    
    return structlog.get_logger()


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
    
    Returns:
        Configured logger
    """
    return structlog.get_logger(name)


class AuditLogger:
    """
    Specialized logger for audit trails.
    
    Records important fraud detection decisions and actions.
    """
    
    def __init__(self, logger: Optional[structlog.BoundLogger] = None):
        """Initialize audit logger."""
        self.logger = logger or get_logger("fraudlens.audit")
    
    def log_fraud_detection(
        self,
        agent_id: str,
        fraud_detected: bool,
        fraud_score: float,
        confidence: float,
        fraud_types: list[str],
        evidence_count: int,
        action_count: int
    ) -> None:
        """Log fraud detection decision."""
        self.logger.info(
            "fraud_detection",
            agent_id=agent_id,
            fraud_detected=fraud_detected,
            fraud_score=fraud_score,
            confidence=confidence,
            fraud_types=fraud_types,
            evidence_count=evidence_count,
            action_count=action_count
        )
    
    def log_consensus(
        self,
        coordinator_id: str,
        num_agents: int,
        final_score: float,
        variance: float,
        agreement_factor: float
    ) -> None:
        """Log consensus building result."""
        self.logger.info(
            "consensus_built",
            coordinator_id=coordinator_id,
            num_agents=num_agents,
            final_score=final_score,
            variance=variance,
            agreement_factor=agreement_factor
        )
    
    def log_action_taken(
        self,
        action_type: str,
        reason: str,
        automated: bool,
        priority: int
    ) -> None:
        """Log action taken."""
        self.logger.info(
            "action_taken",
            action_type=action_type,
            reason=reason,
            automated=automated,
            priority=priority
        )
    
    def log_llm_request(
        self,
        provider: str,
        model: str,
        duration_ms: float,
        tokens: int,
        cost_usd: float,
        success: bool
    ) -> None:
        """Log LLM API request."""
        self.logger.info(
            "llm_request",
            provider=provider,
            model=model,
            duration_ms=duration_ms,
            tokens=tokens,
            cost_usd=cost_usd,
            success=success
        )
    
    def log_circuit_breaker_event(
        self,
        provider: str,
        event: str,  # "opened", "closed", "half_opened"
        failure_count: int
    ) -> None:
        """Log circuit breaker state change."""
        self.logger.warning(
            "circuit_breaker_event",
            provider=provider,
            event=event,
            failure_count=failure_count
        )
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        component: str,
        **extra
    ) -> None:
        """Log error event."""
        self.logger.error(
            "error_occurred",
            error_type=error_type,
            error_message=error_message,
            component=component,
            **extra
        )


# Global audit logger
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger."""
    global _audit_logger
    
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    
    return _audit_logger
