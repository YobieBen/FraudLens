"""
Observability decorators and middleware for easy instrumentation.

Provides @traced, @timed, @counted decorators for automatic
metrics and tracing.

Author: Yobie Benjamin
Date: 2026-02-28
"""

import functools
import time
from typing import Any, Callable, Optional

from fraudlens.observability.logging import get_logger, log_context
from fraudlens.observability.metrics import get_metrics_collector
from fraudlens.observability.tracing import get_tracer, trace_operation

logger = get_logger(__name__)


def traced(
    name: Optional[str] = None,
    component: Optional[str] = None,
    **attributes
) -> Callable:
    """
    Decorator to automatically trace a function with OpenTelemetry.
    
    Args:
        name: Span name (defaults to function name)
        component: Component name for attributes
        **attributes: Additional span attributes
    
    Example:
        ```python
        @traced(component="agent")
        async def analyze_fraud(data):
            return await process(data)
        ```
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        
        span_attributes = {**attributes}
        if component:
            span_attributes["component"] = component
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with trace_operation(span_name, span_attributes):
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with trace_operation(span_name, span_attributes):
                return func(*args, **kwargs)
        
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def timed(
    metric_name: Optional[str] = None,
    labels: Optional[dict] = None
) -> Callable:
    """
    Decorator to automatically time function execution.
    
    Records timing metrics to Prometheus.
    
    Args:
        metric_name: Metric name (defaults to function name)
        labels: Additional metric labels
    
    Example:
        ```python
        @timed(labels={"agent_type": "phishing"})
        async def analyze(data):
            return await process(data)
        ```
    """
    def decorator(func: Callable) -> Callable:
        name = metric_name or f"{func.__module__}_{func.__qualname__}"
        metric_labels = labels or {}
        metrics = get_metrics_collector()
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start
                
                # Record successful execution
                metrics.request_latency_seconds.labels(
                    agent_id=metric_labels.get("agent_id", "unknown"),
                    result="success"
                ).observe(duration)
                
                return result
            except Exception as e:
                duration = time.time() - start
                
                # Record failed execution
                metrics.request_latency_seconds.labels(
                    agent_id=metric_labels.get("agent_id", "unknown"),
                    result="error"
                ).observe(duration)
                
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                
                metrics.request_latency_seconds.labels(
                    agent_id=metric_labels.get("agent_id", "unknown"),
                    result="success"
                ).observe(duration)
                
                return result
            except Exception as e:
                duration = time.time() - start
                
                metrics.request_latency_seconds.labels(
                    agent_id=metric_labels.get("agent_id", "unknown"),
                    result="error"
                ).observe(duration)
                
                raise
        
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def counted(
    counter_name: Optional[str] = None,
    labels: Optional[dict] = None
) -> Callable:
    """
    Decorator to count function invocations.
    
    Args:
        counter_name: Counter name (defaults to function name)
        labels: Counter labels
    
    Example:
        ```python
        @counted(labels={"fraud_type": "phishing"})
        async def detect_phishing(data):
            return await analyze(data)
        ```
    """
    def decorator(func: Callable) -> Callable:
        name = counter_name or f"{func.__module__}_{func.__qualname__}_total"
        metric_labels = labels or {}
        metrics = get_metrics_collector()
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                
                # Increment success counter
                metrics.fraud_detections_total.labels(
                    agent_id=metric_labels.get("agent_id", "unknown"),
                    fraud_type=metric_labels.get("fraud_type", "unknown"),
                    result="success"
                ).inc()
                
                return result
            except Exception as e:
                # Increment error counter
                metrics.fraud_detections_total.labels(
                    agent_id=metric_labels.get("agent_id", "unknown"),
                    fraud_type=metric_labels.get("fraud_type", "unknown"),
                    result="error"
                ).inc()
                
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                
                metrics.fraud_detections_total.labels(
                    agent_id=metric_labels.get("agent_id", "unknown"),
                    fraud_type=metric_labels.get("fraud_type", "unknown"),
                    result="success"
                ).inc()
                
                return result
            except Exception as e:
                metrics.fraud_detections_total.labels(
                    agent_id=metric_labels.get("agent_id", "unknown"),
                    fraud_type=metric_labels.get("fraud_type", "unknown"),
                    result="error"
                ).inc()
                
                raise
        
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def logged(
    level: str = "INFO",
    include_args: bool = False,
    include_result: bool = False
) -> Callable:
    """
    Decorator to automatically log function calls.
    
    Args:
        level: Log level
        include_args: Whether to log function arguments
        include_result: Whether to log function result
    
    Example:
        ```python
        @logged(level="DEBUG", include_result=True)
        async def analyze(data):
            return await process(data)
        ```
    """
    def decorator(func: Callable) -> Callable:
        func_logger = get_logger(func.__module__)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            log_data = {"function": func.__qualname__}
            
            if include_args:
                log_data["args"] = str(args)[:100]  # Truncate long args
                log_data["kwargs"] = str(kwargs)[:100]
            
            func_logger.log(level.lower(), "function_called", **log_data)
            
            try:
                result = await func(*args, **kwargs)
                
                if include_result:
                    func_logger.log(
                        level.lower(),
                        "function_completed",
                        function=func.__qualname__,
                        result=str(result)[:100]
                    )
                
                return result
            except Exception as e:
                func_logger.error(
                    "function_failed",
                    function=func.__qualname__,
                    error=str(e)
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            log_data = {"function": func.__qualname__}
            
            if include_args:
                log_data["args"] = str(args)[:100]
                log_data["kwargs"] = str(kwargs)[:100]
            
            func_logger.log(level.lower(), "function_called", **log_data)
            
            try:
                result = func(*args, **kwargs)
                
                if include_result:
                    func_logger.log(
                        level.lower(),
                        "function_completed",
                        function=func.__qualname__,
                        result=str(result)[:100]
                    )
                
                return result
            except Exception as e:
                func_logger.error(
                    "function_failed",
                    function=func.__qualname__,
                    error=str(e)
                )
                raise
        
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def observable(
    component: str,
    agent_id: Optional[str] = None,
    trace: bool = True,
    time_it: bool = True,
    count: bool = True,
    log: bool = True
) -> Callable:
    """
    All-in-one observability decorator.
    
    Combines tracing, timing, counting, and logging.
    
    Args:
        component: Component name
        agent_id: Agent identifier
        trace: Enable tracing
        time_it: Enable timing metrics
        count: Enable counting
        log: Enable logging
    
    Example:
        ```python
        @observable(component="agent", agent_id="phishing_agent")
        async def analyze_fraud(data):
            return await process(data)
        ```
    """
    def decorator(func: Callable) -> Callable:
        # Apply decorators in reverse order
        result = func
        
        if log:
            result = logged(level="INFO")(result)
        
        if count:
            labels = {"agent_id": agent_id or "unknown", "component": component}
            result = counted(labels=labels)(result)
        
        if time_it:
            labels = {"agent_id": agent_id or "unknown"}
            result = timed(labels=labels)(result)
        
        if trace:
            result = traced(component=component)(result)
        
        return result
    
    return decorator
