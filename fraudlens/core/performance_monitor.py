"""
FraudLens Performance Monitor
Real-time performance monitoring and metrics collection
"""

import time
import psutil
import asyncio
from typing import Dict, Any, List, Optional, Deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from threading import Lock
import statistics
from loguru import logger


@dataclass
class PerformanceMetric:
    """Performance metric data point"""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    active_tasks: int
    cache_hit_rate: float
    requests_per_second: float
    avg_response_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": round(self.cpu_percent, 1),
            "memory_percent": round(self.memory_percent, 1),
            "memory_mb": round(self.memory_mb, 1),
            "disk_io_read_mb": round(self.disk_io_read_mb, 2),
            "disk_io_write_mb": round(self.disk_io_write_mb, 2),
            "network_sent_mb": round(self.network_sent_mb, 2),
            "network_recv_mb": round(self.network_recv_mb, 2),
            "active_tasks": self.active_tasks,
            "cache_hit_rate": round(self.cache_hit_rate, 1),
            "requests_per_second": round(self.requests_per_second, 2),
            "avg_response_time_ms": round(self.avg_response_time_ms, 1),
        }


@dataclass
class OperationMetrics:
    """Metrics for a specific operation type"""

    operation_name: str
    total_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_time_ms: float = 0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0
    response_times: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_count == 0:
            return 0.0
        return (self.success_count / self.total_count) * 100

    @property
    def avg_time_ms(self) -> float:
        """Calculate average response time"""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)

    @property
    def p95_time_ms(self) -> float:
        """Calculate 95th percentile response time"""
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        index = int(len(sorted_times) * 0.95)
        return sorted_times[min(index, len(sorted_times) - 1)]

    def record_operation(self, success: bool, time_ms: float):
        """Record an operation"""
        self.total_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        self.total_time_ms += time_ms
        self.min_time_ms = min(self.min_time_ms, time_ms)
        self.max_time_ms = max(self.max_time_ms, time_ms)
        self.response_times.append(time_ms)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "operation_name": self.operation_name,
            "total_count": self.total_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": round(self.success_rate, 1),
            "avg_time_ms": round(self.avg_time_ms, 1),
            "min_time_ms": round(self.min_time_ms, 1),
            "max_time_ms": round(self.max_time_ms, 1),
            "p95_time_ms": round(self.p95_time_ms, 1),
        }


class PerformanceMonitor:
    """System performance monitor"""

    def __init__(self, history_size: int = 1000):
        """Initialize performance monitor"""
        self.history_size = history_size
        self.metrics_history: Deque[PerformanceMetric] = deque(maxlen=history_size)
        self.operation_metrics: Dict[str, OperationMetrics] = {}
        self.lock = Lock()

        # Initialize system monitors
        self.process = psutil.Process()
        self.last_disk_io = psutil.disk_io_counters()
        self.last_network_io = psutil.net_io_counters()
        self.last_sample_time = time.time()

        # Request tracking
        self.request_times: Deque[float] = deque(maxlen=1000)
        self.response_times: Deque[float] = deque(maxlen=1000)

        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0

        # Active tasks
        self.active_tasks = 0

        logger.info("Performance monitor initialized")

    def collect_metrics(self) -> PerformanceMetric:
        """Collect current system metrics"""
        current_time = time.time()
        time_delta = current_time - self.last_sample_time

        # CPU and Memory
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        memory_mb = memory_info.rss / (1024 * 1024)

        # Disk I/O
        current_disk_io = psutil.disk_io_counters()
        disk_read_mb = (current_disk_io.read_bytes - self.last_disk_io.read_bytes) / (1024 * 1024)
        disk_write_mb = (current_disk_io.write_bytes - self.last_disk_io.write_bytes) / (
            1024 * 1024
        )
        self.last_disk_io = current_disk_io

        # Network I/O
        current_network_io = psutil.net_io_counters()
        network_sent_mb = (current_network_io.bytes_sent - self.last_network_io.bytes_sent) / (
            1024 * 1024
        )
        network_recv_mb = (current_network_io.bytes_recv - self.last_network_io.bytes_recv) / (
            1024 * 1024
        )
        self.last_network_io = current_network_io

        # Calculate rates
        if time_delta > 0:
            disk_read_mb /= time_delta
            disk_write_mb /= time_delta
            network_sent_mb /= time_delta
            network_recv_mb /= time_delta

        # Request metrics
        requests_per_second = len([t for t in self.request_times if current_time - t < 1.0])
        avg_response_time = statistics.mean(self.response_times) if self.response_times else 0

        # Cache metrics
        total_cache_ops = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_cache_ops * 100) if total_cache_ops > 0 else 0

        metric = PerformanceMetric(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_mb=memory_mb,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            active_tasks=self.active_tasks,
            cache_hit_rate=cache_hit_rate,
            requests_per_second=requests_per_second,
            avg_response_time_ms=avg_response_time,
        )

        self.last_sample_time = current_time

        with self.lock:
            self.metrics_history.append(metric)

        return metric

    def record_request(self):
        """Record a new request"""
        with self.lock:
            self.request_times.append(time.time())

    def record_response(self, response_time_ms: float):
        """Record a response time"""
        with self.lock:
            self.response_times.append(response_time_ms)

    def record_cache_hit(self):
        """Record a cache hit"""
        with self.lock:
            self.cache_hits += 1

    def record_cache_miss(self):
        """Record a cache miss"""
        with self.lock:
            self.cache_misses += 1

    def increment_active_tasks(self):
        """Increment active task count"""
        with self.lock:
            self.active_tasks += 1

    def decrement_active_tasks(self):
        """Decrement active task count"""
        with self.lock:
            self.active_tasks = max(0, self.active_tasks - 1)

    def record_operation(self, operation_name: str, success: bool, time_ms: float):
        """Record an operation's metrics"""
        with self.lock:
            if operation_name not in self.operation_metrics:
                self.operation_metrics[operation_name] = OperationMetrics(operation_name)

            self.operation_metrics[operation_name].record_operation(success, time_ms)

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        metric = self.collect_metrics()
        return metric.to_dict()

    def get_metrics_history(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """Get metrics history for the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)

        with self.lock:
            recent_metrics = [
                m.to_dict() for m in self.metrics_history if m.timestamp > cutoff_time
            ]

        return recent_metrics

    def get_operation_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all operations"""
        with self.lock:
            return {name: metrics.to_dict() for name, metrics in self.operation_metrics.items()}

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "disk_usage_percent": psutil.disk_usage("/").percent,
            "python_version": f"{psutil.Process().name()}",
            "uptime_hours": (time.time() - psutil.boot_time()) / 3600,
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        current_metrics = self.collect_metrics()

        # Define health thresholds
        health_status = "healthy"
        warnings = []

        if current_metrics.cpu_percent > 80:
            warnings.append("High CPU usage")
            health_status = "warning"

        if current_metrics.memory_percent > 85:
            warnings.append("High memory usage")
            health_status = "warning"

        if current_metrics.cpu_percent > 95 or current_metrics.memory_percent > 95:
            health_status = "critical"

        return {
            "status": health_status,
            "warnings": warnings,
            "metrics": current_metrics.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }

    async def start_monitoring(self, interval_seconds: int = 5):
        """Start continuous monitoring"""
        logger.info(f"Starting performance monitoring with {interval_seconds}s interval")

        while True:
            try:
                self.collect_metrics()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(interval_seconds)


# Singleton instance
performance_monitor = PerformanceMonitor()


# Context manager for operation timing
class OperationTimer:
    """Context manager for timing operations"""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.success = True

    def __enter__(self):
        self.start_time = time.time()
        performance_monitor.increment_active_tasks()
        performance_monitor.record_request()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.time() - self.start_time) * 1000
        self.success = exc_type is None

        performance_monitor.record_operation(self.operation_name, self.success, elapsed_ms)
        performance_monitor.record_response(elapsed_ms)
        performance_monitor.decrement_active_tasks()

        if not self.success:
            logger.error(f"Operation {self.operation_name} failed: {exc_val}")


# Decorator for automatic performance tracking
def track_performance(operation_name: Optional[str] = None):
    """Decorator to track operation performance"""

    def decorator(func):
        op_name = operation_name or func.__name__

        def sync_wrapper(*args, **kwargs):
            with OperationTimer(op_name):
                return func(*args, **kwargs)

        async def async_wrapper(*args, **kwargs):
            with OperationTimer(op_name):
                return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
