"""
Monitoring and logging infrastructure for FraudLens.

Author: Yobie Benjamin
Date: 2025
"""

import hashlib
import json
import logging
import os
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
from loguru import logger


@dataclass
class PerformanceMetrics:
    """Performance metrics data."""

    timestamp: datetime
    latency_ms: float
    throughput_qps: float
    cpu_percent: float
    memory_mb: float
    active_requests: int
    error_rate: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
            "throughput_qps": self.throughput_qps,
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "active_requests": self.active_requests,
            "error_rate": self.error_rate,
        }


@dataclass
class ModelDrift:
    """Model drift detection data."""

    model_id: str
    baseline_score: float
    current_score: float
    drift_magnitude: float
    drift_type: str  # performance, distribution, concept
    detected_at: datetime
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "baseline_score": self.baseline_score,
            "current_score": self.current_score,
            "drift_magnitude": self.drift_magnitude,
            "drift_type": self.drift_type,
            "detected_at": self.detected_at.isoformat(),
            "recommendations": self.recommendations,
        }


@dataclass
class Alert:
    """System alert."""

    alert_id: str
    severity: str  # info, warning, error, critical
    category: str  # performance, security, drift, system
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
        }


class StructuredLogger:
    """Structured logging with privacy preservation."""

    def __init__(
        self,
        log_dir: str = "logs",
        log_level: str = "INFO",
        enable_privacy: bool = True,
        max_log_size: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 10,
    ):
        """
        Initialize structured logger.

        Args:
            log_dir: Directory for log files
            log_level: Logging level
            enable_privacy: Enable privacy preservation
            max_log_size: Maximum log file size
            backup_count: Number of backup files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.enable_privacy = enable_privacy

        # Configure loguru
        logger.remove()  # Remove default handler

        # Console handler
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_level,
        )

        # File handler with rotation
        logger.add(
            self.log_dir / "fraudlens_{time}.log",
            rotation=max_log_size,
            retention=backup_count,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            level=log_level,
            serialize=True,  # JSON format
        )

        # Error file handler
        logger.add(
            self.log_dir / "errors_{time}.log",
            rotation=max_log_size,
            retention=backup_count,
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}\n{exception}",
        )

        # Audit log for security events
        logger.add(
            self.log_dir / "audit_{time}.log",
            rotation=max_log_size,
            retention=backup_count,
            filter=lambda record: record["extra"].get("audit", False),
            format="{time:YYYY-MM-DD HH:mm:ss} | AUDIT | {message}",
            serialize=True,
        )

        self.logger = logger

    def log(
        self,
        level: str,
        message: str,
        **kwargs,
    ) -> None:
        """
        Log message with structured data.

        Args:
            level: Log level
            message: Log message
            **kwargs: Additional structured data
        """
        # Privacy preservation
        if self.enable_privacy:
            kwargs = self._sanitize_data(kwargs)

        # Add metadata
        kwargs["timestamp"] = datetime.now().isoformat()
        kwargs["hostname"] = os.uname().nodename

        # Log based on level
        if level == "DEBUG":
            self.logger.debug(message, **kwargs)
        elif level == "INFO":
            self.logger.info(message, **kwargs)
        elif level == "WARNING":
            self.logger.warning(message, **kwargs)
        elif level == "ERROR":
            self.logger.error(message, **kwargs)
        elif level == "CRITICAL":
            self.logger.critical(message, **kwargs)
        else:
            self.logger.info(message, **kwargs)

    def _sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize sensitive data."""
        sanitized = {}

        sensitive_fields = [
            "password",
            "token",
            "api_key",
            "secret",
            "ssn",
            "credit_card",
            "email",
            "phone",
            "address",
            "dob",
            "name",
        ]

        for key, value in data.items():
            # Check if field is sensitive
            if any(sensitive in key.lower() for sensitive in sensitive_fields):
                if isinstance(value, str):
                    # Hash sensitive data
                    sanitized[key] = self._hash_value(value)
                else:
                    sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                # Recursively sanitize nested data
                sanitized[key] = self._sanitize_data(value)
            elif isinstance(value, list):
                # Sanitize list items
                sanitized[key] = [
                    self._sanitize_data(item) if isinstance(item, dict) else item for item in value
                ]
            else:
                sanitized[key] = value

        return sanitized

    def _hash_value(self, value: str) -> str:
        """Hash sensitive value."""
        return f"SHA256:{hashlib.sha256(value.encode()).hexdigest()[:16]}..."

    def audit(self, event: str, user: str, **details) -> None:
        """
        Log audit event.

        Args:
            event: Event description
            user: User identifier
            **details: Event details
        """
        self.logger.info(
            f"AUDIT: {event}",
            audit=True,
            user=user,
            event=event,
            details=details,
        )

    def log_detection(
        self,
        fraud_type: str,
        fraud_score: float,
        confidence: float,
        detector: str,
        **metadata,
    ) -> None:
        """Log fraud detection event."""
        self.log(
            "INFO",
            f"Fraud detected: {fraud_type}",
            fraud_type=fraud_type,
            fraud_score=fraud_score,
            confidence=confidence,
            detector=detector,
            **metadata,
        )

    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log error with traceback."""
        self.logger.exception(
            f"Error occurred: {error}",
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
            context=context or {},
        )


class PerformanceMonitor:
    """Performance metrics collection and monitoring."""

    def __init__(
        self,
        window_size: int = 1000,
        alert_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize performance monitor.

        Args:
            window_size: Size of metrics window
            alert_thresholds: Alert threshold values
        """
        self.window_size = window_size

        # Metrics storage
        self.latencies = deque(maxlen=window_size)
        self.throughput = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)

        # Process monitor
        self.process = psutil.Process()

        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "latency_ms": 1000,  # 1 second
            "error_rate": 0.05,  # 5%
            "cpu_percent": 80,  # 80%
            "memory_mb": 4096,  # 4GB
        }

        # Statistics
        self.total_requests = 0
        self.total_errors = 0
        self.start_time = time.time()

    def record_request(
        self,
        latency_ms: float,
        success: bool = True,
        request_type: Optional[str] = None,
    ) -> None:
        """
        Record request metrics.

        Args:
            latency_ms: Request latency in milliseconds
            success: Whether request succeeded
            request_type: Type of request
        """
        timestamp = time.time()

        self.latencies.append(latency_ms)
        self.errors.append(0 if success else 1)

        self.total_requests += 1
        if not success:
            self.total_errors += 1

        # Calculate throughput
        if len(self.throughput) > 0:
            time_diff = timestamp - self.throughput[-1][0]
            if time_diff > 0:
                qps = 1.0 / time_diff
                self.throughput.append((timestamp, qps))

    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        # Calculate statistics
        if self.latencies:
            avg_latency = np.mean(self.latencies)
        else:
            avg_latency = 0.0

        if self.throughput:
            avg_throughput = np.mean([q for _, q in self.throughput])
        else:
            avg_throughput = 0.0

        if self.errors:
            error_rate = np.mean(self.errors)
        else:
            error_rate = 0.0

        # Get system metrics
        cpu_percent = self.process.cpu_percent()
        memory_mb = self.process.memory_info().rss / 1024 / 1024

        # Active requests (simplified)
        active_requests = 0  # Would need request tracking

        return PerformanceMetrics(
            timestamp=datetime.now(),
            latency_ms=avg_latency,
            throughput_qps=avg_throughput,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            active_requests=active_requests,
            error_rate=error_rate,
        )

    def check_alerts(self) -> List[Alert]:
        """Check for performance alerts."""
        alerts = []
        metrics = self.get_current_metrics()

        # Latency alert
        if metrics.latency_ms > self.alert_thresholds["latency_ms"]:
            alerts.append(
                Alert(
                    alert_id=f"PERF_{int(time.time())}",
                    severity="warning",
                    category="performance",
                    message=f"High latency detected: {metrics.latency_ms:.2f}ms",
                    details={"latency_ms": metrics.latency_ms},
                    timestamp=datetime.now(),
                )
            )

        # Error rate alert
        if metrics.error_rate > self.alert_thresholds["error_rate"]:
            alerts.append(
                Alert(
                    alert_id=f"ERROR_{int(time.time())}",
                    severity="error",
                    category="performance",
                    message=f"High error rate: {metrics.error_rate:.2%}",
                    details={"error_rate": metrics.error_rate},
                    timestamp=datetime.now(),
                )
            )

        # CPU alert
        if metrics.cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append(
                Alert(
                    alert_id=f"CPU_{int(time.time())}",
                    severity="warning",
                    category="system",
                    message=f"High CPU usage: {metrics.cpu_percent:.1f}%",
                    details={"cpu_percent": metrics.cpu_percent},
                    timestamp=datetime.now(),
                )
            )

        # Memory alert
        if metrics.memory_mb > self.alert_thresholds["memory_mb"]:
            alerts.append(
                Alert(
                    alert_id=f"MEM_{int(time.time())}",
                    severity="warning",
                    category="system",
                    message=f"High memory usage: {metrics.memory_mb:.1f}MB",
                    details={"memory_mb": metrics.memory_mb},
                    timestamp=datetime.now(),
                )
            )

        return alerts

    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        uptime = time.time() - self.start_time

        # Calculate percentiles
        if self.latencies:
            latencies_sorted = sorted(self.latencies)
            p50 = np.percentile(latencies_sorted, 50)
            p95 = np.percentile(latencies_sorted, 95)
            p99 = np.percentile(latencies_sorted, 99)
        else:
            p50 = p95 = p99 = 0.0

        return {
            "uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "success_rate": (self.total_requests - self.total_errors) / max(self.total_requests, 1),
            "latency_p50": p50,
            "latency_p95": p95,
            "latency_p99": p99,
            "avg_throughput_qps": (
                np.mean([q for _, q in self.throughput]) if self.throughput else 0
            ),
        }


class ModelDriftDetector:
    """Detect model drift and performance degradation."""

    def __init__(
        self,
        baseline_window: int = 1000,
        drift_threshold: float = 0.1,
    ):
        """
        Initialize drift detector.

        Args:
            baseline_window: Size of baseline window
            drift_threshold: Drift detection threshold
        """
        self.baseline_window = baseline_window
        self.drift_threshold = drift_threshold

        # Score storage
        self.model_scores = {}  # model_id -> deque of scores
        self.baselines = {}  # model_id -> baseline statistics

    def record_prediction(
        self,
        model_id: str,
        prediction_score: float,
        ground_truth: Optional[bool] = None,
    ) -> None:
        """
        Record model prediction.

        Args:
            model_id: Model identifier
            prediction_score: Prediction score
            ground_truth: Actual outcome (if available)
        """
        if model_id not in self.model_scores:
            self.model_scores[model_id] = deque(maxlen=self.baseline_window)

        self.model_scores[model_id].append(
            {
                "score": prediction_score,
                "ground_truth": ground_truth,
                "timestamp": datetime.now(),
            }
        )

        # Update baseline if enough data
        if len(self.model_scores[model_id]) == self.baseline_window:
            self._update_baseline(model_id)

    def _update_baseline(self, model_id: str) -> None:
        """Update baseline statistics for model."""
        scores = [s["score"] for s in self.model_scores[model_id]]

        self.baselines[model_id] = {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "updated": datetime.now(),
        }

    def detect_drift(self, model_id: str) -> Optional[ModelDrift]:
        """
        Detect drift for a model.

        Args:
            model_id: Model identifier

        Returns:
            Drift detection result
        """
        if model_id not in self.baselines:
            return None

        if model_id not in self.model_scores:
            return None

        # Get recent scores
        recent_scores = [s["score"] for s in list(self.model_scores[model_id])[-100:]]
        if len(recent_scores) < 10:
            return None

        # Calculate statistics
        baseline = self.baselines[model_id]
        current_mean = np.mean(recent_scores)
        current_std = np.std(recent_scores)

        # Detect drift types
        drift_detected = False
        drift_type = None
        recommendations = []

        # Performance drift (mean shift)
        mean_drift = abs(current_mean - baseline["mean"])
        if mean_drift > self.drift_threshold:
            drift_detected = True
            drift_type = "performance"
            recommendations.append(f"Performance drift detected: mean shifted by {mean_drift:.3f}")
            recommendations.append("Consider retraining the model")

        # Distribution drift (variance change)
        std_drift = abs(current_std - baseline["std"])
        if std_drift > baseline["std"] * 0.5:  # 50% change in std
            drift_detected = True
            if not drift_type:
                drift_type = "distribution"
            recommendations.append(f"Distribution drift detected: std changed by {std_drift:.3f}")
            recommendations.append("Check for changes in input distribution")

        # Concept drift (if ground truth available)
        with_truth = [s for s in self.model_scores[model_id] if s["ground_truth"] is not None]
        if len(with_truth) > 20:
            recent_accuracy = np.mean(
                [(s["score"] > 0.5) == s["ground_truth"] for s in with_truth[-20:]]
            )

            if recent_accuracy < 0.7:  # Below 70% accuracy
                drift_detected = True
                if not drift_type:
                    drift_type = "concept"
                recommendations.append(
                    f"Concept drift detected: accuracy dropped to {recent_accuracy:.2%}"
                )
                recommendations.append("Retrain model with recent data")

        if drift_detected:
            return ModelDrift(
                model_id=model_id,
                baseline_score=baseline["mean"],
                current_score=current_mean,
                drift_magnitude=mean_drift,
                drift_type=drift_type or "unknown",
                detected_at=datetime.now(),
                recommendations=recommendations,
            )

        return None

    def get_model_health(self, model_id: str) -> Dict[str, Any]:
        """
        Get model health status.

        Args:
            model_id: Model identifier

        Returns:
            Health status dictionary
        """
        if model_id not in self.model_scores:
            return {"status": "no_data"}

        scores = [s["score"] for s in self.model_scores[model_id]]

        health = {
            "status": "healthy",
            "total_predictions": len(scores),
            "recent_mean": np.mean(scores[-100:]) if len(scores) > 100 else np.mean(scores),
            "recent_std": np.std(scores[-100:]) if len(scores) > 100 else np.std(scores),
        }

        # Check for drift
        drift = self.detect_drift(model_id)
        if drift:
            health["status"] = "drift_detected"
            health["drift"] = drift.to_dict()

        return health


class FraudLensMonitor:
    """Main monitoring system for FraudLens."""

    def __init__(
        self,
        log_dir: str = "logs",
        metrics_dir: str = "metrics",
        enable_debug: bool = False,
    ):
        """
        Initialize monitoring system.

        Args:
            log_dir: Directory for logs
            metrics_dir: Directory for metrics
            enable_debug: Enable debug mode
        """
        # Initialize components
        self.logger = StructuredLogger(
            log_dir=log_dir,
            log_level="DEBUG" if enable_debug else "INFO",
        )

        self.performance_monitor = PerformanceMonitor()
        self.drift_detector = ModelDriftDetector()

        # Metrics storage
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(exist_ok=True)

        # Alert queue
        self.alerts = deque(maxlen=1000)

        # Start monitoring
        self.start_time = datetime.now()
        self.is_running = True

        self.logger.log("INFO", "FraudLens monitoring system started")

    def log(self, level: str, message: str, **kwargs) -> None:
        """Log message."""
        self.logger.log(level, message, **kwargs)

    def record_detection(
        self,
        detector_id: str,
        fraud_score: float,
        latency_ms: float,
        success: bool = True,
        **metadata,
    ) -> None:
        """
        Record fraud detection event.

        Args:
            detector_id: Detector identifier
            fraud_score: Fraud score
            latency_ms: Processing latency
            success: Whether detection succeeded
            **metadata: Additional metadata
        """
        # Log detection
        self.logger.log_detection(
            fraud_type=metadata.get("fraud_type", "unknown"),
            fraud_score=fraud_score,
            confidence=metadata.get("confidence", 0.0),
            detector=detector_id,
            latency_ms=latency_ms,
        )

        # Record performance
        self.performance_monitor.record_request(
            latency_ms=latency_ms,
            success=success,
            request_type="detection",
        )

        # Record for drift detection
        self.drift_detector.record_prediction(
            model_id=detector_id,
            prediction_score=fraud_score,
            ground_truth=metadata.get("ground_truth"),
        )

        # Check for alerts
        self._check_alerts()

    def _check_alerts(self) -> None:
        """Check and handle alerts."""
        # Performance alerts
        perf_alerts = self.performance_monitor.check_alerts()
        for alert in perf_alerts:
            self.alerts.append(alert)
            self.logger.log(
                alert.severity.upper(),
                alert.message,
                alert_id=alert.alert_id,
                category=alert.category,
                details=alert.details,
            )

        # Drift alerts
        for model_id in self.drift_detector.model_scores.keys():
            drift = self.drift_detector.detect_drift(model_id)
            if drift:
                alert = Alert(
                    alert_id=f"DRIFT_{int(time.time())}",
                    severity="warning",
                    category="drift",
                    message=f"Model drift detected for {model_id}",
                    details=drift.to_dict(),
                    timestamp=datetime.now(),
                )
                self.alerts.append(alert)
                self.logger.log(
                    "WARNING",
                    alert.message,
                    alert_id=alert.alert_id,
                    drift=drift.to_dict(),
                )

    def get_status(self) -> Dict[str, Any]:
        """Get monitoring status."""
        metrics = self.performance_monitor.get_current_metrics()
        stats = self.performance_monitor.get_statistics()

        # Get model health
        model_health = {}
        for model_id in self.drift_detector.model_scores.keys():
            model_health[model_id] = self.drift_detector.get_model_health(model_id)

        # Get recent alerts
        recent_alerts = list(self.alerts)[-10:]

        status = {
            "status": "running" if self.is_running else "stopped",
            "uptime": (datetime.now() - self.start_time).total_seconds(),
            "current_metrics": metrics.to_dict(),
            "statistics": stats,
            "model_health": model_health,
            "recent_alerts": [a.to_dict() for a in recent_alerts],
            "alert_count": len([a for a in self.alerts if not a.resolved]),
        }

        return status

    def export_metrics(self, output_file: Optional[str] = None) -> Path:
        """
        Export metrics to file.

        Args:
            output_file: Output file path

        Returns:
            Path to exported file
        """
        if output_file:
            output_path = Path(output_file)
        else:
            output_path = self.metrics_dir / f"metrics_{datetime.now():%Y%m%d_%H%M%S}.json"

        metrics = self.get_status()

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        self.logger.log("INFO", f"Metrics exported to {output_path}")

        return output_path

    def shutdown(self) -> None:
        """Shutdown monitoring system."""
        self.is_running = False

        # Export final metrics
        self.export_metrics()

        self.logger.log("INFO", "FraudLens monitoring system shutdown")


if __name__ == "__main__":
    # Example usage
    monitor = FraudLensMonitor(enable_debug=True)

    # Simulate detections
    import random

    for i in range(100):
        monitor.record_detection(
            detector_id="test_detector",
            fraud_score=random.random(),
            latency_ms=random.uniform(10, 200),
            success=random.random() > 0.1,
            fraud_type="phishing" if random.random() > 0.5 else "spam",
            confidence=random.random(),
        )

        time.sleep(0.01)

    # Get status
    status = monitor.get_status()
    print(f"\nMonitoring Status:")
    print(f"  Uptime: {status['uptime']:.1f}s")
    print(f"  Total Requests: {status['statistics']['total_requests']}")
    print(f"  Success Rate: {status['statistics']['success_rate']:.2%}")
    print(f"  Alerts: {status['alert_count']}")

    # Export metrics
    metrics_file = monitor.export_metrics()
    print(f"\nMetrics exported to: {metrics_file}")

    # Shutdown
    monitor.shutdown()
