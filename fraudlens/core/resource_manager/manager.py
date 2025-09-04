"""
Resource manager for memory and compute resource monitoring.

Author: Yobie Benjamin
Date: 2025-08-26 18:34:00 PDT
"""

import asyncio
import gc
import os
import platform
import resource
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import psutil

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time."""

    timestamp: datetime
    memory_used_gb: float
    memory_available_gb: float
    memory_percent: float
    cpu_percent: float
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    active_models: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "memory_used_gb": self.memory_used_gb,
            "memory_available_gb": self.memory_available_gb,
            "memory_percent": self.memory_percent,
            "cpu_percent": self.cpu_percent,
            "gpu_memory_used_gb": self.gpu_memory_used_gb,
            "gpu_memory_total_gb": self.gpu_memory_total_gb,
            "active_models": self.active_models,
        }


class ResourceManager:
    """
    Manage system resources for fraud detection pipeline.

    Monitors memory usage and ensures system stability by:
    - Tracking memory consumption
    - Enforcing memory limits (default 100GB for M4 Max)
    - Managing model lifecycle (loading/unloading)
    - Optimizing resource allocation
    """

    def __init__(
        self,
        max_memory_gb: float = 100.0,
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.95,
        monitoring_interval: float = 5.0,
        enable_monitoring: bool = True,
    ):
        """
        Initialize resource manager.

        Args:
            max_memory_gb: Maximum memory usage in GB (default 100GB)
            warning_threshold: Warn when memory usage exceeds this fraction
            critical_threshold: Take action when memory usage exceeds this
            monitoring_interval: Seconds between resource checks
            enable_monitoring: Whether to enable background monitoring
        """
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.monitoring_interval = monitoring_interval
        self.enable_monitoring = enable_monitoring

        self._active_models: Dict[str, Any] = {}
        self._model_memory: Dict[str, int] = {}
        self._history: List[ResourceSnapshot] = []
        self._monitoring_task: Optional[asyncio.Task] = None
        self._callbacks: List[Callable] = []
        self._is_apple_silicon = self._detect_apple_silicon()

    def _detect_apple_silicon(self) -> bool:
        """Detect if running on Apple Silicon."""
        if platform.system() != "Darwin":
            return False

        try:
            # Check for ARM64 architecture
            return platform.machine() in ["arm64", "aarch64"]
        except:
            return False

    async def start_monitoring(self) -> None:
        """Start background resource monitoring."""
        if self.enable_monitoring and not self._monitoring_task:
            self._monitoring_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitoring(self) -> None:
        """Stop background resource monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                snapshot = self.get_snapshot()
                self._history.append(snapshot)

                # Keep only last 1000 snapshots
                if len(self._history) > 1000:
                    self._history = self._history[-1000:]

                # Check thresholds
                memory_fraction = snapshot.memory_used_gb / self.max_memory_gb

                if memory_fraction > self.critical_threshold:
                    await self._handle_critical_memory()
                elif memory_fraction > self.warning_threshold:
                    self._handle_warning_memory()

                # Call registered callbacks
                for callback in self._callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(snapshot)
                        else:
                            callback(snapshot)
                    except Exception as e:
                        warnings.warn(f"Callback error: {e}")

                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                warnings.warn(f"Monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)

    def get_snapshot(self) -> ResourceSnapshot:
        """Get current resource snapshot."""
        process = psutil.Process()
        memory_info = process.memory_info()

        # System memory
        virtual_memory = psutil.virtual_memory()
        memory_used_gb = memory_info.rss / (1024**3)
        memory_available_gb = virtual_memory.available / (1024**3)
        memory_percent = (memory_info.rss / self.max_memory_bytes) * 100

        # CPU
        cpu_percent = process.cpu_percent()

        # GPU memory (if MLX available and on Apple Silicon)
        gpu_memory_used_gb = None
        gpu_memory_total_gb = None

        if MLX_AVAILABLE and self._is_apple_silicon:
            try:
                # MLX doesn't directly expose memory, estimate from active models
                gpu_memory_used_gb = sum(self._model_memory.values()) / (1024**3)
            except:
                pass

        return ResourceSnapshot(
            timestamp=datetime.now(),
            memory_used_gb=memory_used_gb,
            memory_available_gb=memory_available_gb,
            memory_percent=memory_percent,
            cpu_percent=cpu_percent,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_total_gb=gpu_memory_total_gb,
            active_models=len(self._active_models),
        )

    def register_model(self, model_id: str, model: Any, estimated_memory_mb: int = 0) -> None:
        """
        Register a loaded model.

        Args:
            model_id: Unique identifier for the model
            model: The model object
            estimated_memory_mb: Estimated memory usage in MB
        """
        self._active_models[model_id] = model

        if estimated_memory_mb == 0:
            # Try to estimate memory usage
            estimated_memory_mb = self._estimate_model_memory(model)

        self._model_memory[model_id] = estimated_memory_mb * 1024 * 1024

    def unregister_model(self, model_id: str) -> None:
        """
        Unregister and cleanup a model.

        Args:
            model_id: Model identifier to remove
        """
        if model_id in self._active_models:
            model = self._active_models.pop(model_id)
            self._model_memory.pop(model_id, None)

            # Attempt to free memory
            del model
            gc.collect()

            if MLX_AVAILABLE and self._is_apple_silicon:
                try:
                    mx.clear_cache()
                except:
                    pass

    def _estimate_model_memory(self, model: Any) -> int:
        """
        Estimate model memory usage in MB.

        Args:
            model: Model object

        Returns:
            Estimated memory in MB
        """
        try:
            # Simple estimation based on parameter count
            param_count = 0

            if hasattr(model, "parameters"):
                for param in model.parameters():
                    if hasattr(param, "numel"):
                        param_count += param.numel()
                    elif hasattr(param, "size"):
                        import numpy as np

                        param_count += np.prod(param.size)

            # Assume 4 bytes per parameter (float32)
            return (param_count * 4) // (1024 * 1024)
        except:
            # Default estimation
            return 500  # 500MB default

    async def request_memory(self, required_mb: int) -> bool:
        """
        Request memory allocation.

        Args:
            required_mb: Required memory in MB

        Returns:
            True if memory is available, False otherwise
        """
        snapshot = self.get_snapshot()
        required_gb = required_mb / 1024

        if snapshot.memory_used_gb + required_gb > self.max_memory_gb * self.critical_threshold:
            # Try to free memory
            await self._handle_critical_memory()

            # Check again
            snapshot = self.get_snapshot()
            if snapshot.memory_used_gb + required_gb > self.max_memory_gb:
                return False

        return True

    async def _handle_critical_memory(self) -> None:
        """Handle critical memory situation."""
        warnings.warn(
            f"Critical memory usage: {self.get_snapshot().memory_used_gb:.1f}GB / "
            f"{self.max_memory_gb:.1f}GB"
        )

        # Try to free memory
        gc.collect()

        # Unload least recently used models if needed
        if len(self._active_models) > 1:
            # Remove the oldest model (simple LRU)
            oldest_model_id = next(iter(self._active_models))
            self.unregister_model(oldest_model_id)
            warnings.warn(f"Unloaded model {oldest_model_id} to free memory")

    def _handle_warning_memory(self) -> None:
        """Handle warning memory threshold."""
        warnings.warn(
            f"High memory usage: {self.get_snapshot().memory_used_gb:.1f}GB / "
            f"{self.max_memory_gb:.1f}GB"
        )

    def register_callback(self, callback: Callable[[ResourceSnapshot], None]) -> None:
        """
        Register a callback for resource events.

        Args:
            callback: Function to call with resource snapshots
        """
        self._callbacks.append(callback)

    def get_statistics(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        # Always return full structure, even with no history
        if not self._history:
            current_snapshot = self.get_snapshot()
            memory_mb = current_snapshot.memory_used_gb * 1024
            return {
                # Flat keys for compatibility
                "memory_usage_mb": memory_mb,
                "memory_usage_percent": current_snapshot.memory_percent,
                "memory_used_gb": current_snapshot.memory_used_gb,
                "memory_available_gb": current_snapshot.memory_available_gb,
                "cpu_percent": current_snapshot.cpu_percent,
                "peak_memory_mb": memory_mb,
                "gpu_available": current_snapshot.gpu_memory_used_gb is not None,
                "gpu_memory_used_mb": (
                    (current_snapshot.gpu_memory_used_gb * 1024)
                    if current_snapshot.gpu_memory_used_gb
                    else 0
                ),
                # Detailed structure
                "current": current_snapshot.to_dict(),
                "statistics": {
                    "memory": {
                        "mean_gb": current_snapshot.memory_used_gb,
                        "max_gb": current_snapshot.memory_used_gb,
                        "min_gb": current_snapshot.memory_used_gb,
                    },
                    "cpu": {
                        "mean_percent": current_snapshot.cpu_percent,
                        "max_percent": current_snapshot.cpu_percent,
                        "min_percent": current_snapshot.cpu_percent,
                    },
                    "models": {
                        "active": len(self._active_models),
                        "total_memory_mb": sum(self._model_memory.values()) / (1024 * 1024),
                    },
                },
                "config": {
                    "max_memory_gb": self.max_memory_gb,
                    "warning_threshold": self.warning_threshold,
                    "critical_threshold": self.critical_threshold,
                    "is_apple_silicon": self._is_apple_silicon,
                },
            }

        memory_usage = [s.memory_used_gb for s in self._history]
        cpu_usage = [s.cpu_percent for s in self._history]

        current_snapshot = self.get_snapshot()
        memory_mb = current_snapshot.memory_used_gb * 1024
        peak_mb = max(memory_usage) * 1024 if memory_usage else memory_mb

        return {
            # Flat keys for compatibility
            "memory_usage_mb": memory_mb,
            "memory_usage_percent": current_snapshot.memory_percent,
            "memory_used_gb": current_snapshot.memory_used_gb,
            "memory_available_gb": current_snapshot.memory_available_gb,
            "cpu_percent": current_snapshot.cpu_percent,
            "peak_memory_mb": peak_mb,
            "gpu_available": current_snapshot.gpu_memory_used_gb is not None,
            "gpu_memory_used_mb": (
                (current_snapshot.gpu_memory_used_gb * 1024)
                if current_snapshot.gpu_memory_used_gb
                else 0
            ),
            # Detailed structure
            "current": current_snapshot.to_dict(),
            "statistics": {
                "memory": {
                    "mean_gb": sum(memory_usage) / len(memory_usage),
                    "max_gb": max(memory_usage),
                    "min_gb": min(memory_usage),
                },
                "cpu": {
                    "mean_percent": sum(cpu_usage) / len(cpu_usage),
                    "max_percent": max(cpu_usage),
                    "min_percent": min(cpu_usage),
                },
                "models": {
                    "active": len(self._active_models),
                    "total_memory_mb": sum(self._model_memory.values()) / (1024 * 1024),
                },
            },
            "config": {
                "max_memory_gb": self.max_memory_gb,
                "warning_threshold": self.warning_threshold,
                "critical_threshold": self.critical_threshold,
                "is_apple_silicon": self._is_apple_silicon,
            },
        }

    def optimize_for_apple_silicon(self) -> None:
        """Apply Apple Silicon specific optimizations."""
        if not self._is_apple_silicon:
            return

        # Set environment variables for Metal Performance Shaders
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        if MLX_AVAILABLE:
            # Configure MLX for optimal performance
            try:
                # Set memory growth
                mx.set_memory_limit(int(self.max_memory_gb * 0.9 * 1024))  # Leave 10% buffer
            except:
                pass

    def __repr__(self) -> str:
        """String representation."""
        snapshot = self.get_snapshot()
        return (
            f"ResourceManager("
            f"memory={snapshot.memory_used_gb:.1f}/{self.max_memory_gb:.1f}GB, "
            f"models={len(self._active_models)}, "
            f"apple_silicon={self._is_apple_silicon})"
        )
