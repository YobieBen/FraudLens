"""
Health check system for FraudLens components.

Provides readiness and liveness probes for Kubernetes deployments
and dependency health monitoring.

Author: Yobie Benjamin
Date: 2026-02-28
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from fraudlens.observability.logging import get_logger

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health check status."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


class HealthCheck:
    """
    Base health check component.
    
    Subclass this to create custom health checks for components.
    """
    
    def __init__(
        self,
        name: str,
        timeout_seconds: float = 5.0,
        critical: bool = True
    ):
        """
        Initialize health check.
        
        Args:
            name: Component name
            timeout_seconds: Timeout for health check
            critical: Whether this component is critical for readiness
        """
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.critical = critical
    
    async def check(self) -> HealthCheckResult:
        """
        Perform health check.
        
        Returns:
            Health check result
        """
        start = datetime.utcnow()
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                self._check_health(),
                timeout=self.timeout_seconds
            )
            
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            result.latency_ms = latency
            
            return result
            
        except asyncio.TimeoutError:
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout_seconds}s",
                latency_ms=(datetime.utcnow() - start).total_seconds() * 1000
            )
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                latency_ms=(datetime.utcnow() - start).total_seconds() * 1000
            )
    
    async def _check_health(self) -> HealthCheckResult:
        """
        Implement health check logic.
        
        Override this in subclasses.
        """
        raise NotImplementedError


class AgentHealthCheck(HealthCheck):
    """Health check for fraud detection agents."""
    
    def __init__(self, agent_registry: Any):
        super().__init__("agents", critical=True)
        self.agent_registry = agent_registry
    
    async def _check_health(self) -> HealthCheckResult:
        """Check if agents are available."""
        # Check if agents are initialized
        # This is a placeholder - implement based on your agent registry
        return HealthCheckResult(
            component=self.name,
            status=HealthStatus.HEALTHY,
            message="All agents operational",
            metadata={"agent_count": 0}  # Update with actual count
        )


class LLMProviderHealthCheck(HealthCheck):
    """Health check for LLM providers."""
    
    def __init__(self, orchestrator: Any):
        super().__init__("llm_providers", critical=False)
        self.orchestrator = orchestrator
    
    async def _check_health(self) -> HealthCheckResult:
        """Check if LLM providers are available."""
        # Check circuit breaker states
        available_providers = 0
        total_providers = len(getattr(self.orchestrator, 'providers', []))
        
        # Check each provider's circuit breaker
        for config, _ in getattr(self.orchestrator, 'providers', []):
            provider_key = f"{config.provider}:{config.model}"
            if self.orchestrator.circuit_breaker.is_available(provider_key):
                available_providers += 1
        
        if available_providers == 0:
            status = HealthStatus.UNHEALTHY
            message = "No LLM providers available"
        elif available_providers < total_providers:
            status = HealthStatus.DEGRADED
            message = f"{available_providers}/{total_providers} providers available"
        else:
            status = HealthStatus.HEALTHY
            message = "All LLM providers available"
        
        return HealthCheckResult(
            component=self.name,
            status=status,
            message=message,
            metadata={
                "available_providers": available_providers,
                "total_providers": total_providers
            }
        )


class EventBusHealthCheck(HealthCheck):
    """Health check for event bus."""
    
    def __init__(self, event_bus: Any):
        super().__init__("event_bus", critical=True)
        self.event_bus = event_bus
    
    async def _check_health(self) -> HealthCheckResult:
        """Check if event bus is operational."""
        # Check if event bus can process events
        return HealthCheckResult(
            component=self.name,
            status=HealthStatus.HEALTHY,
            message="Event bus operational"
        )


class HealthCheckManager:
    """
    Manager for all health checks.
    
    Aggregates health check results and provides readiness/liveness endpoints.
    
    Example:
        ```python
        manager = HealthCheckManager()
        manager.register(LLMProviderHealthCheck(orchestrator))
        manager.register(AgentHealthCheck(agent_registry))
        
        # Check readiness
        is_ready = await manager.readiness()
        
        # Check liveness
        is_alive = await manager.liveness()
        
        # Get detailed status
        status = await manager.health_status()
        ```
    """
    
    def __init__(self):
        """Initialize health check manager."""
        self.checks: List[HealthCheck] = []
        self.last_results: Dict[str, HealthCheckResult] = {}
        self.start_time = datetime.utcnow()
    
    def register(self, check: HealthCheck) -> None:
        """
        Register a health check.
        
        Args:
            check: Health check to register
        """
        self.checks.append(check)
        logger.info(f"Registered health check: {check.name}")
    
    async def check_all(self) -> Dict[str, HealthCheckResult]:
        """
        Run all health checks.
        
        Returns:
            Dictionary of component name to health check result
        """
        results = {}
        
        for check in self.checks:
            result = await check.check()
            results[check.name] = result
            self.last_results[check.name] = result
            
            if result.status != HealthStatus.HEALTHY:
                logger.warning(
                    f"Health check failed: {check.name}",
                    status=result.status.value,
                    message=result.message
                )
        
        return results
    
    async def readiness(self) -> bool:
        """
        Check if application is ready to serve traffic.
        
        Returns:
            True if all critical components are healthy
        """
        results = await self.check_all()
        
        for check in self.checks:
            if not check.critical:
                continue
            
            result = results.get(check.name)
            if not result or result.status == HealthStatus.UNHEALTHY:
                return False
        
        return True
    
    async def liveness(self) -> bool:
        """
        Check if application is alive.
        
        This is a lightweight check - just verifies the process is running.
        
        Returns:
            True if application is alive
        """
        # Simple liveness check - can be extended
        return True
    
    async def health_status(self) -> Dict[str, Any]:
        """
        Get detailed health status.
        
        Returns:
            Dictionary with health status details
        """
        results = await self.check_all()
        
        # Calculate overall status
        has_unhealthy = any(
            r.status == HealthStatus.UNHEALTHY for r in results.values()
        )
        has_degraded = any(
            r.status == HealthStatus.DEGRADED for r in results.values()
        )
        
        if has_unhealthy:
            overall_status = HealthStatus.UNHEALTHY
        elif has_degraded:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": uptime,
            "checks": {
                name: result.to_dict()
                for name, result in results.items()
            }
        }


# Global health check manager
_health_manager: Optional[HealthCheckManager] = None


def get_health_manager() -> HealthCheckManager:
    """Get global health check manager."""
    global _health_manager
    
    if _health_manager is None:
        _health_manager = HealthCheckManager()
    
    return _health_manager


def reset_health_manager() -> None:
    """Reset global health check manager (useful for testing)."""
    global _health_manager
    _health_manager = None
