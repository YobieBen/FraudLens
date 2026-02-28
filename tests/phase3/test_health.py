"""Tests for health check system."""

import pytest

from fraudlens.observability.health import (
    HealthCheck,
    HealthCheckManager,
    HealthCheckResult,
    HealthStatus,
    get_health_manager,
    reset_health_manager,
)


class MockHealthCheck(HealthCheck):
    """Mock health check for testing."""
    
    def __init__(self, name: str, status: HealthStatus = HealthStatus.HEALTHY):
        super().__init__(name, critical=True)
        self._status = status
    
    async def _check_health(self) -> HealthCheckResult:
        """Mock health check."""
        return HealthCheckResult(
            component=self.name,
            status=self._status,
            message=f"{self.name} is {self._status.value}"
        )


class TestHealthCheckResult:
    """Test HealthCheckResult."""
    
    def test_result_creation(self):
        """Test creating health check result."""
        result = HealthCheckResult(
            component="test",
            status=HealthStatus.HEALTHY,
            message="All good"
        )
        
        assert result.component == "test"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All good"
    
    def test_result_to_dict(self):
        """Test converting result to dict."""
        result = HealthCheckResult(
            component="test",
            status=HealthStatus.HEALTHY,
            message="All good",
            latency_ms=10.5
        )
        
        result_dict = result.to_dict()
        assert result_dict["component"] == "test"
        assert result_dict["status"] == "healthy"
        assert result_dict["latency_ms"] == 10.5


class TestHealthCheck:
    """Test HealthCheck base class."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        check = MockHealthCheck("test", HealthStatus.HEALTHY)
        result = await check.check()
        
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms is not None
        assert result.latency_ms > 0
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        """Test unhealthy check."""
        check = MockHealthCheck("test", HealthStatus.UNHEALTHY)
        result = await check.check()
        
        assert result.status == HealthStatus.UNHEALTHY
    
    @pytest.mark.asyncio
    async def test_health_check_timeout(self):
        """Test health check timeout."""
        class SlowCheck(HealthCheck):
            async def _check_health(self):
                import asyncio
                await asyncio.sleep(10)  # Sleep longer than timeout
                return HealthCheckResult(
                    component=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Never reached"
                )
        
        check = SlowCheck("slow", timeout_seconds=0.1)
        result = await check.check()
        
        assert result.status == HealthStatus.UNHEALTHY
        assert "timed out" in result.message.lower()


class TestHealthCheckManager:
    """Test HealthCheckManager."""
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = HealthCheckManager()
        
        assert len(manager.checks) == 0
        assert manager.start_time is not None
    
    def test_register_check(self):
        """Test registering a health check."""
        manager = HealthCheckManager()
        check = MockHealthCheck("test")
        
        manager.register(check)
        
        assert len(manager.checks) == 1
        assert manager.checks[0] == check
    
    @pytest.mark.asyncio
    async def test_check_all(self):
        """Test running all health checks."""
        manager = HealthCheckManager()
        manager.register(MockHealthCheck("check1", HealthStatus.HEALTHY))
        manager.register(MockHealthCheck("check2", HealthStatus.HEALTHY))
        
        results = await manager.check_all()
        
        assert len(results) == 2
        assert "check1" in results
        assert "check2" in results
        assert all(r.status == HealthStatus.HEALTHY for r in results.values())
    
    @pytest.mark.asyncio
    async def test_readiness_all_healthy(self):
        """Test readiness when all checks healthy."""
        manager = HealthCheckManager()
        manager.register(MockHealthCheck("check1", HealthStatus.HEALTHY))
        manager.register(MockHealthCheck("check2", HealthStatus.HEALTHY))
        
        is_ready = await manager.readiness()
        
        assert is_ready is True
    
    @pytest.mark.asyncio
    async def test_readiness_one_unhealthy(self):
        """Test readiness when one check unhealthy."""
        manager = HealthCheckManager()
        manager.register(MockHealthCheck("check1", HealthStatus.HEALTHY))
        manager.register(MockHealthCheck("check2", HealthStatus.UNHEALTHY))
        
        is_ready = await manager.readiness()
        
        assert is_ready is False
    
    @pytest.mark.asyncio
    async def test_readiness_non_critical_unhealthy(self):
        """Test readiness with non-critical check unhealthy."""
        manager = HealthCheckManager()
        manager.register(MockHealthCheck("check1", HealthStatus.HEALTHY))
        
        # Add non-critical check
        check2 = MockHealthCheck("check2", HealthStatus.UNHEALTHY)
        check2.critical = False
        manager.register(check2)
        
        is_ready = await manager.readiness()
        
        # Should still be ready because unhealthy check is not critical
        assert is_ready is True
    
    @pytest.mark.asyncio
    async def test_liveness(self):
        """Test liveness check."""
        manager = HealthCheckManager()
        
        is_alive = await manager.liveness()
        
        assert is_alive is True
    
    @pytest.mark.asyncio
    async def test_health_status(self):
        """Test getting health status."""
        manager = HealthCheckManager()
        manager.register(MockHealthCheck("check1", HealthStatus.HEALTHY))
        manager.register(MockHealthCheck("check2", HealthStatus.DEGRADED))
        
        status = await manager.health_status()
        
        assert "status" in status
        assert status["status"] == "degraded"  # One degraded = overall degraded
        assert "uptime_seconds" in status
        assert "checks" in status
        assert len(status["checks"]) == 2


class TestGlobalHealthManager:
    """Test global health manager."""
    
    def test_get_health_manager(self):
        """Test getting global health manager."""
        reset_health_manager()
        
        manager1 = get_health_manager()
        manager2 = get_health_manager()
        
        assert manager1 is manager2
    
    def test_reset_health_manager(self):
        """Test resetting health manager."""
        manager1 = get_health_manager()
        reset_health_manager()
        manager2 = get_health_manager()
        
        assert manager1 is not manager2
