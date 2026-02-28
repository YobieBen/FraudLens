"""
Tests for Phase 1 dependency injection container.

Author: Yobie Benjamin  
Date: 2026-02-28
"""

import pytest

from fraudlens.core.container import Container, configure_default_container, get_container


# Test classes
class ServiceA:
    """Test service A."""
    
    def __init__(self):
        self.name = "ServiceA"


class ServiceB:
    """Test service B with dependency on ServiceA."""
    
    def __init__(self, service_a: ServiceA):
        self.service_a = service_a
        self.name = "ServiceB"


class ServiceC:
    """Test service with optional dependency."""
    
    def __init__(self, service_a: ServiceA, value: int = 42):
        self.service_a = service_a
        self.value = value


def create_service_a() -> ServiceA:
    """Factory function for ServiceA."""
    service = ServiceA()
    service.name = "FactoryServiceA"
    return service


class TestContainer:
    """Test DI container functionality."""
    
    @pytest.fixture
    def container(self):
        """Create fresh container for each test."""
        return Container()
    
    def test_register_and_resolve_singleton(self, container):
        """Test singleton registration and resolution."""
        container.register_singleton(ServiceA)
        
        instance1 = container.resolve(ServiceA)
        instance2 = container.resolve(ServiceA)
        
        # Should return same instance
        assert instance1 is instance2
        assert instance1.name == "ServiceA"
    
    def test_register_and_resolve_transient(self, container):
        """Test transient registration and resolution."""
        container.register_transient(ServiceA)
        
        instance1 = container.resolve(ServiceA)
        instance2 = container.resolve(ServiceA)
        
        # Should return different instances
        assert instance1 is not instance2
        assert instance1.name == "ServiceA"
    
    def test_register_singleton_with_instance(self, container):
        """Test registering pre-created instance."""
        existing_instance = ServiceA()
        existing_instance.name = "PreCreated"
        
        container.register_singleton(ServiceA, instance=existing_instance)
        
        resolved = container.resolve(ServiceA)
        
        assert resolved is existing_instance
        assert resolved.name == "PreCreated"
    
    def test_register_factory(self, container):
        """Test factory registration."""
        container.register_factory(ServiceA, create_service_a)
        
        instance = container.resolve(ServiceA)
        
        assert instance.name == "FactoryServiceA"
    
    def test_dependency_injection(self, container):
        """Test automatic dependency injection."""
        container.register_singleton(ServiceA)
        container.register_singleton(ServiceB)
        
        service_b = container.resolve(ServiceB)
        
        assert service_b.name == "ServiceB"
        assert service_b.service_a is not None
        assert service_b.service_a.name == "ServiceA"
    
    def test_optional_dependency(self, container):
        """Test service with optional dependency."""
        container.register_singleton(ServiceA)
        container.register_singleton(ServiceC)
        
        service_c = container.resolve(ServiceC)
        
        assert service_c.service_a is not None
        assert service_c.value == 42  # Default value
    
    def test_unregistered_interface_raises_error(self, container):
        """Test that resolving unregistered interface raises error."""
        with pytest.raises(ValueError, match="No registration found"):
            container.resolve(ServiceA)
    
    def test_clear(self, container):
        """Test clearing container."""
        container.register_singleton(ServiceA)
        container.resolve(ServiceA)
        
        container.clear()
        
        with pytest.raises(ValueError):
            container.resolve(ServiceA)
    
    def test_get_registrations(self, container):
        """Test getting registration information."""
        container.register_singleton(ServiceA)
        container.register_transient(ServiceB)
        container.register_factory(ServiceC, lambda: ServiceC(ServiceA()))
        
        registrations = container.get_registrations()
        
        assert "ServiceA" in registrations["singletons"]
        assert "ServiceB" in registrations["transients"]
        assert "ServiceC" in registrations["factories"]


class TestGlobalContainer:
    """Test global container functions."""
    
    def test_get_container_singleton(self):
        """Test that get_container returns same instance."""
        container1 = get_container()
        container2 = get_container()
        
        assert container1 is container2
    
    def test_configure_default_container(self):
        """Test default container configuration."""
        container = configure_default_container()
        
        # Should have settings registered
        from fraudlens.config import FraudLensSettings
        settings = container.resolve(FraudLensSettings)
        
        assert settings is not None
        assert isinstance(settings, FraudLensSettings)
        
        # Should have event bus registered
        from fraudlens.events import EventBus
        bus = container.resolve(EventBus)
        
        assert bus is not None
        assert isinstance(bus, EventBus)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
