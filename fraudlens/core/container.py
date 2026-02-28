"""
Dependency injection container for FraudLens.

Manages component lifecycle and dependencies using a simple
but powerful DI container pattern.

Author: Yobie Benjamin
Date: 2026-02-28
"""

import inspect
from typing import Any, Callable, TypeVar, get_type_hints

from loguru import logger

T = TypeVar("T")


class Container:
    """
    Dependency injection container.
    
    Features:
    - Singleton and transient lifetimes
    - Factory functions
    - Automatic dependency resolution
    - Lifecycle management
    """
    
    def __init__(self):
        """Initialize container."""
        self._singletons: dict[type, Any] = {}
        self._factories: dict[type, Callable] = {}
        self._transients: dict[type, Callable] = {}
        self._instances: dict[type, Any] = {}
    
    def register_singleton(
        self,
        interface: type[T],
        implementation: type[T] | None = None,
        instance: T | None = None,
    ) -> None:
        """
        Register a singleton (single instance for lifetime of container).
        
        Args:
            interface: Interface type
            implementation: Implementation class (if not provided, uses interface)
            instance: Pre-created instance (if provided, used instead of creating new)
        """
        if instance is not None:
            self._instances[interface] = instance
            logger.debug(f"Registered singleton instance: {interface.__name__}")
        else:
            impl = implementation or interface
            self._singletons[interface] = impl
            logger.debug(f"Registered singleton: {interface.__name__} -> {impl.__name__}")
    
    def register_transient(
        self,
        interface: type[T],
        implementation: type[T] | None = None,
    ) -> None:
        """
        Register a transient (new instance each time).
        
        Args:
            interface: Interface type
            implementation: Implementation class (if not provided, uses interface)
        """
        impl = implementation or interface
        self._transients[interface] = impl
        logger.debug(f"Registered transient: {interface.__name__} -> {impl.__name__}")
    
    def register_factory(
        self,
        interface: type[T],
        factory: Callable[..., T],
    ) -> None:
        """
        Register a factory function.
        
        Args:
            interface: Interface type
            factory: Factory function that creates instances
        """
        self._factories[interface] = factory
        logger.debug(f"Registered factory for: {interface.__name__}")
    
    def resolve(self, interface: type[T]) -> T:
        """
        Resolve an instance of the interface.
        
        Args:
            interface: Interface type to resolve
        
        Returns:
            Instance of the interface
        
        Raises:
            ValueError: If interface not registered
        """
        # Check if already instantiated singleton
        if interface in self._instances:
            return self._instances[interface]
        
        # Check for factory
        if interface in self._factories:
            factory = self._factories[interface]
            instance = self._invoke_with_dependencies(factory)
            return instance
        
        # Check for singleton
        if interface in self._singletons:
            impl = self._singletons[interface]
            instance = self._create_instance(impl)
            self._instances[interface] = instance
            return instance
        
        # Check for transient
        if interface in self._transients:
            impl = self._transients[interface]
            return self._create_instance(impl)
        
        raise ValueError(f"No registration found for: {interface.__name__}")
    
    def _create_instance(self, impl: type[T]) -> T:
        """
        Create instance with automatic dependency injection.
        
        Args:
            impl: Implementation class
        
        Returns:
            Created instance
        """
        return self._invoke_with_dependencies(impl)
    
    def _invoke_with_dependencies(self, func: Callable) -> Any:
        """
        Invoke function/constructor with dependency injection.
        
        Args:
            func: Function or class to invoke
        
        Returns:
            Result of invocation
        """
        # Get signature
        sig = inspect.signature(func)
        
        # Resolve dependencies
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            
            # Get type hint
            if param.annotation != inspect.Parameter.empty:
                param_type = param.annotation
                
                # Try to resolve
                try:
                    kwargs[param_name] = self.resolve(param_type)
                except ValueError:
                    # Use default if available
                    if param.default != inspect.Parameter.empty:
                        kwargs[param_name] = param.default
                    else:
                        logger.warning(
                            f"Could not resolve dependency: {param_name} "
                            f"({param_type.__name__})"
                        )
        
        return func(**kwargs)
    
    def clear(self) -> None:
        """Clear all registrations and instances."""
        self._singletons.clear()
        self._factories.clear()
        self._transients.clear()
        self._instances.clear()
        logger.info("Container cleared")
    
    def get_registrations(self) -> dict[str, list[str]]:
        """
        Get all registrations.
        
        Returns:
            Dictionary of registration types and their interfaces
        """
        return {
            "singletons": [t.__name__ for t in self._singletons.keys()],
            "transients": [t.__name__ for t in self._transients.keys()],
            "factories": [t.__name__ for t in self._factories.keys()],
            "instances": [t.__name__ for t in self._instances.keys()],
        }


# Global container instance
_global_container: Container | None = None


def get_container() -> Container:
    """
    Get the global container instance.
    
    Returns:
        Global Container instance
    """
    global _global_container
    if _global_container is None:
        _global_container = Container()
    return _global_container


def set_container(container: Container) -> None:
    """
    Set the global container instance.
    
    Args:
        container: Container instance to set as global
    """
    global _global_container
    _global_container = container


def configure_default_container() -> Container:
    """
    Configure container with default FraudLens components.
    
    Returns:
        Configured container
    """
    from fraudlens.config import FraudLensSettings, get_settings
    from fraudlens.events import EventBus, get_event_bus
    
    container = Container()
    
    # Register configuration
    container.register_singleton(
        FraudLensSettings,
        instance=get_settings()
    )
    
    # Register event bus
    container.register_singleton(
        EventBus,
        instance=get_event_bus()
    )
    
    logger.info("Default container configured")
    return container
