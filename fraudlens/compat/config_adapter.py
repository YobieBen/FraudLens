"""
Config adapter for backward compatibility.

Wraps new FraudLensSettings to provide old Config interface.

Author: Yobie Benjamin
Date: 2026-02-28
"""

from typing import Any

from fraudlens.config import FraudLensSettings


class ConfigAdapter:
    """
    Adapter that provides old Config interface using new FraudLensSettings.
    
    This allows existing code using the old dict-based Config to work
    with the new Pydantic-based settings.
    """
    
    def __init__(self, settings: FraudLensSettings | None = None):
        """
        Initialize config adapter.
        
        Args:
            settings: Settings instance (if None, creates default)
        """
        from fraudlens.config import get_settings
        
        self._settings = settings or get_settings()
        self._config = self._settings_to_dict()
    
    def _settings_to_dict(self) -> dict:
        """Convert settings to old dict format."""
        return {
            "processors": {
                "text": {
                    "enabled": self._settings.text_processor.enabled,
                    "batch_size": self._settings.text_processor.batch_size,
                },
                "image": {
                    "enabled": self._settings.image_processor.enabled,
                    "batch_size": self._settings.image_processor.batch_size,
                },
                "vision": {
                    "enabled": self._settings.image_processor.enabled,
                    "batch_size": self._settings.image_processor.batch_size,
                    "use_metal": self._settings.models.use_metal,
                },
            },
            "resource_limits": {
                "max_memory_gb": 100,  # Default
                "max_cpu_percent": 80,  # Default
                "enable_gpu": self._settings.models.device != "cpu",
            },
            "cache": {
                "enabled": True,
                "max_size": self._settings.cache.max_size,
                "ttl_seconds": self._settings.cache.ttl_seconds,
            },
            "plugins": {
                "enabled": True,
                "directory": "plugins",
            },
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.
        
        Args:
            key: Configuration key (e.g., "processors.text.enabled")
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot-notation key.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        keys = key.split(".")
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> dict:
        """
        Get configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self._config.copy()


def create_compatible_config(settings: FraudLensSettings | None = None):
    """
    Create config object compatible with old code.
    
    Args:
        settings: Settings instance
    
    Returns:
        ConfigAdapter instance that works like old Config
    """
    return ConfigAdapter(settings)
