"""
Configuration management for FraudLens.

Author: Yobie Benjamin
Date: 2025-08-27 18:50:00 PDT
"""

from typing import Any, Dict, Optional


class Config:
    """Configuration management for FraudLens."""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        """Initialize configuration."""
        self._config = config_dict or self._default_config()
    
    def _default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "processors": {
                "text": {"enabled": True, "batch_size": 32},
                "image": {"enabled": True, "batch_size": 16},
                "video": {"enabled": False, "batch_size": 4},
                "audio": {"enabled": False, "batch_size": 8},
            },
            "resource_limits": {
                "max_memory_gb": 100,
                "max_cpu_percent": 80,
                "enable_gpu": True,
            },
            "cache": {
                "enabled": True,
                "max_size": 1000,
                "ttl_seconds": 3600,
            },
            "plugins": {
                "enabled": True,
                "directory": "plugins",
            },
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by dot-notation key."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict:
        """Get configuration as dictionary."""
        return self._config.copy()