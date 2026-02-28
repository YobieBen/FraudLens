"""
Pydantic-based configuration settings for FraudLens.

Author: Yobie Benjamin
Date: 2026-02-28
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProcessorSettings(BaseSettings):
    """Settings for fraud detection processors."""
    
    enabled: bool = True
    batch_size: int = Field(default=32, ge=1, le=1000)
    timeout_seconds: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=0)


class ModelSettings(BaseSettings):
    """Settings for AI models."""
    
    cache_dir: Path = Field(default_factory=lambda: Path.home() / ".fraudlens" / "models")
    enable_model_fallback: bool = True
    model_timeout_seconds: int = Field(default=60, ge=1)
    use_metal: bool = True  # Apple Silicon GPU
    device: Literal["cpu", "mps", "cuda"] = "mps"
    
    @field_validator("cache_dir", mode="before")
    @classmethod
    def create_cache_dir(cls, v: Path | str) -> Path:
        """Ensure cache directory exists."""
        path = Path(v) if isinstance(v, str) else v
        path.mkdir(parents=True, exist_ok=True)
        return path


class CacheSettings(BaseSettings):
    """Settings for caching."""
    
    backend: Literal["memory", "redis", "memcached"] = "memory"
    max_size: int = Field(default=1000, ge=1)
    ttl_seconds: int = Field(default=3600, ge=1)
    redis_url: str | None = None
    enable_semantic_cache: bool = False


class StorageSettings(BaseSettings):
    """Settings for data storage."""
    
    backend: Literal["memory", "sqlite", "postgresql"] = "sqlite"
    database_url: str | None = None
    connection_pool_size: int = Field(default=10, ge=1)
    enable_compression: bool = True


class ObservabilitySettings(BaseSettings):
    """Settings for monitoring and observability."""
    
    enable_tracing: bool = True
    enable_metrics: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    otel_endpoint: str | None = None
    metrics_port: int = Field(default=9090, ge=1024, le=65535)


class SecuritySettings(BaseSettings):
    """Settings for security and authentication."""
    
    api_key: str | None = None
    enable_auth: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    rate_limit_requests_per_minute: int = Field(default=100, ge=1)


class FraudLensSettings(BaseSettings):
    """
    Main configuration settings for FraudLens.
    
    Configuration can be provided via:
    - Environment variables with FRAUDLENS_ prefix
    - .env file in current directory or project root
    - Direct instantiation with kwargs
    
    Example:
        ```python
        # From environment
        settings = FraudLensSettings()
        
        # From .env file
        settings = FraudLensSettings(_env_file=".env.production")
        
        # Direct configuration
        settings = FraudLensSettings(
            environment="production",
            log_level="WARNING"
        )
        ```
    """
    
    model_config = SettingsConfigDict(
        env_prefix="FRAUDLENS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Core settings
    environment: Literal["dev", "staging", "prod"] = "dev"
    debug: bool = Field(default=False)
    
    # Processing settings
    max_concurrent_requests: int = Field(default=100, ge=1)
    default_timeout_seconds: int = Field(default=30, ge=1)
    enable_streaming: bool = True
    enable_batch_processing: bool = True
    
    # Component settings
    text_processor: ProcessorSettings = Field(default_factory=ProcessorSettings)
    image_processor: ProcessorSettings = Field(default_factory=ProcessorSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    
    # Feature flags
    enable_vector_search: bool = False
    enable_llm_orchestration: bool = False
    enable_multi_agent: bool = False
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        if v not in ["dev", "staging", "prod"]:
            raise ValueError(f"Invalid environment: {v}")
        return v
    
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "prod"
    
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "dev"
    
    def to_dict(self) -> dict:
        """Convert settings to dictionary."""
        return self.model_dump()


@lru_cache
def get_settings(env_file: str | None = None) -> FraudLensSettings:
    """
    Get cached settings instance.
    
    This function is cached to ensure single instance across the application.
    To reload settings, clear the cache with `get_settings.cache_clear()`.
    
    Args:
        env_file: Optional path to .env file
    
    Returns:
        Settings instance
    """
    if env_file:
        return FraudLensSettings(_env_file=env_file)
    
    # Auto-detect .env file
    for env_path in [".env", ".env.local", Path.cwd() / ".env"]:
        if Path(env_path).exists():
            return FraudLensSettings(_env_file=str(env_path))
    
    return FraudLensSettings()


def load_settings_from_yaml(yaml_path: Path) -> FraudLensSettings:
    """
    Load settings from YAML file.
    
    Args:
        yaml_path: Path to YAML configuration file
    
    Returns:
        Settings instance
    """
    import yaml
    
    with open(yaml_path) as f:
        config_dict = yaml.safe_load(f)
    
    return FraudLensSettings(**config_dict)


def load_settings_from_toml(toml_path: Path) -> FraudLensSettings:
    """
    Load settings from TOML file.
    
    Args:
        toml_path: Path to TOML configuration file
    
    Returns:
        Settings instance
    """
    import tomllib
    
    with open(toml_path, "rb") as f:
        config_dict = tomllib.load(f)
    
    return FraudLensSettings(**config_dict)
