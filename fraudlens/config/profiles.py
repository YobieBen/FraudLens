"""
Configuration profiles for different environments.

Author: Yobie Benjamin
Date: 2026-02-28
"""

import os
from enum import Enum
from typing import Any

from fraudlens.config.settings import FraudLensSettings


class Profile(str, Enum):
    """Configuration profiles."""
    
    DEVELOPMENT = "dev"
    STAGING = "staging"
    PRODUCTION = "prod"
    TEST = "test"


def get_profile() -> Profile:
    """
    Get current configuration profile from environment.
    
    Checks FRAUDLENS_ENVIRONMENT or ENV environment variables.
    Falls back to development profile.
    
    Returns:
        Current profile
    """
    env = os.getenv("FRAUDLENS_ENVIRONMENT") or os.getenv("ENV") or "dev"
    
    try:
        return Profile(env.lower())
    except ValueError:
        return Profile.DEVELOPMENT


def get_development_settings() -> FraudLensSettings:
    """
    Get settings for development environment.
    
    Returns:
        Development settings with:
        - Debug mode enabled
        - Verbose logging
        - In-memory cache/storage
        - No authentication required
    """
    from fraudlens.config.settings import CacheSettings, StorageSettings
    
    return FraudLensSettings(
        environment="dev",
        debug=True,
        cache=CacheSettings(backend="memory"),
        storage=StorageSettings(backend="memory"),
    )


def get_staging_settings() -> FraudLensSettings:
    """
    Get settings for staging environment.
    
    Returns:
        Staging settings with:
        - Debug mode disabled
        - Info logging
        - Redis cache
        - SQLite storage
        - Optional authentication
    """
    return FraudLensSettings(
        environment="staging",
        debug=False,
        observability__log_level="INFO",
        cache__backend="redis",
        storage__backend="sqlite",
        security__enable_auth=True,
        observability__enable_tracing=True,
        observability__enable_metrics=True,
    )


def get_production_settings() -> FraudLensSettings:
    """
    Get settings for production environment.
    
    Returns:
        Production settings with:
        - Debug mode disabled
        - Warning logging
        - Redis cache with semantic search
        - PostgreSQL storage
        - Authentication required
        - Full observability
    """
    from fraudlens.config.settings import CacheSettings, StorageSettings
    
    return FraudLensSettings(
        environment="prod",
        debug=False,
        cache=CacheSettings(backend="redis", enable_semantic_cache=True),
        storage=StorageSettings(backend="postgresql"),
        enable_vector_search=True,
        enable_llm_orchestration=True,
    )


def get_test_settings() -> FraudLensSettings:
    """
    Get settings for test environment.
    
    Returns:
        Test settings with:
        - Fast in-memory backends
        - Minimal logging
        - No external dependencies
        - Faster timeouts
    """
    return FraudLensSettings(
        environment="dev",
        debug=True,
        observability__log_level="ERROR",
        cache__backend="memory",
        storage__backend="memory",
        security__enable_auth=False,
        default_timeout_seconds=5,
        models__model_timeout_seconds=10,
        max_concurrent_requests=10,
    )


def get_settings_for_profile(profile: Profile | str) -> FraudLensSettings:
    """
    Get settings for specific profile.
    
    Args:
        profile: Configuration profile
    
    Returns:
        Settings instance for the profile
    """
    if isinstance(profile, str):
        profile = Profile(profile)
    
    profile_map = {
        Profile.DEVELOPMENT: get_development_settings,
        Profile.STAGING: get_staging_settings,
        Profile.PRODUCTION: get_production_settings,
        Profile.TEST: get_test_settings,
    }
    
    return profile_map[profile]()


def merge_settings(
    base: FraudLensSettings,
    overrides: dict[str, Any],
) -> FraudLensSettings:
    """
    Merge settings with overrides.
    
    Args:
        base: Base settings
        overrides: Dictionary of setting overrides
    
    Returns:
        New settings instance with overrides applied
    """
    base_dict = base.model_dump()
    
    # Deep merge nested settings
    for key, value in overrides.items():
        if "__" in key:
            # Handle nested settings (e.g., "cache__backend")
            parts = key.split("__")
            current = base_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            base_dict[key] = value
    
    return FraudLensSettings(**base_dict)
