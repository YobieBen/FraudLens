"""
Modern configuration management for FraudLens.

Uses Pydantic BaseSettings for type-safe, validated configuration
with support for environment variables, .env files, and YAML/TOML.

Author: Yobie Benjamin
Date: 2026-02-28
"""

from fraudlens.config.profiles import (
    Profile,
    get_profile,
    get_settings_for_profile,
)
from fraudlens.config.settings import FraudLensSettings, get_settings

__all__ = [
    "FraudLensSettings",
    "get_settings",
    "Profile",
    "get_profile",
    "get_settings_for_profile",
]
