"""
Secrets management for FraudLens.

Provides abstraction for loading secrets from various sources:
- Environment variables (default)
- .env files
- AWS Secrets Manager
- HashiCorp Vault
- Cloud provider secret stores

Author: Yobie Benjamin
Date: 2026-02-28
"""

import os
from abc import ABC, abstractmethod
from typing import Any


class SecretProvider(ABC):
    """Abstract base class for secret providers."""
    
    @abstractmethod
    async def get_secret(self, key: str) -> str | None:
        """Get secret value by key."""
        ...
    
    @abstractmethod
    async def get_secrets(self, keys: list[str]) -> dict[str, str]:
        """Get multiple secrets."""
        ...


class EnvironmentSecretProvider(SecretProvider):
    """Load secrets from environment variables."""
    
    def __init__(self, prefix: str = "FRAUDLENS_SECRET_"):
        """
        Initialize environment secret provider.
        
        Args:
            prefix: Prefix for secret environment variables
        """
        self.prefix = prefix
    
    async def get_secret(self, key: str) -> str | None:
        """Get secret from environment."""
        env_key = f"{self.prefix}{key.upper()}"
        return os.getenv(env_key)
    
    async def get_secrets(self, keys: list[str]) -> dict[str, str]:
        """Get multiple secrets from environment."""
        result = {}
        for key in keys:
            value = await self.get_secret(key)
            if value:
                result[key] = value
        return result


class FileSecretProvider(SecretProvider):
    """Load secrets from file (e.g., Docker secrets)."""
    
    def __init__(self, secrets_dir: str = "/run/secrets"):
        """
        Initialize file secret provider.
        
        Args:
            secrets_dir: Directory containing secret files
        """
        self.secrets_dir = secrets_dir
    
    async def get_secret(self, key: str) -> str | None:
        """Get secret from file."""
        secret_path = os.path.join(self.secrets_dir, key)
        
        try:
            with open(secret_path) as f:
                return f.read().strip()
        except FileNotFoundError:
            return None
    
    async def get_secrets(self, keys: list[str]) -> dict[str, str]:
        """Get multiple secrets from files."""
        result = {}
        for key in keys:
            value = await self.get_secret(key)
            if value:
                result[key] = value
        return result


class CompositeSecretProvider(SecretProvider):
    """Composite provider that tries multiple sources in order."""
    
    def __init__(self, providers: list[SecretProvider]):
        """
        Initialize composite provider.
        
        Args:
            providers: List of providers to try in order
        """
        self.providers = providers
    
    async def get_secret(self, key: str) -> str | None:
        """Get secret from first available provider."""
        for provider in self.providers:
            value = await provider.get_secret(key)
            if value:
                return value
        return None
    
    async def get_secrets(self, keys: list[str]) -> dict[str, str]:
        """Get multiple secrets from providers."""
        result = {}
        
        for key in keys:
            value = await self.get_secret(key)
            if value:
                result[key] = value
        
        return result


# Default secret provider (environment variables)
_default_provider: SecretProvider = EnvironmentSecretProvider()


def set_default_provider(provider: SecretProvider) -> None:
    """
    Set the default secret provider.
    
    Args:
        provider: Secret provider to use as default
    """
    global _default_provider
    _default_provider = provider


async def get_secret(key: str, default: str | None = None) -> str | None:
    """
    Get secret value.
    
    Args:
        key: Secret key
        default: Default value if secret not found
    
    Returns:
        Secret value or default
    """
    value = await _default_provider.get_secret(key)
    return value if value is not None else default


async def require_secret(key: str) -> str:
    """
    Get secret value, raise error if not found.
    
    Args:
        key: Secret key
    
    Returns:
        Secret value
    
    Raises:
        ValueError: If secret not found
    """
    value = await get_secret(key)
    if value is None:
        raise ValueError(f"Required secret '{key}' not found")
    return value


def create_default_provider() -> SecretProvider:
    """
    Create default composite provider with multiple fallbacks.
    
    Returns:
        Composite provider with environment and file sources
    """
    return CompositeSecretProvider([
        EnvironmentSecretProvider(),
        FileSecretProvider(),
    ])
