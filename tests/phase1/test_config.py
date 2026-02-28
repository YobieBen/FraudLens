"""
Tests for Phase 1 configuration system.

Author: Yobie Benjamin
Date: 2026-02-28
"""

import os
import tempfile
from pathlib import Path

import pytest

from fraudlens.config import (
    FraudLensSettings,
    Profile,
    get_profile,
    get_settings,
    get_settings_for_profile,
)
from fraudlens.config.profiles import (
    get_development_settings,
    get_production_settings,
    merge_settings,
)


class TestFraudLensSettings:
    """Test FraudLensSettings configuration."""
    
    def test_default_settings(self):
        """Test default settings creation."""
        settings = FraudLensSettings()
        
        assert settings.environment == "dev"
        assert settings.debug == False
        assert settings.max_concurrent_requests == 100
        assert settings.enable_streaming == True
    
    def test_environment_validation(self):
        """Test environment validation."""
        # Valid environments
        for env in ["dev", "staging", "prod"]:
            settings = FraudLensSettings(environment=env)
            assert settings.environment == env
        
        # Invalid environment
        with pytest.raises(ValueError):
            FraudLensSettings(environment="invalid")
    
    def test_nested_settings(self):
        """Test nested settings access."""
        settings = FraudLensSettings()
        
        assert settings.cache.backend == "memory"
        assert settings.cache.max_size == 1000
        assert settings.storage.backend == "sqlite"
        assert settings.models.use_metal == True
    
    def test_environment_variable_override(self, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv("FRAUDLENS_ENVIRONMENT", "prod")
        monkeypatch.setenv("FRAUDLENS_DEBUG", "true")
        monkeypatch.setenv("FRAUDLENS_MAX_CONCURRENT_REQUESTS", "200")
        
        settings = FraudLensSettings()
        
        assert settings.environment == "prod"
        assert settings.debug == True
        assert settings.max_concurrent_requests == 200
    
    def test_is_production(self):
        """Test is_production helper."""
        dev_settings = FraudLensSettings(environment="dev")
        prod_settings = FraudLensSettings(environment="prod")
        
        assert dev_settings.is_production() == False
        assert prod_settings.is_production() == True
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        settings = FraudLensSettings()
        config_dict = settings.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "environment" in config_dict
        assert "cache" in config_dict
        assert config_dict["environment"] == "dev"


class TestProfiles:
    """Test configuration profiles."""
    
    def test_get_profile(self, monkeypatch):
        """Test profile detection."""
        monkeypatch.setenv("FRAUDLENS_ENVIRONMENT", "prod")
        profile = get_profile()
        assert profile == Profile.PRODUCTION
        
        monkeypatch.setenv("FRAUDLENS_ENVIRONMENT", "dev")
        profile = get_profile()
        assert profile == Profile.DEVELOPMENT
    
    def test_development_settings(self):
        """Test development profile settings."""
        settings = get_development_settings()
        
        assert settings.environment == "dev"
        assert settings.debug == True
        assert settings.cache.backend == "memory"
        assert settings.storage.backend == "memory"
    
    def test_production_settings(self):
        """Test production profile settings."""
        settings = get_production_settings()
        
        assert settings.environment == "prod"
        assert settings.debug == False
        assert settings.cache.backend == "redis"
        assert settings.storage.backend == "postgresql"
        assert settings.enable_vector_search == True
    
    def test_get_settings_for_profile(self):
        """Test getting settings by profile."""
        dev_settings = get_settings_for_profile(Profile.DEVELOPMENT)
        prod_settings = get_settings_for_profile("prod")
        
        assert dev_settings.environment == "dev"
        assert prod_settings.environment == "prod"
    
    def test_merge_settings(self):
        """Test settings merging."""
        base = FraudLensSettings()
        overrides = {
            "debug": True,
            "max_concurrent_requests": 50,
            "cache__backend": "redis",
        }
        
        merged = merge_settings(base, overrides)
        
        assert merged.debug == True
        assert merged.max_concurrent_requests == 50
        assert merged.cache.backend == "redis"


class TestGetSettings:
    """Test settings singleton."""
    
    def test_get_settings_singleton(self):
        """Test that get_settings returns same instance."""
        # Clear cache first
        get_settings.cache_clear()
        
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2
    
    def test_get_settings_with_env_file(self, tmp_path):
        """Test loading settings from .env file."""
        # Create temporary .env file
        env_file = tmp_path / ".env.test"
        env_file.write_text(
            "FRAUDLENS_ENVIRONMENT=staging\n"
            "FRAUDLENS_DEBUG=true\n"
            "FRAUDLENS_MAX_CONCURRENT_REQUESTS=75\n"
        )
        
        get_settings.cache_clear()
        settings = get_settings(env_file=str(env_file))
        
        assert settings.environment == "staging"
        assert settings.debug == True
        assert settings.max_concurrent_requests == 75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
