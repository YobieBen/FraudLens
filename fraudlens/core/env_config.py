"""
FraudLens Environment Configuration Management
Centralized configuration with environment variable support
"""

import json
import os
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from pydantic import BaseSettings, Field, validator

# Load environment variables
env_file = os.getenv("ENV_FILE", ".env")
if Path(env_file).exists():
    load_dotenv(env_file)


class EnvironmentConfig(BaseSettings):
    """Environment-based configuration"""

    # Application
    fraudlens_env: str = Field(default="development", env="FRAUDLENS_ENV")
    debug: bool = Field(default=False, env="FRAUDLENS_DEBUG")
    log_level: str = Field(default="INFO", env="FRAUDLENS_LOG_LEVEL")
    max_memory_gb: int = Field(default=8, env="FRAUDLENS_MAX_MEMORY_GB")
    version: str = Field(default="1.0.0", env="VERSION")

    # Security
    secret_key: str = Field(default="change-me-in-production", env="SECRET_KEY")
    jwt_secret_key: str = Field(default="change-me-in-production", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")

    # Database
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="fraudlens", env="POSTGRES_DB")
    postgres_user: str = Field(default="fraudlens", env="POSTGRES_USER")
    postgres_password: str = Field(default="fraudlens", env="POSTGRES_PASSWORD")
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")

    # Redis
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=1, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")

    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    api_rate_limit: int = Field(default=1000, env="API_RATE_LIMIT")
    api_cors_origins: List[str] = Field(default=["*"], env="API_CORS_ORIGINS")
    max_upload_size_mb: int = Field(default=100, env="MAX_UPLOAD_SIZE_MB")

    # Worker
    worker_concurrency: int = Field(default=4, env="WORKER_CONCURRENCY")
    worker_max_tasks_per_child: int = Field(default=100, env="WORKER_MAX_TASKS_PER_CHILD")

    # Models
    model_path: str = Field(default="/app/models", env="MODEL_PATH")
    use_gpu: bool = Field(default=False, env="USE_GPU")
    model_cache_size_gb: int = Field(default=2, env="MODEL_CACHE_SIZE_GB")

    # External APIs
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    huggingface_api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")

    # Email
    smtp_host: Optional[str] = Field(default=None, env="SMTP_HOST")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_user: Optional[str] = Field(default=None, env="SMTP_USER")
    smtp_password: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    smtp_use_tls: bool = Field(default=True, env="SMTP_USE_TLS")
    email_from: str = Field(default="noreply@fraudlens.com", env="EMAIL_FROM")

    # Monitoring
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    grafana_user: str = Field(default="admin", env="GRAFANA_USER")
    grafana_password: str = Field(default="admin", env="GRAFANA_PASSWORD")

    # Logging
    log_file_path: str = Field(default="/app/logs/fraudlens.log", env="LOG_FILE_PATH")
    log_max_size_mb: int = Field(default=100, env="LOG_MAX_SIZE_MB")
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    log_format: str = Field(default="json", env="LOG_FORMAT")

    # Cache
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    cache_max_size_mb: int = Field(default=1000, env="CACHE_MAX_SIZE_MB")
    cache_eviction_policy: str = Field(default="lru", env="CACHE_EVICTION_POLICY")

    # Feature Flags
    enable_email_scanning: bool = Field(default=True, env="ENABLE_EMAIL_SCANNING")
    enable_video_analysis: bool = Field(default=True, env="ENABLE_VIDEO_ANALYSIS")
    enable_document_validation: bool = Field(default=True, env="ENABLE_DOCUMENT_VALIDATION")
    enable_batch_processing: bool = Field(default=True, env="ENABLE_BATCH_PROCESSING")
    enable_real_time_monitoring: bool = Field(default=True, env="ENABLE_REAL_TIME_MONITORING")

    @validator("api_cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except:
                return [v]
        return v

    @validator("database_url", pre=True, always=True)
    def construct_database_url(cls, v, values):
        if v:
            return v
        return f"postgresql://{values.get('postgres_user')}:{values.get('postgres_password')}@{values.get('postgres_host')}:{values.get('postgres_port')}/{values.get('postgres_db')}"

    @validator("redis_url", pre=True, always=True)
    def construct_redis_url(cls, v, values):
        if v:
            return v
        password = values.get("redis_password")
        if password:
            return f"redis://:{password}@{values.get('redis_host')}:{values.get('redis_port')}/{values.get('redis_db')}"
        return f"redis://{values.get('redis_host')}:{values.get('redis_port')}/{values.get('redis_db')}"

    @validator(
        "debug",
        "smtp_use_tls",
        "prometheus_enabled",
        "enable_email_scanning",
        "enable_video_analysis",
        "enable_document_validation",
        "enable_batch_processing",
        "enable_real_time_monitoring",
        "use_gpu",
        pre=True,
    )
    def parse_bool(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False


# Singleton configuration instance
env_config = EnvironmentConfig()


def get_env_config() -> EnvironmentConfig:
    """Get configuration instance"""
    return env_config


def update_env_config(**kwargs):
    """Update configuration values"""
    for key, value in kwargs.items():
        if hasattr(env_config, key):
            setattr(env_config, key, value)


def is_production() -> bool:
    """Check if running in production"""
    return env_config.fraudlens_env == "production"


def is_development() -> bool:
    """Check if running in development"""
    return env_config.fraudlens_env == "development"


def is_testing() -> bool:
    """Check if running in testing"""
    return env_config.fraudlens_env == "testing"


def get_database_url() -> str:
    """Get database connection URL"""
    return env_config.database_url


def get_redis_url() -> str:
    """Get Redis connection URL"""
    return env_config.redis_url


def feature_enabled(feature: str) -> bool:
    """Check if a feature is enabled"""
    feature_map = {
        "email": env_config.enable_email_scanning,
        "video": env_config.enable_video_analysis,
        "document": env_config.enable_document_validation,
        "batch": env_config.enable_batch_processing,
        "monitoring": env_config.enable_real_time_monitoring,
    }
    return feature_map.get(feature, False)


# Export configuration
__all__ = [
    "env_config",
    "get_env_config",
    "update_env_config",
    "is_production",
    "is_development",
    "is_testing",
    "get_database_url",
    "get_redis_url",
    "feature_enabled",
    "EnvironmentConfig",
]
