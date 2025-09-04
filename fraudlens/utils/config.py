"""
Configuration management utilities.

Author: Yobie Benjamin
Date: 2025-08-26 18:34:00 PDT
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

try:
    import toml

    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False


class ConfigManager:
    """Manage configuration files and environment variables."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        environment: str = "development",
    ):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file or directory
            environment: Environment name (development, production, testing)
        """
        self.environment = environment
        self.config_path = Path(config_path) if config_path else self._get_default_config_path()
        self._config: Dict[str, Any] = {}
        self._env_prefix = "FRAUDLENS_"

    def _get_default_config_path(self) -> Path:
        """Get default configuration path based on environment."""
        base_path = Path(__file__).parent.parent / "configs"
        return base_path / self.environment / "config.yaml"

    def load(self) -> Dict[str, Any]:
        """Load configuration from file and environment."""
        # Load from file
        if self.config_path.exists():
            if self.config_path.suffix == ".yaml" or self.config_path.suffix == ".yml":
                self._config = self._load_yaml()
            elif self.config_path.suffix == ".toml":
                self._config = self._load_toml()
            elif self.config_path.suffix == ".json":
                self._config = self._load_json()

        # Override with environment variables
        self._load_env_vars()

        return self._config

    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration."""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f) or {}

    def _load_toml(self) -> Dict[str, Any]:
        """Load TOML configuration."""
        if not TOML_AVAILABLE:
            raise ImportError("toml package not installed. Install with: pip install toml")
        with open(self.config_path, "r") as f:
            return toml.load(f)

    def _load_json(self) -> Dict[str, Any]:
        """Load JSON configuration."""
        with open(self.config_path, "r") as f:
            return json.load(f)

    def _load_env_vars(self) -> None:
        """Load configuration from environment variables."""
        for key, value in os.environ.items():
            if key.startswith(self._env_prefix):
                config_key = key[len(self._env_prefix) :].lower()
                # Convert nested keys (FRAUDLENS_MODEL_PATH -> model.path)
                nested_keys = config_key.split("_")
                self._set_nested_value(nested_keys, value)

    def _set_nested_value(self, keys: list, value: str) -> None:
        """Set nested configuration value."""
        current = self._config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Try to parse value
        try:
            parsed_value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            parsed_value = value

        current[keys[-1]] = parsed_value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        current = self._config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to file.

        Args:
            path: Output path (uses current config_path if not specified)
        """
        output_path = Path(path) if path else self.config_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix in [".yaml", ".yml"]:
            with open(output_path, "w") as f:
                yaml.safe_dump(self._config, f, default_flow_style=False)
        elif output_path.suffix == ".toml":
            if not TOML_AVAILABLE:
                raise ImportError("toml package not installed")
            with open(output_path, "w") as f:
                toml.dump(self._config, f)
        elif output_path.suffix == ".json":
            with open(output_path, "w") as f:
                json.dump(self._config, f, indent=2)

    def validate(self, schema: Dict[str, Any]) -> bool:
        """
        Validate configuration against schema.

        Args:
            schema: JSON schema dictionary

        Returns:
            True if valid, raises exception otherwise
        """
        try:
            import jsonschema

            jsonschema.validate(self._config, schema)
            return True
        except ImportError:
            raise ImportError("jsonschema package not installed")

    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self._config.copy()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ConfigManager("
            f"environment='{self.environment}', "
            f"config_path='{self.config_path}')"
        )
