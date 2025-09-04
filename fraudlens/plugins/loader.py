"""
Plugin loader for dynamic extension loading.

Author: Yobie Benjamin
Date: 2025-08-26 18:34:00 PDT
"""

import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import yaml

from fraudlens.plugins.base import FraudLensPlugin, PluginMetadata


class PluginLoader:
    """
    Load and manage FraudLens plugins.

    Supports loading plugins from:
    - Python packages
    - Directory paths
    - ZIP files
    - Remote repositories (with git)
    """

    def __init__(
        self,
        plugin_dirs: Optional[List[Path]] = None,
        auto_discover: bool = True,
    ):
        """
        Initialize plugin loader.

        Args:
            plugin_dirs: Directories to search for plugins
            auto_discover: Whether to auto-discover plugins
        """
        self.plugin_dirs = plugin_dirs or [Path("plugins")]
        self.auto_discover = auto_discover

        self._plugins: Dict[str, FraudLensPlugin] = {}
        self._plugin_classes: Dict[str, Type[FraudLensPlugin]] = {}

        if self.auto_discover:
            self.discover_plugins()

    def discover_plugins(self) -> List[str]:
        """
        Discover available plugins.

        Returns:
            List of discovered plugin names
        """
        discovered = []

        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue

            # Look for plugin directories
            for path in plugin_dir.iterdir():
                if path.is_dir() and not path.name.startswith("_"):
                    # Check for plugin.yaml or plugin.json
                    config_file = path / "plugin.yaml"
                    if not config_file.exists():
                        config_file = path / "plugin.json"

                    if config_file.exists():
                        try:
                            plugin_name = self._load_plugin_from_directory(path)
                            if plugin_name:
                                discovered.append(plugin_name)
                        except Exception as e:
                            print(f"Failed to load plugin from {path}: {e}")

            # Look for Python files
            for path in plugin_dir.glob("*.py"):
                if not path.name.startswith("_"):
                    try:
                        plugin_name = self._load_plugin_from_file(path)
                        if plugin_name:
                            discovered.append(plugin_name)
                    except Exception as e:
                        print(f"Failed to load plugin from {path}: {e}")

        return discovered

    def _load_plugin_from_directory(self, path: Path) -> Optional[str]:
        """Load plugin from directory."""
        # Read configuration
        config_file = path / "plugin.yaml"
        if not config_file.exists():
            config_file = path / "plugin.json"

        if config_file.suffix == ".yaml":
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
        else:
            with open(config_file, "r") as f:
                config = json.load(f)

        # Get plugin entry point
        entry_point = config.get("entry_point", "__init__.py")
        plugin_file = path / entry_point

        if not plugin_file.exists():
            return None

        # Load the module
        module_name = f"fraudlens_plugin_{path.name}"
        spec = importlib.util.spec_from_file_location(module_name, plugin_file)
        if not spec or not spec.loader:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Find plugin class
        plugin_class = self._find_plugin_class(module)
        if not plugin_class:
            return None

        # Register plugin class
        plugin_name = config.get("name", path.name)
        self._plugin_classes[plugin_name] = plugin_class

        return plugin_name

    def _load_plugin_from_file(self, path: Path) -> Optional[str]:
        """Load plugin from Python file."""
        module_name = f"fraudlens_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if not spec or not spec.loader:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Find plugin class
        plugin_class = self._find_plugin_class(module)
        if not plugin_class:
            return None

        # Get plugin name from class
        try:
            temp_instance = plugin_class()
            metadata = temp_instance.get_metadata()
            plugin_name = metadata.name
        except:
            plugin_name = path.stem

        self._plugin_classes[plugin_name] = plugin_class

        return plugin_name

    def _find_plugin_class(self, module: Any) -> Optional[Type[FraudLensPlugin]]:
        """Find FraudLensPlugin subclass in module."""
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, FraudLensPlugin) and obj != FraudLensPlugin:
                return obj
        return None

    async def load_plugin(
        self, plugin_name: str, config: Optional[Dict[str, Any]] = None
    ) -> FraudLensPlugin:
        """
        Load and initialize a plugin.

        Args:
            plugin_name: Name of plugin to load
            config: Plugin configuration

        Returns:
            Initialized plugin instance
        """
        if plugin_name in self._plugins:
            return self._plugins[plugin_name]

        if plugin_name not in self._plugin_classes:
            # Try to discover again
            self.discover_plugins()
            if plugin_name not in self._plugin_classes:
                raise ValueError(f"Plugin '{plugin_name}' not found")

        # Create plugin instance
        plugin_class = self._plugin_classes[plugin_name]
        plugin = plugin_class()

        # Validate dependencies
        valid, missing = plugin.validate_dependencies()
        if not valid:
            raise ImportError(f"Plugin '{plugin_name}' has missing dependencies: {missing}")

        # Initialize plugin
        await plugin.initialize(config or {})

        # Store plugin
        self._plugins[plugin_name] = plugin

        return plugin

    async def unload_plugin(self, plugin_name: str) -> None:
        """
        Unload a plugin.

        Args:
            plugin_name: Name of plugin to unload
        """
        if plugin_name in self._plugins:
            plugin = self._plugins[plugin_name]
            await plugin.cleanup()
            del self._plugins[plugin_name]

    def get_plugin(self, plugin_name: str) -> Optional[FraudLensPlugin]:
        """
        Get loaded plugin instance.

        Args:
            plugin_name: Plugin name

        Returns:
            Plugin instance or None
        """
        return self._plugins.get(plugin_name)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """
        List all available plugins.

        Returns:
            List of plugin information dictionaries
        """
        plugins = []

        # List loaded plugins
        for name, plugin in self._plugins.items():
            info = plugin.get_info()
            info["loaded"] = True
            plugins.append(info)

        # List discovered but not loaded plugins
        for name, plugin_class in self._plugin_classes.items():
            if name not in self._plugins:
                try:
                    temp_instance = plugin_class()
                    metadata = temp_instance.get_metadata()
                    plugins.append(
                        {
                            "metadata": metadata.to_dict(),
                            "loaded": False,
                        }
                    )
                except:
                    plugins.append(
                        {
                            "metadata": {"name": name},
                            "loaded": False,
                        }
                    )

        return plugins

    def install_plugin(self, source: str, force: bool = False) -> str:
        """
        Install plugin from source.

        Args:
            source: Plugin source (git URL, path, package name)
            force: Whether to force reinstall

        Returns:
            Installed plugin name
        """
        # This would implement plugin installation from various sources
        # For now, just a placeholder
        raise NotImplementedError("Plugin installation not yet implemented")

    async def reload_plugin(self, plugin_name: str) -> FraudLensPlugin:
        """
        Reload a plugin.

        Args:
            plugin_name: Plugin to reload

        Returns:
            Reloaded plugin instance
        """
        # Get current config if loaded
        config = {}
        if plugin_name in self._plugins:
            # Save config before unloading
            await self.unload_plugin(plugin_name)

        # Reload the module
        if plugin_name in self._plugin_classes:
            del self._plugin_classes[plugin_name]

        # Rediscover and load
        self.discover_plugins()
        return await self.load_plugin(plugin_name, config)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PluginLoader("
            f"loaded={len(self._plugins)}, "
            f"available={len(self._plugin_classes)})"
        )
