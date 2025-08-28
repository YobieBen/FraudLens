"""
Plugin Manager for FraudLens extensibility.

Author: Yobie Benjamin
Date: 2025-08-27 18:49:00 PDT
"""

import asyncio
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from fraudlens.plugins.base import FraudDetectorPlugin


class PluginManager:
    """
    Manages loading, registration, and execution of fraud detection plugins.
    """
    
    def __init__(self, plugin_dir: Optional[Path] = None):
        """
        Initialize plugin manager.
        
        Args:
            plugin_dir: Directory containing plugins
        """
        self.plugin_dir = plugin_dir or Path("plugins")
        self._plugins: Dict[str, FraudDetectorPlugin] = {}
        self._initialized = False
    
    async def load_plugins(self):
        """Load all plugins from plugin directory."""
        if not self.plugin_dir.exists():
            logger.warning(f"Plugin directory {self.plugin_dir} does not exist")
            return
        
        logger.info(f"Loading plugins from {self.plugin_dir}")
        
        # Find all Python files in plugin directory
        plugin_files = list(self.plugin_dir.glob("*.py"))
        
        for plugin_file in plugin_files:
            if plugin_file.stem.startswith("_"):
                continue  # Skip private files
            
            try:
                await self._load_plugin_file(plugin_file)
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_file}: {e}")
        
        self._initialized = True
        logger.info(f"Loaded {len(self._plugins)} plugins")
    
    async def _load_plugin_file(self, plugin_file: Path):
        """Load a single plugin file."""
        # Load module dynamically
        spec = importlib.util.spec_from_file_location(
            plugin_file.stem,
            plugin_file
        )
        
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin classes
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, FraudDetectorPlugin) and
                    obj != FraudDetectorPlugin):
                    
                    # Instantiate plugin
                    plugin = obj()
                    await plugin.initialize()
                    self.register_plugin(plugin)
                    
                    logger.info(f"Loaded plugin: {plugin.name} v{plugin.version}")
    
    def register_plugin(self, plugin: FraudDetectorPlugin):
        """
        Register a plugin.
        
        Args:
            plugin: Plugin instance to register
        """
        if not isinstance(plugin, FraudDetectorPlugin):
            raise ValueError("Plugin must inherit from FraudDetectorPlugin")
        
        self._plugins[plugin.name] = plugin
        logger.debug(f"Registered plugin: {plugin.name}")
    
    def unregister_plugin(self, name: str):
        """
        Unregister a plugin.
        
        Args:
            name: Plugin name
        """
        if name in self._plugins:
            del self._plugins[name]
            logger.debug(f"Unregistered plugin: {name}")
    
    def get_plugin(self, name: str) -> Optional[FraudDetectorPlugin]:
        """
        Get plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None
        """
        return self._plugins.get(name)
    
    def list_plugins(self) -> List[Dict[str, str]]:
        """
        List all registered plugins.
        
        Returns:
            List of plugin information
        """
        return [
            {
                "name": plugin.name,
                "version": plugin.version,
                "description": plugin.description,
                "author": plugin.author,
                "enabled": plugin.enabled,
            }
            for plugin in self._plugins.values()
        ]
    
    def has_plugins(self) -> bool:
        """Check if any plugins are registered."""
        return len(self._plugins) > 0
    
    async def execute_plugin(
        self,
        plugin_name: str,
        input_data: Any,
        **kwargs
    ) -> Optional[Dict]:
        """
        Execute a specific plugin.
        
        Args:
            plugin_name: Name of plugin to execute
            input_data: Input data for plugin
            **kwargs: Additional arguments
            
        Returns:
            Plugin results or None
        """
        plugin = self._plugins.get(plugin_name)
        if not plugin:
            logger.warning(f"Plugin {plugin_name} not found")
            return None
        
        if not plugin.enabled:
            logger.debug(f"Plugin {plugin_name} is disabled")
            return None
        
        try:
            result = await plugin.process(input_data, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Plugin {plugin_name} execution failed: {e}")
            return None
    
    async def execute_plugins(
        self,
        input_data: Any,
        detection_result: Any = None,
        **kwargs
    ) -> List[Dict]:
        """
        Execute all enabled plugins.
        
        Args:
            input_data: Input data for plugins
            detection_result: Initial detection result
            **kwargs: Additional arguments
            
        Returns:
            List of plugin results
        """
        results = []
        
        # Execute plugins concurrently
        tasks = []
        plugin_names = []
        
        for name, plugin in self._plugins.items():
            if plugin.enabled:
                tasks.append(
                    plugin.process(input_data, detection_result=detection_result, **kwargs)
                )
                plugin_names.append(name)
        
        if tasks:
            plugin_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for name, result in zip(plugin_names, plugin_results):
                if isinstance(result, Exception):
                    logger.error(f"Plugin {name} failed: {result}")
                elif result is not None:
                    results.append({"plugin": name, **result})
        
        return results
    
    def enable_plugin(self, name: str):
        """Enable a plugin."""
        plugin = self._plugins.get(name)
        if plugin:
            plugin.enabled = True
            logger.info(f"Enabled plugin: {name}")
    
    def disable_plugin(self, name: str):
        """Disable a plugin."""
        plugin = self._plugins.get(name)
        if plugin:
            plugin.enabled = False
            logger.info(f"Disabled plugin: {name}")
    
    async def reload_plugins(self):
        """Reload all plugins."""
        logger.info("Reloading plugins...")
        
        # Clean up existing plugins
        for plugin in self._plugins.values():
            if hasattr(plugin, 'cleanup'):
                await plugin.cleanup()
        
        self._plugins.clear()
        
        # Load plugins again
        await self.load_plugins()
    
    async def cleanup(self):
        """Clean up plugin manager."""
        logger.info("Cleaning up plugin manager...")
        
        # Clean up all plugins
        cleanup_tasks = []
        for plugin in self._plugins.values():
            if hasattr(plugin, 'cleanup'):
                cleanup_tasks.append(plugin.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self._plugins.clear()
        self._initialized = False
        
        logger.info("Plugin manager cleanup complete")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get plugin statistics."""
        return {
            "total_plugins": len(self._plugins),
            "enabled_plugins": sum(1 for p in self._plugins.values() if p.enabled),
            "disabled_plugins": sum(1 for p in self._plugins.values() if not p.enabled),
            "plugins": self.list_plugins(),
        }