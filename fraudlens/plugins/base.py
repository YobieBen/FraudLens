"""
Base plugin interface for FraudLens extensions.

Author: Yobie Benjamin
Date: 2025-08-26 18:34:00 PDT
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Tuple

from fraudlens.core.base.detector import FraudDetector
from fraudlens.core.base.processor import ModalityProcessor
from fraudlens.core.base.scorer import RiskScorer


@dataclass
class PluginMetadata:
    """Metadata for a FraudLens plugin."""
    
    name: str
    version: str
    author: str
    description: str
    dependencies: List[str]
    fraudlens_version: str
    license: str
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "dependencies": self.dependencies,
            "fraudlens_version": self.fraudlens_version,
            "license": self.license,
            "tags": self.tags,
        }


class FraudDetectorPlugin(ABC):
    """Simple base class for fraud detector plugins."""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        """Initialize plugin."""
        self.name = name
        self.version = version
        self.enabled = True
        self.description = ""
        self.author = ""
    
    async def initialize(self):
        """Initialize plugin (optional)."""
        pass
    
    @abstractmethod
    async def process(self, data: Any, **kwargs) -> Optional[Dict]:
        """Process data through plugin."""
        pass
    
    async def cleanup(self):
        """Clean up plugin resources (optional)."""
        pass


class FraudLensPlugin(ABC):
    """
    Abstract base class for FraudLens plugins.
    
    Plugins can provide:
    - Custom fraud detectors
    - Modality processors
    - Risk scorers
    - Processing pipelines
    - Model implementations
    """
    
    def __init__(self, plugin_dir: Optional[Path] = None):
        """
        Initialize plugin.
        
        Args:
            plugin_dir: Directory containing plugin files
        """
        self.plugin_dir = plugin_dir
        self._initialized = False
        self._resources: Dict[str, Any] = {}
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize plugin with configuration.
        
        Args:
            config: Plugin configuration dictionary
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass
    
    def get_detectors(self) -> Dict[str, Type[FraudDetector]]:
        """
        Get fraud detector classes provided by this plugin.
        
        Returns:
            Dictionary mapping detector names to detector classes
        """
        return {}
    
    def get_processors(self) -> Dict[str, Type[ModalityProcessor]]:
        """
        Get modality processor classes provided by this plugin.
        
        Returns:
            Dictionary mapping processor names to processor classes
        """
        return {}
    
    def get_scorers(self) -> Dict[str, Type[RiskScorer]]:
        """
        Get risk scorer classes provided by this plugin.
        
        Returns:
            Dictionary mapping scorer names to scorer classes
        """
        return {}
    
    def get_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get model definitions provided by this plugin.
        
        Returns:
            Dictionary mapping model names to model configurations
        """
        return {}
    
    def validate_dependencies(self) -> Tuple[bool, List[str]]:
        """
        Validate plugin dependencies.
        
        Returns:
            Tuple of (is_valid, missing_dependencies)
        """
        metadata = self.get_metadata()
        missing = []
        
        for dep in metadata.dependencies:
            try:
                __import__(dep.split("==")[0])
            except ImportError:
                missing.append(dep)
        
        return len(missing) == 0, missing
    
    def get_configuration_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for plugin configuration.
        
        Returns:
            JSON schema dictionary
        """
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        metadata = self.get_metadata()
        return {
            "metadata": metadata.to_dict(),
            "initialized": self._initialized,
            "detectors": list(self.get_detectors().keys()),
            "processors": list(self.get_processors().keys()),
            "scorers": list(self.get_scorers().keys()),
            "models": list(self.get_models().keys()),
        }
    
    def __repr__(self) -> str:
        """String representation."""
        metadata = self.get_metadata()
        return (
            f"{self.__class__.__name__}("
            f"name='{metadata.name}', "
            f"version='{metadata.version}', "
            f"initialized={self._initialized})"
        )