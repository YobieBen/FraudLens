"""
Plugin system for FraudLens.

Author: Yobie Benjamin
Date: 2025-08-26 18:34:00 PDT
"""

from fraudlens.plugins.base import FraudLensPlugin
from fraudlens.plugins.loader import PluginLoader

__all__ = ["PluginLoader", "FraudLensPlugin"]
