"""
Backward compatibility adapters for FraudLens.

Provides compatibility layer between old and new architecture
to ensure existing code continues to work during transition.

Author: Yobie Benjamin
Date: 2026-02-28
"""

from fraudlens.compat.config_adapter import ConfigAdapter
from fraudlens.compat.detector_adapter import DetectorAdapter

__all__ = [
    "ConfigAdapter",
    "DetectorAdapter",
]
