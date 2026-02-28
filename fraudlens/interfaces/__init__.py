"""
Protocol-based interfaces for FraudLens components.

This module defines the contracts that all implementations must follow,
enabling dependency injection, testing, and swappable implementations.

Author: Yobie Benjamin
Date: 2026-02-28
"""

from fraudlens.interfaces.analyzer import AnalyzerProtocol
from fraudlens.interfaces.cache import CacheProtocol
from fraudlens.interfaces.detector import DetectorProtocol
from fraudlens.interfaces.storage import StorageProtocol

__all__ = [
    "DetectorProtocol",
    "AnalyzerProtocol",
    "StorageProtocol",
    "CacheProtocol",
]
