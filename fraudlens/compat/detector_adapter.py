"""
Detector adapter for backward compatibility.

Author: Yobie Benjamin
Date: 2026-02-28
"""

from typing import Any


class DetectorAdapter:
    """
    Adapter to make new detector implementations work with old code.
    
    This is a placeholder for future detector compatibility needs.
    """
    
    def __init__(self, detector: Any):
        """
        Initialize detector adapter.
        
        Args:
            detector: New-style detector
        """
        self._detector = detector
    
    async def detect(self, *args, **kwargs):
        """Forward detect call to wrapped detector."""
        return await self._detector.detect(*args, **kwargs)
    
    async def initialize(self):
        """Forward initialize call."""
        if hasattr(self._detector, "initialize"):
            return await self._detector.initialize()
    
    async def cleanup(self):
        """Forward cleanup call."""
        if hasattr(self._detector, "cleanup"):
            return await self._detector.cleanup()
