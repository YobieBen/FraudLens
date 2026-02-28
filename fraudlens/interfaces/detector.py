"""
Detector protocol for fraud detection implementations.

Author: Yobie Benjamin
Date: 2026-02-28
"""

from pathlib import Path
from typing import Any, AsyncIterator, Protocol, Union

import numpy as np
from numpy.typing import NDArray

from fraudlens.core.base.detector import DetectionResult


class DetectorProtocol(Protocol):
    """
    Protocol defining the interface for fraud detectors.
    
    Any fraud detector implementation must follow this contract to be
    compatible with the FraudLens framework.
    """
    
    detector_id: str
    """Unique identifier for this detector instance."""
    
    async def initialize(self) -> None:
        """
        Initialize the detector and load required resources.
        
        This should be idempotent - calling multiple times should be safe.
        """
        ...
    
    async def detect(
        self,
        input_data: Union[str, bytes, NDArray[np.uint8], Path],
        **kwargs: Any,
    ) -> DetectionResult:
        """
        Perform fraud detection on input data.
        
        Args:
            input_data: Input to analyze (format depends on detector type)
            **kwargs: Additional detector-specific parameters
        
        Returns:
            Detection result with fraud score and analysis
        
        Raises:
            ValueError: If input format is invalid
            RuntimeError: If detector not initialized
        """
        ...
    
    async def detect_stream(
        self,
        input_data: Union[str, bytes, NDArray[np.uint8], Path],
        **kwargs: Any,
    ) -> AsyncIterator[DetectionResult]:
        """
        Stream detection results as they become available.
        
        Args:
            input_data: Input to analyze
            **kwargs: Additional parameters
        
        Yields:
            Partial or incremental detection results
        """
        ...
    
    async def cleanup(self) -> None:
        """
        Clean up resources and unload models.
        
        Should release memory, close connections, etc.
        """
        ...
    
    def get_memory_usage(self) -> int:
        """
        Get current memory usage in bytes.
        
        Returns:
            Memory usage in bytes
        """
        ...
    
    def validate_input(
        self,
        input_data: Union[str, bytes, NDArray[np.uint8], Path],
    ) -> bool:
        """
        Validate input data format and content.
        
        Args:
            input_data: Input to validate
        
        Returns:
            True if input is valid, False otherwise
        """
        ...
    
    def get_capabilities(self) -> dict[str, Any]:
        """
        Get detector capabilities and metadata.
        
        Returns:
            Dictionary with detector information:
            - supported_modalities: List of supported input types
            - max_input_size: Maximum input size in bytes
            - streaming_support: Whether streaming is supported
            - batch_support: Whether batch processing is supported
        """
        ...
