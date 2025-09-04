"""
Base modality processor for preprocessing inputs.

Author: Yobie Benjamin
Date: 2025-08-26 18:34:00 PDT
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class ProcessedData:
    """Container for processed modality data."""

    data: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]
    original_shape: Optional[Tuple[int, ...]]
    metadata: Dict[str, Any]
    preprocessing_steps: List[str]

    def get_tensor(self, framework: str = "numpy") -> Any:
        """
        Get data as tensor for specified framework.

        Args:
            framework: Target framework ('numpy', 'mlx', 'torch', 'tensorflow')

        Returns:
            Tensor in specified framework format
        """
        if framework == "numpy":
            return self.data
        elif framework == "mlx":
            try:
                import mlx.core as mx

                return mx.array(self.data)
            except ImportError:
                raise ImportError("MLX not installed. Install with: pip install mlx")
        elif framework == "torch":
            try:
                import torch

                return (
                    torch.from_numpy(self.data) if isinstance(self.data, np.ndarray) else self.data
                )
            except ImportError:
                raise ImportError("PyTorch not installed")
        elif framework == "tensorflow":
            try:
                import tensorflow as tf

                return tf.constant(self.data)
            except ImportError:
                raise ImportError("TensorFlow not installed")
        else:
            raise ValueError(f"Unsupported framework: {framework}")


class ModalityProcessor(ABC):
    """Abstract base class for modality-specific preprocessing."""

    def __init__(
        self,
        modality: str,
        config: Optional[Dict[str, Any]] = None,
        cache_enabled: bool = True,
        max_cache_size: int = 100,
    ):
        """
        Initialize modality processor.

        Args:
            modality: Type of modality to process
            config: Configuration dictionary
            cache_enabled: Whether to cache processed data
            max_cache_size: Maximum number of cached items
        """
        self.modality = modality
        self.config = config or {}
        self.cache_enabled = cache_enabled
        self.max_cache_size = max_cache_size
        self._cache: Dict[str, ProcessedData] = {}
        self._preprocessing_pipeline: List[callable] = []

    @abstractmethod
    async def process(
        self, input_data: Union[str, bytes, np.ndarray, Path], **kwargs
    ) -> ProcessedData:
        """
        Process raw input data for model consumption.

        Args:
            input_data: Raw input data
            **kwargs: Additional processing parameters

        Returns:
            ProcessedData ready for model inference
        """
        pass

    @abstractmethod
    async def validate(
        self, input_data: Union[str, bytes, np.ndarray, Path]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate input data.

        Args:
            input_data: Input to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass

    @abstractmethod
    def get_output_shape(self) -> Tuple[int, ...]:
        """Get expected output shape after processing."""
        pass

    @abstractmethod
    def get_preprocessing_steps(self) -> List[str]:
        """Get list of preprocessing steps applied."""
        pass

    def add_preprocessing_step(self, step: callable, name: str) -> None:
        """
        Add a preprocessing step to the pipeline.

        Args:
            step: Callable preprocessing function
            name: Name/description of the step
        """
        self._preprocessing_pipeline.append((name, step))

    def clear_cache(self) -> None:
        """Clear the processing cache."""
        self._cache.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "enabled": self.cache_enabled,
            "current_size": len(self._cache),
            "max_size": self.max_cache_size,
            "cache_keys": list(self._cache.keys()),
        }

    async def batch_process(
        self, inputs: List[Union[str, bytes, np.ndarray, Path]], batch_size: int = 32, **kwargs
    ) -> List[ProcessedData]:
        """
        Process multiple inputs in batches.

        Args:
            inputs: List of inputs to process
            batch_size: Number of items to process together
            **kwargs: Additional processing parameters

        Returns:
            List of ProcessedData objects
        """
        results = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size]
            batch_results = await self._process_batch(batch, **kwargs)
            results.extend(batch_results)
        return results

    async def _process_batch(
        self, batch: List[Union[str, bytes, np.ndarray, Path]], **kwargs
    ) -> List[ProcessedData]:
        """Process a single batch of inputs."""
        results = []
        for item in batch:
            result = await self.process(item, **kwargs)
            results.append(result)
        return results

    def __repr__(self) -> str:
        """String representation of processor."""
        return (
            f"{self.__class__.__name__}("
            f"modality='{self.modality}', "
            f"cache_enabled={self.cache_enabled}, "
            f"pipeline_steps={len(self._preprocessing_pipeline)})"
        )
