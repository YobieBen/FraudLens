"""
Model registry for versioning and management.

Author: Yobie Benjamin
Date: 2025-08-26 18:34:00 PDT
"""

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


class ModelFormat(Enum):
    """Supported model formats."""
    
    ONNX = "onnx"
    MLX = "mlx"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    COREML = "coreml"
    SAFETENSORS = "safetensors"
    CUSTOM = "custom"


class QuantizationType(Enum):
    """Model quantization types."""
    
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    FP16 = "fp16"
    MIXED = "mixed"


@dataclass
class ModelInfo:
    """Information about a registered model."""
    
    model_id: str
    name: str
    version: str
    format: ModelFormat
    modality: str
    path: Path
    size_mb: float
    checksum: str
    quantization: QuantizationType
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    tags: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "format": self.format.value,
            "modality": self.modality,
            "path": str(self.path),
            "size_mb": self.size_mb,
            "checksum": self.checksum,
            "quantization": self.quantization.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "performance_metrics": self.performance_metrics,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        """Create from dictionary."""
        return cls(
            model_id=data["model_id"],
            name=data["name"],
            version=data["version"],
            format=ModelFormat(data["format"]),
            modality=data["modality"],
            path=Path(data["path"]),
            size_mb=data["size_mb"],
            checksum=data["checksum"],
            quantization=QuantizationType(data["quantization"]),
            metadata=data["metadata"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            tags=data.get("tags", []),
            performance_metrics=data.get("performance_metrics", {}),
        )


class ModelRegistry:
    """
    Central registry for model management.
    
    Features:
    - Model versioning
    - Lazy loading support
    - Model conversion utilities
    - Performance tracking
    - Automatic quantization
    """
    
    def __init__(
        self,
        registry_dir: Optional[Path] = None,
        auto_save: bool = True,
        enable_versioning: bool = True,
    ):
        """
        Initialize model registry.
        
        Args:
            registry_dir: Directory for registry storage
            auto_save: Whether to auto-save registry changes
            enable_versioning: Whether to enable model versioning
        """
        self.registry_dir = registry_dir or Path("models/registry")
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.auto_save = auto_save
        self.enable_versioning = enable_versioning
        
        self._models: Dict[str, ModelInfo] = {}
        self._model_cache: Dict[str, Any] = {}
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load registry from disk."""
        registry_file = self.registry_dir / "registry.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                data = json.load(f)
                for model_data in data.get("models", []):
                    model_info = ModelInfo.from_dict(model_data)
                    self._models[model_info.model_id] = model_info
    
    def _save_registry(self) -> None:
        """Save registry to disk."""
        if not self.auto_save:
            return
        
        registry_file = self.registry_dir / "registry.json"
        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "models": [model.to_dict() for model in self._models.values()]
        }
        
        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_model(
        self,
        name: str,
        path: Union[str, Path],
        format: ModelFormat,
        modality: str,
        version: Optional[str] = None,
        quantization: QuantizationType = QuantizationType.NONE,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> ModelInfo:
        """
        Register a new model.
        
        Args:
            name: Model name
            path: Path to model file
            format: Model format
            modality: Input modality
            version: Model version (auto-generated if not provided)
            quantization: Quantization type
            metadata: Additional metadata
            tags: Model tags
            
        Returns:
            ModelInfo for registered model
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Generate version if needed
        if version is None:
            version = self._generate_version(name)
        
        # Calculate checksum
        checksum = self._calculate_checksum(path)
        
        # Get file size
        size_mb = path.stat().st_size / (1024 * 1024)
        
        # Generate model ID
        model_id = f"{name}_{version}_{checksum[:8]}"
        
        # Create model info
        model_info = ModelInfo(
            model_id=model_id,
            name=name,
            version=version,
            format=format,
            modality=modality,
            path=path,
            size_mb=size_mb,
            checksum=checksum,
            quantization=quantization,
            metadata=metadata or {},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=tags or [],
            performance_metrics={},
        )
        
        # Copy model to registry if versioning enabled
        if self.enable_versioning:
            registry_path = self.registry_dir / name / version / path.name
            registry_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, registry_path)
            model_info.path = registry_path
        
        # Store model info
        self._models[model_id] = model_info
        self._save_registry()
        
        return model_info
    
    def get_model(
        self,
        model_id: Optional[str] = None,
        name: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Optional[ModelInfo]:
        """
        Get model information.
        
        Args:
            model_id: Model ID
            name: Model name
            version: Model version
            
        Returns:
            ModelInfo or None
        """
        if model_id:
            return self._models.get(model_id)
        
        if name:
            # Find by name and optional version
            for model in self._models.values():
                if model.name == name:
                    if version is None or model.version == version:
                        return model
        
        return None
    
    def load_model(
        self,
        model_id: str,
        lazy: bool = True,
        device: str = "mps",
    ) -> Any:
        """
        Load a model.
        
        Args:
            model_id: Model ID to load
            lazy: Whether to use lazy loading
            device: Target device
            
        Returns:
            Loaded model object
        """
        # Check cache
        cache_key = f"{model_id}_{device}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        model_info = self._models.get(model_id)
        if not model_info:
            raise ValueError(f"Model not found: {model_id}")
        
        # Load based on format
        model = self._load_model_file(model_info, device, lazy)
        
        # Cache if not lazy loading
        if not lazy:
            self._model_cache[cache_key] = model
        
        return model
    
    def _load_model_file(
        self,
        model_info: ModelInfo,
        device: str,
        lazy: bool
    ) -> Any:
        """Load model from file."""
        if model_info.format == ModelFormat.ONNX:
            return self._load_onnx_model(model_info.path, device)
        elif model_info.format == ModelFormat.MLX:
            return self._load_mlx_model(model_info.path, device, lazy)
        elif model_info.format == ModelFormat.PYTORCH:
            return self._load_pytorch_model(model_info.path, device)
        elif model_info.format == ModelFormat.COREML:
            return self._load_coreml_model(model_info.path)
        elif model_info.format == ModelFormat.SAFETENSORS:
            return self._load_safetensors_model(model_info.path, device)
        else:
            # Custom format - return path for manual loading
            return model_info.path
    
    def _load_onnx_model(self, path: Path, device: str) -> Any:
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            
            providers = []
            if device == "mps":
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            elif device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
            
            return ort.InferenceSession(str(path), providers=providers)
        except ImportError:
            raise ImportError("onnxruntime not installed")
    
    def _load_mlx_model(self, path: Path, device: str, lazy: bool) -> Any:
        """Load MLX model."""
        try:
            import mlx.core as mx
            import mlx.nn as nn
            
            # Load weights
            weights = mx.load(str(path))
            
            if lazy:
                # Return weights dict for lazy initialization
                return weights
            else:
                # Would need model architecture to fully instantiate
                return weights
        except ImportError:
            raise ImportError("MLX not installed")
    
    def _load_pytorch_model(self, path: Path, device: str) -> Any:
        """Load PyTorch model."""
        try:
            import torch
            
            if device == "mps":
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            
            return torch.load(str(path), map_location=device)
        except ImportError:
            raise ImportError("PyTorch not installed")
    
    def _load_coreml_model(self, path: Path) -> Any:
        """Load CoreML model."""
        try:
            import coremltools as ct
            return ct.models.MLModel(str(path))
        except ImportError:
            raise ImportError("coremltools not installed")
    
    def _load_safetensors_model(self, path: Path, device: str) -> Any:
        """Load SafeTensors model."""
        try:
            from safetensors import safe_open
            
            tensors = {}
            with safe_open(str(path), framework="pt", device=device) as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
            return tensors
        except ImportError:
            raise ImportError("safetensors not installed")
    
    def unload_model(self, model_id: str) -> None:
        """
        Unload a model from cache.
        
        Args:
            model_id: Model to unload
        """
        keys_to_remove = [k for k in self._model_cache if k.startswith(model_id)]
        for key in keys_to_remove:
            del self._model_cache[key]
    
    def update_performance_metrics(
        self,
        model_id: str,
        metrics: Dict[str, float]
    ) -> None:
        """
        Update model performance metrics.
        
        Args:
            model_id: Model ID
            metrics: Performance metrics
        """
        if model_id in self._models:
            self._models[model_id].performance_metrics.update(metrics)
            self._models[model_id].updated_at = datetime.now()
            self._save_registry()
    
    def quantize_model(
        self,
        model_id: str,
        quantization: QuantizationType,
        output_path: Optional[Path] = None,
    ) -> ModelInfo:
        """
        Quantize a model.
        
        Args:
            model_id: Model to quantize
            quantization: Target quantization
            output_path: Output path for quantized model
            
        Returns:
            ModelInfo for quantized model
        """
        model_info = self._models.get(model_id)
        if not model_info:
            raise ValueError(f"Model not found: {model_id}")
        
        # This would implement actual quantization
        # For now, just create a reference
        quantized_name = f"{model_info.name}_quantized_{quantization.value}"
        quantized_version = f"{model_info.version}_q{quantization.value}"
        
        # Register as new model
        return self.register_model(
            name=quantized_name,
            path=model_info.path,  # Would be quantized path
            format=model_info.format,
            modality=model_info.modality,
            version=quantized_version,
            quantization=quantization,
            metadata={
                "original_model": model_id,
                "quantization_date": datetime.now().isoformat(),
            },
            tags=model_info.tags + ["quantized"],
        )
    
    def list_models(
        self,
        modality: Optional[str] = None,
        format: Optional[ModelFormat] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ModelInfo]:
        """
        List registered models.
        
        Args:
            modality: Filter by modality
            format: Filter by format
            tags: Filter by tags
            
        Returns:
            List of ModelInfo objects
        """
        models = list(self._models.values())
        
        if modality:
            models = [m for m in models if m.modality == modality]
        
        if format:
            models = [m for m in models if m.format == format]
        
        if tags:
            tag_set = set(tags)
            models = [m for m in models if tag_set.intersection(m.tags)]
        
        return sorted(models, key=lambda m: (m.name, m.version))
    
    def delete_model(self, model_id: str, remove_files: bool = False) -> None:
        """
        Delete a model from registry.
        
        Args:
            model_id: Model to delete
            remove_files: Whether to remove model files
        """
        if model_id not in self._models:
            return
        
        model_info = self._models[model_id]
        
        # Remove from cache
        self.unload_model(model_id)
        
        # Remove files if requested
        if remove_files and model_info.path.exists():
            if model_info.path.is_file():
                model_info.path.unlink()
            else:
                shutil.rmtree(model_info.path)
        
        # Remove from registry
        del self._models[model_id]
        self._save_registry()
    
    def export_model(
        self,
        model_id: str,
        output_path: Path,
        format: Optional[ModelFormat] = None,
    ) -> Path:
        """
        Export model to different format.
        
        Args:
            model_id: Model to export
            output_path: Output path
            format: Target format
            
        Returns:
            Path to exported model
        """
        model_info = self._models.get(model_id)
        if not model_info:
            raise ValueError(f"Model not found: {model_id}")
        
        # This would implement actual conversion
        # For now, just copy
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_info.path, output_path)
        
        return output_path
    
    def _generate_version(self, name: str) -> str:
        """Generate version number for model."""
        existing_versions = [
            m.version for m in self._models.values()
            if m.name == name
        ]
        
        if not existing_versions:
            return "1.0.0"
        
        # Parse versions and increment
        versions = []
        for v in existing_versions:
            try:
                parts = v.split('.')
                versions.append(tuple(int(p) for p in parts))
            except:
                continue
        
        if versions:
            latest = max(versions)
            return f"{latest[0]}.{latest[1]}.{latest[2] + 1}"
        
        return "1.0.0"
    
    def _calculate_checksum(self, path: Path) -> str:
        """Calculate file checksum."""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_size_gb = sum(m.size_mb for m in self._models.values()) / 1024
        
        modality_counts = {}
        format_counts = {}
        quantization_counts = {}
        
        for model in self._models.values():
            modality_counts[model.modality] = modality_counts.get(model.modality, 0) + 1
            format_counts[model.format.value] = format_counts.get(model.format.value, 0) + 1
            quantization_counts[model.quantization.value] = quantization_counts.get(model.quantization.value, 0) + 1
        
        return {
            "total_models": len(self._models),
            "total_size_gb": total_size_gb,
            "cached_models": len(self._model_cache),
            "modalities": modality_counts,
            "formats": format_counts,
            "quantization": quantization_counts,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ModelRegistry("
            f"models={len(self._models)}, "
            f"cached={len(self._model_cache)})"
        )