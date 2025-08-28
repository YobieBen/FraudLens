"""
Multi-modal fusion engine for combining fraud detection results.

Author: Yobie Benjamin
Date: 2025
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

from fraudlens.core.base.detector import DetectionResult, FraudType, Modality


class FusionStrategy(Enum):
    """Fusion strategies for multi-modal integration."""
    EARLY = "early"  # Combine raw features
    LATE = "late"  # Combine predictions
    HYBRID = "hybrid"  # Attention-based hybrid
    HIERARCHICAL = "hierarchical"  # Multi-level fusion
    ADAPTIVE = "adaptive"  # Dynamic strategy selection


@dataclass
class ModalityWeight:
    """Weight configuration for a modality."""
    modality: Modality
    base_weight: float
    confidence_weight: float
    reliability: float = 1.0
    
    def get_effective_weight(self, confidence: float) -> float:
        """Calculate effective weight based on confidence."""
        return self.base_weight * self.confidence_weight * confidence * self.reliability


@dataclass
class FusedResult:
    """Result from multi-modal fusion."""
    fraud_score: float
    confidence: float
    fraud_types: List[FraudType]
    modality_scores: Dict[str, float]
    fusion_strategy: FusionStrategy
    consistency_score: float
    explanation: str
    evidence: Dict[str, Any]
    processing_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fraud_score": self.fraud_score,
            "confidence": self.confidence,
            "fraud_types": [ft.value for ft in self.fraud_types],
            "modality_scores": self.modality_scores,
            "fusion_strategy": self.fusion_strategy.value,
            "consistency_score": self.consistency_score,
            "explanation": self.explanation,
            "evidence": self.evidence,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RiskScore:
    """Comprehensive risk score with breakdown."""
    overall_risk: float
    risk_level: str  # low, medium, high, critical
    risk_factors: List[Dict[str, Any]]
    confidence_intervals: Tuple[float, float]
    modality_contributions: Dict[str, float]
    anomaly_score: float
    trend: str  # increasing, stable, decreasing
    metadata: Dict[str, Any] = field(default_factory=dict)


class AttentionMechanism:
    """Attention mechanism for hybrid fusion."""
    
    def __init__(self, num_modalities: int = 3, hidden_dim: int = 256):
        """Initialize attention mechanism."""
        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim
        self.attention_weights = None
        
    def compute_attention(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute attention weights for modality features.
        
        Args:
            features: Dictionary of modality features
            
        Returns:
            Attention weights
        """
        try:
            import torch
            import torch.nn.functional as F
            
            # Convert features to tensors
            feature_list = []
            for modality in ["text", "vision", "audio"]:
                if modality in features:
                    feat = features[modality]
                    if not isinstance(feat, np.ndarray):
                        feat = np.array(feat)
                    feature_list.append(feat.flatten()[:self.hidden_dim])
            
            if not feature_list:
                return np.ones(len(features)) / len(features)
            
            # Stack features
            stacked = np.stack(feature_list)
            tensor_features = torch.from_numpy(stacked).float()
            
            # Compute attention scores (simplified)
            scores = torch.matmul(tensor_features, tensor_features.T)
            attention = F.softmax(scores.sum(dim=1), dim=0)
            
            return attention.numpy()
            
        except ImportError:
            # Fallback to uniform weights
            return np.ones(len(features)) / len(features)


class MultiModalFraudFusion:
    """
    Multi-modal fusion system for fraud detection.
    
    Combines insights from text, vision, audio, and video inputs
    to generate unified fraud risk scores.
    """
    
    def __init__(
        self,
        strategy: FusionStrategy = FusionStrategy.HYBRID,
        weights_config: Optional[Dict[str, float]] = None,
        enable_attention: bool = True,
        cache_size: int = 100,
    ):
        """
        Initialize fusion system.
        
        Args:
            strategy: Fusion strategy to use
            weights_config: Modality weight configuration
            enable_attention: Enable attention mechanism for hybrid fusion
            cache_size: Size of result cache
        """
        self.strategy = strategy
        self.enable_attention = enable_attention
        self.cache_size = cache_size
        
        # Initialize modality weights
        self.modality_weights = self._init_weights(weights_config)
        
        # Initialize attention mechanism
        self.attention = AttentionMechanism() if enable_attention else None
        
        # Cache for fusion results
        self._cache = {}
        self._cache_order = []
        
        # Performance tracking
        self._fusion_times = []
        self._strategy_performance = {s: [] for s in FusionStrategy}
        
        logger.info(f"MultiModalFraudFusion initialized with {strategy.value} strategy")
    
    def _init_weights(self, config: Optional[Dict[str, float]] = None) -> Dict[str, ModalityWeight]:
        """Initialize modality weights."""
        default_weights = {
            "text": {"base": 0.35, "confidence": 0.9},
            "image": {"base": 0.35, "confidence": 0.85},
            "audio": {"base": 0.20, "confidence": 0.8},
            "video": {"base": 0.10, "confidence": 0.75},
        }
        
        if config:
            default_weights.update(config)
        
        weights = {}
        for modality, params in default_weights.items():
            weights[modality] = ModalityWeight(
                modality=Modality(modality),
                base_weight=params["base"],
                confidence_weight=params["confidence"],
            )
        
        return weights
    
    async def fuse_modalities(
        self,
        text_result: Optional[DetectionResult] = None,
        vision_result: Optional[DetectionResult] = None,
        audio_result: Optional[DetectionResult] = None,
        video_result: Optional[DetectionResult] = None,
        features: Optional[Dict[str, np.ndarray]] = None,
    ) -> FusedResult:
        """
        Fuse results from multiple modalities.
        
        Args:
            text_result: Text analysis result
            vision_result: Vision analysis result
            audio_result: Audio analysis result
            video_result: Video analysis result
            features: Raw features for early fusion
            
        Returns:
            Fused detection result
        """
        start_time = time.time()
        
        # Collect available results
        results = {
            "text": text_result,
            "vision": vision_result,
            "audio": audio_result,
            "video": video_result,
        }
        results = {k: v for k, v in results.items() if v is not None}
        
        if not results:
            return self._create_empty_result()
        
        # Check cache
        cache_key = self._generate_cache_key(results)
        if cache_key in self._cache:
            logger.debug("Returning cached fusion result")
            return self._cache[cache_key]
        
        # Select fusion strategy
        if self.strategy == FusionStrategy.ADAPTIVE:
            strategy = self._select_strategy(results, features)
        else:
            strategy = self.strategy
        
        # Perform fusion based on strategy
        if strategy == FusionStrategy.EARLY:
            fused = await self._early_fusion(results, features)
        elif strategy == FusionStrategy.LATE:
            fused = await self._late_fusion(results)
        elif strategy == FusionStrategy.HYBRID:
            fused = await self._hybrid_fusion(results, features)
        elif strategy == FusionStrategy.HIERARCHICAL:
            fused = await self._hierarchical_fusion(results, features)
        else:
            fused = await self._late_fusion(results)  # Default
        
        # Calculate consistency
        consistency = await self._calculate_consistency(results)
        fused.consistency_score = consistency
        
        # Track performance
        processing_time = (time.time() - start_time) * 1000
        fused.processing_time_ms = processing_time
        self._fusion_times.append(processing_time)
        
        # Cache result
        self._add_to_cache(cache_key, fused)
        
        return fused
    
    async def _early_fusion(
        self,
        results: Dict[str, DetectionResult],
        features: Optional[Dict[str, np.ndarray]] = None,
    ) -> FusedResult:
        """Early fusion: combine raw features."""
        if not features:
            # Extract features from results if not provided
            features = self._extract_features(results)
        
        # Concatenate features
        combined_features = []
        modality_scores = {}
        
        for modality, feat in features.items():
            if modality in results:
                combined_features.append(feat.flatten())
                modality_scores[modality] = results[modality].fraud_score
        
        if not combined_features:
            return await self._late_fusion(results)
        
        # Simple averaging for now (can be replaced with ML model)
        combined = np.concatenate(combined_features)
        fraud_score = float(np.clip(combined.mean() + 0.5, 0, 1))
        
        # Aggregate fraud types
        fraud_types = set()
        for result in results.values():
            fraud_types.update(result.fraud_types)
        
        return FusedResult(
            fraud_score=fraud_score,
            confidence=np.mean([r.confidence for r in results.values()]),
            fraud_types=list(fraud_types),
            modality_scores=modality_scores,
            fusion_strategy=FusionStrategy.EARLY,
            consistency_score=0.0,  # Will be updated
            explanation=f"Early fusion of {len(results)} modalities",
            evidence={"features_shape": combined.shape},
            processing_time_ms=0.0,  # Will be updated
        )
    
    async def _late_fusion(self, results: Dict[str, DetectionResult]) -> FusedResult:
        """Late fusion: combine predictions."""
        weighted_scores = []
        total_weight = 0
        modality_scores = {}
        fraud_types = set()
        confidences = []
        
        for modality, result in results.items():
            # Always include the modality score
            modality_scores[modality] = result.fraud_score
            fraud_types.update(result.fraud_types)
            confidences.append(result.confidence)
            
            # Apply weight if available
            if modality in self.modality_weights:
                weight = self.modality_weights[modality].get_effective_weight(result.confidence)
                weighted_scores.append(result.fraud_score * weight)
                total_weight += weight
            else:
                # Default weight for unknown modalities
                default_weight = 0.1
                weighted_scores.append(result.fraud_score * default_weight)
                total_weight += default_weight
        
        if total_weight > 0:
            fraud_score = sum(weighted_scores) / total_weight
        else:
            fraud_score = np.mean([r.fraud_score for r in results.values()])
        
        return FusedResult(
            fraud_score=float(np.clip(fraud_score, 0, 1)),
            confidence=float(np.mean(confidences)),
            fraud_types=list(fraud_types),
            modality_scores=modality_scores,
            fusion_strategy=FusionStrategy.LATE,
            consistency_score=0.0,  # Will be updated
            explanation=f"Weighted fusion of {len(results)} modalities",
            evidence={"weights": {k: v.base_weight for k, v in self.modality_weights.items()}},
            processing_time_ms=0.0,  # Will be updated
        )
    
    async def _hybrid_fusion(
        self,
        results: Dict[str, DetectionResult],
        features: Optional[Dict[str, np.ndarray]] = None,
    ) -> FusedResult:
        """Hybrid fusion with attention mechanism."""
        # Get both early and late fusion results
        early = await self._early_fusion(results, features)
        late = await self._late_fusion(results)
        
        # Apply attention if available
        if self.attention and features:
            attention_weights = self.attention.compute_attention(features)
            
            # Weight modality scores by attention
            modality_scores = {}
            for i, (modality, result) in enumerate(results.items()):
                if i < len(attention_weights):
                    modality_scores[modality] = result.fraud_score * attention_weights[i]
                else:
                    modality_scores[modality] = result.fraud_score
            
            # Combine early and late with attention
            fraud_score = (0.4 * early.fraud_score + 0.6 * late.fraud_score)
            fraud_score = float(np.clip(fraud_score, 0, 1))
        else:
            # Simple combination without attention
            fraud_score = (0.5 * early.fraud_score + 0.5 * late.fraud_score)
            modality_scores = late.modality_scores
        
        # Combine fraud types from both
        fraud_types = set(early.fraud_types) | set(late.fraud_types)
        
        return FusedResult(
            fraud_score=fraud_score,
            confidence=(early.confidence + late.confidence) / 2,
            fraud_types=list(fraud_types),
            modality_scores=modality_scores,
            fusion_strategy=FusionStrategy.HYBRID,
            consistency_score=0.0,  # Will be updated
            explanation=f"Hybrid fusion with {'attention' if self.attention else 'averaging'}",
            evidence={
                "early_score": early.fraud_score,
                "late_score": late.fraud_score,
                "attention_enabled": self.attention is not None,
            },
            processing_time_ms=0.0,  # Will be updated
        )
    
    async def _hierarchical_fusion(
        self,
        results: Dict[str, DetectionResult],
        features: Optional[Dict[str, np.ndarray]] = None,
    ) -> FusedResult:
        """Hierarchical multi-level fusion."""
        # Level 1: Group related modalities
        text_vision = {}
        audio_video = {}
        
        if "text" in results:
            text_vision["text"] = results["text"]
        if "vision" in results:
            text_vision["vision"] = results["vision"]
        if "audio" in results:
            audio_video["audio"] = results["audio"]
        if "video" in results:
            audio_video["video"] = results["video"]
        
        # Level 2: Fuse within groups
        group_scores = []
        if text_vision:
            tv_fused = await self._late_fusion(text_vision)
            group_scores.append(tv_fused.fraud_score)
        if audio_video:
            av_fused = await self._late_fusion(audio_video)
            group_scores.append(av_fused.fraud_score)
        
        # Level 3: Final fusion
        if group_scores:
            fraud_score = float(np.mean(group_scores))
        else:
            fraud_score = 0.0
        
        # Collect all fraud types
        fraud_types = set()
        modality_scores = {}
        confidences = []
        
        for modality, result in results.items():
            fraud_types.update(result.fraud_types)
            modality_scores[modality] = result.fraud_score
            confidences.append(result.confidence)
        
        return FusedResult(
            fraud_score=fraud_score,
            confidence=float(np.mean(confidences)) if confidences else 0.0,
            fraud_types=list(fraud_types),
            modality_scores=modality_scores,
            fusion_strategy=FusionStrategy.HIERARCHICAL,
            consistency_score=0.0,  # Will be updated
            explanation=f"Hierarchical fusion with {len(group_scores)} groups",
            evidence={"group_scores": group_scores},
            processing_time_ms=0.0,  # Will be updated
        )
    
    def _select_strategy(
        self,
        results: Dict[str, DetectionResult],
        features: Optional[Dict[str, np.ndarray]] = None,
    ) -> FusionStrategy:
        """Adaptively select fusion strategy based on inputs."""
        num_modalities = len(results)
        
        # Simple heuristics for strategy selection
        if num_modalities == 1:
            return FusionStrategy.LATE
        
        if features and all(f.size > 100 for f in features.values()):
            # Rich features available - use hybrid
            return FusionStrategy.HYBRID
        
        if num_modalities > 3:
            # Many modalities - use hierarchical
            return FusionStrategy.HIERARCHICAL
        
        # Check confidence variance
        confidences = [r.confidence for r in results.values()]
        confidence_var = np.var(confidences)
        
        if confidence_var > 0.1:
            # High variance - use weighted late fusion
            return FusionStrategy.LATE
        else:
            # Similar confidence - use early fusion if features available
            return FusionStrategy.EARLY if features else FusionStrategy.LATE
    
    async def _calculate_consistency(self, results: Dict[str, DetectionResult]) -> float:
        """Calculate consistency across modalities."""
        if len(results) < 2:
            return 1.0
        
        scores = [r.fraud_score for r in results.values()]
        fraud_types_sets = [set(r.fraud_types) for r in results.values()]
        
        # Score consistency
        score_consistency = 1.0 - np.std(scores) if scores else 0.0
        
        # Fraud type consistency (Jaccard similarity)
        type_consistency_scores = []
        for i in range(len(fraud_types_sets)):
            for j in range(i + 1, len(fraud_types_sets)):
                set1, set2 = fraud_types_sets[i], fraud_types_sets[j]
                if set1 or set2:
                    jaccard = len(set1 & set2) / len(set1 | set2)
                    type_consistency_scores.append(jaccard)
        
        type_consistency = np.mean(type_consistency_scores) if type_consistency_scores else 0.5
        
        # Weighted average
        consistency = 0.6 * score_consistency + 0.4 * type_consistency
        
        return float(np.clip(consistency, 0, 1))
    
    def _extract_features(self, results: Dict[str, DetectionResult]) -> Dict[str, np.ndarray]:
        """Extract features from detection results."""
        features = {}
        
        for modality, result in results.items():
            # Create feature vector from result
            feat = [
                result.fraud_score,
                result.confidence,
                len(result.fraud_types),
                result.processing_time_ms / 1000.0,  # Normalize
            ]
            
            # Add fraud type indicators
            for fraud_type in FraudType:
                feat.append(1.0 if fraud_type in result.fraud_types else 0.0)
            
            features[modality] = np.array(feat)
        
        return features
    
    def _generate_cache_key(self, results: Dict[str, DetectionResult]) -> str:
        """Generate cache key from results."""
        key_parts = []
        for modality in sorted(results.keys()):
            result = results[modality]
            key_parts.append(f"{modality}:{result.fraud_score:.3f}")
        return "|".join(key_parts)
    
    def _add_to_cache(self, key: str, result: FusedResult) -> None:
        """Add result to cache."""
        self._cache[key] = result
        self._cache_order.append(key)
        
        # Maintain cache size
        if len(self._cache) > self.cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
    
    def _create_empty_result(self) -> FusedResult:
        """Create empty fusion result."""
        return FusedResult(
            fraud_score=0.0,
            confidence=0.0,
            fraud_types=[],
            modality_scores={},
            fusion_strategy=self.strategy,
            consistency_score=1.0,
            explanation="No modality results available",
            evidence={},
            processing_time_ms=0.0,
        )
    
    def update_weights(self, modality: str, weight: float) -> None:
        """Update modality weight."""
        if modality in self.modality_weights:
            self.modality_weights[modality].base_weight = weight
            logger.info(f"Updated {modality} weight to {weight}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get fusion performance statistics."""
        return {
            "avg_fusion_time_ms": np.mean(self._fusion_times) if self._fusion_times else 0,
            "total_fusions": len(self._fusion_times),
            "cache_hit_rate": len(self._cache) / max(len(self._fusion_times), 1),
            "strategy_usage": {
                s.value: len(perf) for s, perf in self._strategy_performance.items()
            },
        }