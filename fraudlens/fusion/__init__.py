"""
Multi-modal fusion system for FraudLens.

This module provides advanced fusion techniques for combining insights
from text, vision, audio, and video modalities to generate unified fraud risk scores.

Author: Yobie Benjamin
Date: 2025
"""

from fraudlens.fusion.adaptive_learning import (
    AdaptiveLearner,
    OnlineLearner,
    ThresholdOptimizer,
)
from fraudlens.fusion.explainer import (
    Explanation,
    ExplanationGenerator,
    ReportExporter,
)
from fraudlens.fusion.fusion_engine import (
    FusedResult,
    FusionStrategy,
    MultiModalFraudFusion,
    RiskScore,
)
from fraudlens.fusion.pattern_matcher import (
    FraudPatternMatcher,
    Match,
    Pattern,
    PatternLibrary,
)
from fraudlens.fusion.risk_scorer import (
    AnomalyDetector,
    BayesianRiskAggregator,
    RiskScoringEngine,
    TimeSeriesAnalyzer,
)
from fraudlens.fusion.validators import (
    ConsistencyReport,
    CrossModalValidator,
    ValidationResult,
)

__all__ = [
    "MultiModalFraudFusion",
    "FusionStrategy",
    "FusedResult",
    "RiskScore",
    "RiskScoringEngine",
    "BayesianRiskAggregator",
    "AnomalyDetector",
    "TimeSeriesAnalyzer",
    "CrossModalValidator",
    "ConsistencyReport",
    "ValidationResult",
    "FraudPatternMatcher",
    "PatternLibrary",
    "Pattern",
    "Match",
    "ExplanationGenerator",
    "Explanation",
    "ReportExporter",
    "AdaptiveLearner",
    "OnlineLearner",
    "ThresholdOptimizer",
]
