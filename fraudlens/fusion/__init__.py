"""
Multi-modal fusion system for FraudLens.

This module provides advanced fusion techniques for combining insights
from text, vision, audio, and video modalities to generate unified fraud risk scores.

Author: Yobie Benjamin
Date: 2025
"""

from fraudlens.fusion.fusion_engine import (
    MultiModalFraudFusion,
    FusionStrategy,
    FusedResult,
    RiskScore,
)
from fraudlens.fusion.risk_scorer import (
    RiskScoringEngine,
    BayesianRiskAggregator,
    AnomalyDetector,
    TimeSeriesAnalyzer,
)
from fraudlens.fusion.validators import (
    CrossModalValidator,
    ConsistencyReport,
    ValidationResult,
)
from fraudlens.fusion.pattern_matcher import (
    FraudPatternMatcher,
    PatternLibrary,
    Pattern,
    Match,
)
from fraudlens.fusion.explainer import (
    ExplanationGenerator,
    Explanation,
    ReportExporter,
)
from fraudlens.fusion.adaptive_learning import (
    AdaptiveLearner,
    OnlineLearner,
    ThresholdOptimizer,
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
