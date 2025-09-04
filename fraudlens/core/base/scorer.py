"""
Base risk scorer for aggregating fraud detection results.

Author: Yobie Benjamin
Date: 2025-08-26 18:34:00 PDT
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class RiskLevel(Enum):
    """Risk level categories."""

    VERY_LOW = "very_low"  # 0.0 - 0.2
    LOW = "low"  # 0.2 - 0.4
    MEDIUM = "medium"  # 0.4 - 0.6
    HIGH = "high"  # 0.6 - 0.8
    VERY_HIGH = "very_high"  # 0.8 - 1.0

    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        """Get risk level from numerical score."""
        if score < 0.2:
            return cls.VERY_LOW
        elif score < 0.4:
            return cls.LOW
        elif score < 0.6:
            return cls.MEDIUM
        elif score < 0.8:
            return cls.HIGH
        else:
            return cls.VERY_HIGH


@dataclass
class RiskAssessment:
    """Comprehensive risk assessment result."""

    overall_score: float  # 0.0 to 1.0
    risk_level: RiskLevel
    confidence: float  # 0.0 to 1.0
    contributing_factors: List[Dict[str, Any]]
    recommendations: List[str]
    timestamp: datetime
    assessment_id: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert assessment to dictionary."""
        return {
            "overall_score": self.overall_score,
            "risk_level": self.risk_level.value,
            "confidence": self.confidence,
            "contributing_factors": self.contributing_factors,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
            "assessment_id": self.assessment_id,
            "metadata": self.metadata or {},
        }

    def get_top_risks(self, n: int = 3) -> List[Dict[str, Any]]:
        """Get top n contributing risk factors."""
        sorted_factors = sorted(
            self.contributing_factors,
            key=lambda x: x.get("weight", 0) * x.get("score", 0),
            reverse=True,
        )
        return sorted_factors[:n]


class RiskScorer(ABC):
    """Abstract base class for risk scoring and aggregation."""

    def __init__(
        self,
        scorer_id: str,
        config: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize risk scorer.

        Args:
            scorer_id: Unique identifier for this scorer
            config: Configuration dictionary
            weights: Weights for different risk factors
        """
        self.scorer_id = scorer_id
        self.config = config or {}
        self.weights = weights or {}
        self._history: List[RiskAssessment] = []
        self._thresholds = {
            "very_low": 0.2,
            "low": 0.4,
            "medium": 0.6,
            "high": 0.8,
        }

    @abstractmethod
    async def score(self, detection_results: List[Any], **kwargs) -> RiskAssessment:
        """
        Calculate risk score from detection results.

        Args:
            detection_results: List of detection results from various detectors
            **kwargs: Additional scoring parameters

        Returns:
            RiskAssessment with overall risk evaluation
        """
        pass

    @abstractmethod
    def aggregate_scores(self, scores: List[float], weights: Optional[List[float]] = None) -> float:
        """
        Aggregate multiple scores into a single score.

        Args:
            scores: List of individual scores
            weights: Optional weights for each score

        Returns:
            Aggregated score between 0.0 and 1.0
        """
        pass

    @abstractmethod
    def calculate_confidence(self, detection_results: List[Any]) -> float:
        """
        Calculate confidence level for the risk assessment.

        Args:
            detection_results: Detection results to analyze

        Returns:
            Confidence score between 0.0 and 1.0
        """
        pass

    @abstractmethod
    def generate_recommendations(self, risk_assessment: RiskAssessment) -> List[str]:
        """
        Generate actionable recommendations based on risk assessment.

        Args:
            risk_assessment: Current risk assessment

        Returns:
            List of recommendation strings
        """
        pass

    def set_thresholds(self, thresholds: Dict[str, float]) -> None:
        """
        Set custom risk level thresholds.

        Args:
            thresholds: Dictionary mapping risk levels to threshold values
        """
        self._thresholds.update(thresholds)

    def get_statistics(self) -> Dict[str, Any]:
        """Get scoring statistics from history."""
        if not self._history:
            return {"total_assessments": 0}

        scores = [a.overall_score for a in self._history]
        return {
            "total_assessments": len(self._history),
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores),
            "risk_distribution": self._get_risk_distribution(),
        }

    def _get_risk_distribution(self) -> Dict[str, int]:
        """Get distribution of risk levels in history."""
        distribution = {level.value: 0 for level in RiskLevel}
        for assessment in self._history:
            distribution[assessment.risk_level.value] += 1
        return distribution

    def add_to_history(self, assessment: RiskAssessment) -> None:
        """Add assessment to history for statistics."""
        self._history.append(assessment)
        # Keep only last 1000 assessments
        if len(self._history) > 1000:
            self._history = self._history[-1000:]

    def clear_history(self) -> None:
        """Clear assessment history."""
        self._history.clear()

    def calibrate(
        self, true_labels: List[bool], predicted_scores: List[float]
    ) -> Tuple[float, float]:
        """
        Calibrate scorer using true labels.

        Args:
            true_labels: Ground truth fraud labels
            predicted_scores: Predicted risk scores

        Returns:
            Tuple of (optimal_threshold, accuracy)
        """
        best_threshold = 0.5
        best_accuracy = 0.0

        for threshold in np.linspace(0.1, 0.9, 17):
            predictions = [score >= threshold for score in predicted_scores]
            accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(true_labels)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        return best_threshold, best_accuracy

    def __repr__(self) -> str:
        """String representation of scorer."""
        return (
            f"{self.__class__.__name__}("
            f"id='{self.scorer_id}', "
            f"history_size={len(self._history)})"
        )
