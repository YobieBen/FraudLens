"""
Risk scoring engine with multiple algorithms.

Author: Yobie Benjamin
Date: 2025
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class RiskProfile:
    """Risk profile for an entity."""
    entity_id: str
    risk_score: float
    risk_level: str
    anomaly_score: float
    trend: str
    confidence_interval: Tuple[float, float]
    factors: List[Dict[str, Any]]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity_id": self.entity_id,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "anomaly_score": self.anomaly_score,
            "trend": self.trend,
            "confidence_interval": self.confidence_interval,
            "factors": self.factors,
            "timestamp": self.timestamp.isoformat(),
        }


class RiskScoringEngine:
    """
    Comprehensive risk scoring engine.
    
    Combines multiple algorithms to generate robust risk scores.
    """
    
    def __init__(
        self,
        enable_bayesian: bool = True,
        enable_anomaly: bool = True,
        enable_time_series: bool = True,
        enable_graph: bool = True,
    ):
        """
        Initialize risk scoring engine.
        
        Args:
            enable_bayesian: Enable Bayesian risk aggregation
            enable_anomaly: Enable anomaly detection
            enable_time_series: Enable time series analysis
            enable_graph: Enable graph-based analysis
        """
        self.enable_bayesian = enable_bayesian
        self.enable_anomaly = enable_anomaly
        self.enable_time_series = enable_time_series
        self.enable_graph = enable_graph
        
        # Initialize components
        self.bayesian = BayesianRiskAggregator() if enable_bayesian else None
        self.anomaly_detector = AnomalyDetector() if enable_anomaly else None
        self.time_series = TimeSeriesAnalyzer() if enable_time_series else None
        self.graph_analyzer = GraphAnalyzer() if enable_graph else None
        
        # Risk thresholds
        self.thresholds = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "critical": 0.95,
        }
        
        logger.info("RiskScoringEngine initialized")
    
    async def calculate_risk_score(
        self,
        fraud_score: float,
        confidence: float,
        modality_scores: Dict[str, float],
        historical_data: Optional[List[float]] = None,
        entity_features: Optional[Dict[str, Any]] = None,
        relationships: Optional[Dict[str, List[str]]] = None,
    ) -> RiskProfile:
        """
        Calculate comprehensive risk score.
        
        Args:
            fraud_score: Base fraud score
            confidence: Confidence level
            modality_scores: Scores from different modalities
            historical_data: Historical fraud scores
            entity_features: Entity-specific features
            relationships: Entity relationships for graph analysis
            
        Returns:
            Comprehensive risk profile
        """
        start_time = time.time()
        
        risk_scores = []
        risk_factors = []
        
        # Base score
        risk_scores.append(fraud_score)
        risk_factors.append({
            "factor": "base_fraud_score",
            "value": fraud_score,
            "weight": 0.3,
        })
        
        # Bayesian aggregation
        if self.bayesian and modality_scores:
            bayesian_score = await self.bayesian.aggregate(modality_scores, confidence)
            risk_scores.append(bayesian_score)
            risk_factors.append({
                "factor": "bayesian_aggregation",
                "value": bayesian_score,
                "weight": 0.2,
            })
        
        # Anomaly detection
        anomaly_score = 0.0
        if self.anomaly_detector and entity_features:
            anomaly_score = await self.anomaly_detector.detect(entity_features)
            risk_scores.append(anomaly_score)
            risk_factors.append({
                "factor": "anomaly_detection",
                "value": anomaly_score,
                "weight": 0.2,
            })
        
        # Time series analysis
        trend = "stable"
        if self.time_series and historical_data:
            trend_score, trend = await self.time_series.analyze(historical_data)
            risk_scores.append(trend_score)
            risk_factors.append({
                "factor": "temporal_pattern",
                "value": trend_score,
                "weight": 0.15,
                "trend": trend,
            })
        
        # Graph-based analysis
        if self.graph_analyzer and relationships:
            graph_score = await self.graph_analyzer.analyze(relationships)
            risk_scores.append(graph_score)
            risk_factors.append({
                "factor": "relationship_risk",
                "value": graph_score,
                "weight": 0.15,
            })
        
        # Calculate weighted average
        if risk_factors:
            total_weight = sum(f["weight"] for f in risk_factors)
            weighted_score = sum(f["value"] * f["weight"] for f in risk_factors) / total_weight
        else:
            weighted_score = fraud_score
        
        # Calculate confidence interval
        if len(risk_scores) > 1:
            ci_lower, ci_upper = stats.t.interval(
                confidence=0.95,
                df=len(risk_scores) - 1,
                loc=np.mean(risk_scores),
                scale=stats.sem(risk_scores),
            )
            confidence_interval = (max(0, ci_lower), min(1, ci_upper))
        else:
            confidence_interval = (weighted_score * 0.9, min(1, weighted_score * 1.1))
        
        # Determine risk level
        risk_level = self._get_risk_level(weighted_score)
        
        # Processing time
        processing_time = (time.time() - start_time) * 1000
        
        return RiskProfile(
            entity_id="",  # Will be set by caller
            risk_score=float(np.clip(weighted_score, 0, 1)),
            risk_level=risk_level,
            anomaly_score=anomaly_score,
            trend=trend,
            confidence_interval=confidence_interval,
            factors=risk_factors,
            timestamp=datetime.now(),
        )
    
    def _get_risk_level(self, score: float) -> str:
        """Determine risk level from score."""
        if score >= self.thresholds["critical"]:
            return "critical"
        elif score >= self.thresholds["high"]:
            return "high"
        elif score >= self.thresholds["medium"]:
            return "medium"
        else:
            return "low"
    
    def update_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Update risk thresholds."""
        self.thresholds.update(thresholds)
        logger.info(f"Updated risk thresholds: {self.thresholds}")


class BayesianRiskAggregator:
    """Bayesian approach to risk aggregation."""
    
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """
        Initialize Bayesian aggregator.
        
        Args:
            prior_alpha: Alpha parameter for Beta prior
            prior_beta: Beta parameter for Beta prior
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        
    async def aggregate(
        self,
        scores: Dict[str, float],
        confidence: float = 0.5,
    ) -> float:
        """
        Aggregate scores using Bayesian approach.
        
        Args:
            scores: Dictionary of modality scores
            confidence: Overall confidence level
            
        Returns:
            Aggregated risk score
        """
        if not scores:
            return 0.0
        
        # Convert scores to successes and failures
        successes = sum(s for s in scores.values())
        failures = len(scores) - successes
        
        # Update posterior
        posterior_alpha = self.prior_alpha + successes
        posterior_beta = self.prior_beta + failures
        
        # Calculate posterior mean
        posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
        
        # Adjust by confidence
        adjusted_score = posterior_mean * confidence + (1 - confidence) * 0.5
        
        return float(np.clip(adjusted_score, 0, 1))


class AnomalyDetector:
    """Anomaly detection using Isolation Forest."""
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: int = 256,
    ):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers
            n_estimators: Number of base estimators
            max_samples: Number of samples to draw
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=42,
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    async def detect(self, features: Dict[str, Any]) -> float:
        """
        Detect anomalies in features.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Anomaly score (0-1, higher is more anomalous)
        """
        # Convert features to array
        feature_vector = self._dict_to_array(features)
        
        if not self.is_fitted:
            # Fit with the first sample as normal baseline
            self.scaler.fit([feature_vector])
            scaled = self.scaler.transform([feature_vector])
            
            # Generate synthetic normal data around this point
            np.random.seed(42)
            n_features = len(feature_vector)
            normal_data = np.random.randn(100, n_features) * 0.1 + scaled[0]
            
            # Fit model
            self.model.fit(normal_data)
            self.is_fitted = True
        
        # Scale features
        scaled = self.scaler.transform([feature_vector])
        
        # Get anomaly score
        decision_score = self.model.decision_function(scaled)[0]
        anomaly_score = self.model.score_samples(scaled)[0]
        
        # Convert to 0-1 range (more negative = more anomalous)
        normalized_score = 1.0 / (1.0 + np.exp(anomaly_score))
        
        return float(normalized_score)
    
    def _dict_to_array(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dictionary to array."""
        vector = []
        
        for key, value in sorted(features.items()):
            if isinstance(value, (int, float)):
                vector.append(value)
            elif isinstance(value, bool):
                vector.append(1.0 if value else 0.0)
            elif isinstance(value, str):
                # Simple hash encoding for strings
                vector.append(hash(value) % 1000 / 1000.0)
            elif isinstance(value, (list, tuple)):
                vector.append(len(value))
        
        return np.array(vector)
    
    def _fit_default(self):
        """Fit with default normal data."""
        # Don't fit with default data - wait for actual data
        self.is_fitted = False
    
    def fit(self, training_data: List[Dict[str, Any]]) -> None:
        """
        Fit the anomaly detector with training data.
        
        Args:
            training_data: List of feature dictionaries
        """
        if not training_data:
            return
        
        # Convert to array
        X = np.array([self._dict_to_array(d) for d in training_data])
        
        # Fit scaler and model
        self.scaler.fit(X)
        scaled_X = self.scaler.transform(X)
        self.model.fit(scaled_X)
        self.is_fitted = True
        
        logger.info(f"Anomaly detector fitted with {len(training_data)} samples")


class TimeSeriesAnalyzer:
    """Time series analysis for behavioral patterns."""
    
    def __init__(self, window_size: int = 10, trend_threshold: float = 0.09):
        """
        Initialize time series analyzer.
        
        Args:
            window_size: Size of sliding window
            trend_threshold: Threshold for trend detection
        """
        self.window_size = window_size
        self.trend_threshold = trend_threshold
        
        # Historical data storage
        self.history = deque(maxlen=100)
        
    async def analyze(self, data: List[float]) -> Tuple[float, str]:
        """
        Analyze time series data.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (risk_score, trend)
        """
        if not data or len(data) < 2:
            return 0.5, "insufficient_data"
        
        # Update history
        self.history.extend(data)
        
        # Calculate basic statistics
        recent = data[-self.window_size:] if len(data) >= self.window_size else data
        mean = np.mean(recent)
        std = np.std(recent)
        
        # Detect trend
        trend = self._detect_trend(data)
        
        # Calculate anomalies
        z_scores = np.abs(stats.zscore(recent))
        anomaly_count = np.sum(z_scores > 2)
        
        # Risk score based on mean, volatility, and anomalies
        risk_score = mean * 0.5 + std * 0.3 + (anomaly_count / len(recent)) * 0.2
        
        # Adjust for trend
        if trend == "increasing":
            risk_score *= 1.2
        elif trend == "decreasing":
            risk_score *= 0.8
        
        return float(np.clip(risk_score, 0, 1)), trend
    
    def _detect_trend(self, data: List[float]) -> str:
        """Detect trend in time series."""
        if len(data) < 3:
            return "stable"
        
        # Simple linear regression
        x = np.arange(len(data))
        slope, _ = np.polyfit(x, data, 1)
        
        if slope >= self.trend_threshold:
            return "increasing"
        elif slope <= -self.trend_threshold:
            return "decreasing"
        else:
            return "stable"
    
    def add_observation(self, value: float) -> None:
        """Add new observation to history."""
        self.history.append(value)


class GraphAnalyzer:
    """Graph-based analysis for relationship mapping."""
    
    def __init__(self, suspicious_threshold: int = 3):
        """
        Initialize graph analyzer.
        
        Args:
            suspicious_threshold: Number of suspicious connections threshold
        """
        self.suspicious_threshold = suspicious_threshold
        self.known_suspicious = set()
        
    async def analyze(self, relationships: Dict[str, List[str]]) -> float:
        """
        Analyze entity relationships.
        
        Args:
            relationships: Dictionary of entity relationships
            
        Returns:
            Risk score based on relationships
        """
        if not relationships:
            return 0.0
        
        risk_score = 0.0
        total_connections = 0
        suspicious_connections = 0
        
        for entity, connections in relationships.items():
            total_connections += len(connections)
            
            # Check for suspicious connections
            for conn in connections:
                if conn in self.known_suspicious:
                    suspicious_connections += 1
                
            # Check for unusual patterns
            if len(connections) > 10:  # Unusually many connections
                risk_score += 0.1
            
            # Check for isolated clusters
            if len(connections) == 1:  # Only one connection
                risk_score += 0.05
        
        # Calculate risk based on suspicious connections
        if total_connections > 0:
            suspicious_ratio = suspicious_connections / total_connections
            risk_score += suspicious_ratio * 0.5
        
        # Check for circular relationships
        circular = self._detect_circular(relationships)
        if circular:
            risk_score += 0.2
        
        return float(np.clip(risk_score, 0, 1))
    
    def _detect_circular(self, relationships: Dict[str, List[str]]) -> bool:
        """Detect circular relationships."""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            if node in relationships:
                for neighbor in relationships[node]:
                    if neighbor not in visited:
                        if has_cycle(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        return True
            
            rec_stack.remove(node)
            return False
        
        for entity in relationships:
            if entity not in visited:
                if has_cycle(entity):
                    return True
        
        return False
    
    def add_suspicious_entity(self, entity: str) -> None:
        """Mark entity as suspicious."""
        self.known_suspicious.add(entity)
        logger.info(f"Added suspicious entity: {entity}")