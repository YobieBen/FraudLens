"""
Adaptive learning system for fraud detection.

Author: Yobie Benjamin
Date: 2025
"""

import json
import pickle
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class Feedback:
    """User feedback on detection result."""
    case_id: str
    true_label: bool  # True = fraud, False = legitimate
    detection_result: Dict[str, Any]
    confidence: float
    feedback_type: str  # "manual", "automatic", "verified"
    notes: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "case_id": self.case_id,
            "true_label": self.true_label,
            "detection_result": self.detection_result,
            "confidence": self.confidence,
            "feedback_type": self.feedback_type,
            "notes": self.notes,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ThresholdConfig:
    """Threshold configuration."""
    modality: str
    fraud_threshold: float
    confidence_threshold: float
    last_updated: datetime
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "modality": self.modality,
            "fraud_threshold": self.fraud_threshold,
            "confidence_threshold": self.confidence_threshold,
            "last_updated": self.last_updated.isoformat(),
            "performance_metrics": self.performance_metrics,
        }


@dataclass
class ABTestConfig:
    """A/B test configuration."""
    test_id: str
    name: str
    description: str
    variant_a: Dict[str, Any]  # Current configuration
    variant_b: Dict[str, Any]  # Test configuration
    traffic_split: float  # Percentage for variant B
    start_date: datetime
    end_date: Optional[datetime] = None
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)  # variant -> metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "name": self.name,
            "description": self.description,
            "variant_a": self.variant_a,
            "variant_b": self.variant_b,
            "traffic_split": self.traffic_split,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "metrics": self.metrics,
        }


class AdaptiveLearner:
    """
    Adaptive learning system for continuous improvement.
    
    Learns from feedback to improve detection accuracy and
    optimize thresholds.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        feedback_buffer_size: int = 1000,
        min_feedback_for_update: int = 10,
    ):
        """
        Initialize adaptive learner.
        
        Args:
            learning_rate: Learning rate for updates
            feedback_buffer_size: Size of feedback buffer
            min_feedback_for_update: Minimum feedback required for updates
        """
        self.learning_rate = learning_rate
        self.min_feedback_for_update = min_feedback_for_update
        
        # Feedback storage
        self.feedback_buffer = deque(maxlen=feedback_buffer_size)
        self.feedback_history: List[Feedback] = []
        
        # Performance tracking
        self.performance_metrics = {
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0,
        }
        
        # Threshold configurations
        self.thresholds: Dict[str, ThresholdConfig] = {}
        self._initialize_default_thresholds()
        
        # Feature importance tracking
        self.feature_importance: Dict[str, float] = {}
        
        logger.info("AdaptiveLearner initialized")
    
    def _initialize_default_thresholds(self) -> None:
        """Initialize default threshold configurations."""
        modalities = ["text", "vision", "audio", "fusion"]
        
        for modality in modalities:
            self.thresholds[modality] = ThresholdConfig(
                modality=modality,
                fraud_threshold=0.5,
                confidence_threshold=0.7,
                last_updated=datetime.now(),
            )
    
    async def process_feedback(self, feedback: Feedback) -> None:
        """
        Process user feedback.
        
        Args:
            feedback: User feedback on detection result
        """
        # Add to buffer and history
        self.feedback_buffer.append(feedback)
        self.feedback_history.append(feedback)
        
        # Update performance metrics
        detected_fraud = feedback.detection_result.get("fraud_score", 0) > 0.5
        actual_fraud = feedback.true_label
        
        if detected_fraud and actual_fraud:
            self.performance_metrics["true_positives"] += 1
        elif detected_fraud and not actual_fraud:
            self.performance_metrics["false_positives"] += 1
        elif not detected_fraud and actual_fraud:
            self.performance_metrics["false_negatives"] += 1
        else:
            self.performance_metrics["true_negatives"] += 1
        
        # Check if we should update thresholds
        if len(self.feedback_buffer) >= self.min_feedback_for_update:
            await self.update_thresholds()
        
        logger.info(f"Processed feedback for case {feedback.case_id}")
    
    async def update_thresholds(self) -> None:
        """Update thresholds based on recent feedback."""
        if len(self.feedback_buffer) < self.min_feedback_for_update:
            return
        
        # Calculate current performance
        tp = self.performance_metrics["true_positives"]
        fp = self.performance_metrics["false_positives"]
        tn = self.performance_metrics["true_negatives"]
        fn = self.performance_metrics["false_negatives"]
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
        
        # Update thresholds based on performance
        for modality, config in self.thresholds.items():
            # Adjust fraud threshold
            if precision < 0.8:  # Too many false positives
                config.fraud_threshold = min(0.95, config.fraud_threshold + self.learning_rate)
            elif recall < 0.7:  # Too many false negatives
                config.fraud_threshold = max(0.05, config.fraud_threshold - self.learning_rate)
            
            # Update performance metrics
            config.performance_metrics = {
                "precision": precision,
                "recall": recall,
                "f1_score": 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0,
            }
            
            config.last_updated = datetime.now()
        
        logger.info(f"Updated thresholds based on {len(self.feedback_buffer)} feedback samples")
    
    def get_threshold(self, modality: str) -> ThresholdConfig:
        """Get current threshold configuration."""
        return self.thresholds.get(modality, self.thresholds.get("fusion"))
    
    async def update_feature_importance(
        self,
        features: Dict[str, float],
        outcome: bool,
    ) -> None:
        """
        Update feature importance based on outcome.
        
        Args:
            features: Feature values
            outcome: True if correctly detected
        """
        for feature, value in features.items():
            if feature not in self.feature_importance:
                self.feature_importance[feature] = 0.5
            
            # Update importance based on outcome
            if outcome:
                # Feature contributed to correct detection
                self.feature_importance[feature] += self.learning_rate * value
            else:
                # Feature contributed to incorrect detection
                self.feature_importance[feature] -= self.learning_rate * value
            
            # Clip to [0, 1]
            self.feature_importance[feature] = np.clip(self.feature_importance[feature], 0, 1)
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top n most important features."""
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_features[:n]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report."""
        tp = self.performance_metrics["true_positives"]
        fp = self.performance_metrics["false_positives"]
        tn = self.performance_metrics["true_negatives"]
        fn = self.performance_metrics["false_negatives"]
        
        total = tp + fp + tn + fn
        
        if total == 0:
            return {
                "total_feedback": 0,
                "metrics": {},
            }
        
        return {
            "total_feedback": len(self.feedback_history),
            "recent_feedback": len(self.feedback_buffer),
            "metrics": {
                "accuracy": (tp + tn) / total,
                "precision": tp / (tp + fp) if tp + fp > 0 else 0,
                "recall": tp / (tp + fn) if tp + fn > 0 else 0,
                "f1_score": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0,
                "false_positive_rate": fp / (fp + tn) if fp + tn > 0 else 0,
            },
            "confusion_matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
            },
            "thresholds": {k: v.to_dict() for k, v in self.thresholds.items()},
            "top_features": self.get_top_features(5),
        }


class OnlineLearner:
    """
    Online learning for real-time model updates.
    """
    
    def __init__(
        self,
        model_type: str = "sgd",
        learning_rate: float = 0.001,
        batch_size: int = 32,
    ):
        """
        Initialize online learner.
        
        Args:
            model_type: Type of online learning model
            learning_rate: Learning rate
            batch_size: Batch size for updates
        """
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Initialize model
        if model_type == "sgd":
            self.model = SGDClassifier(
                loss="log_loss",
                learning_rate="constant",
                eta0=learning_rate,
                random_state=42,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Batch storage
        self.batch_X = []
        self.batch_y = []
        
        logger.info(f"OnlineLearner initialized with {model_type} model")
    
    async def partial_fit(
        self,
        features: Dict[str, float],
        label: bool,
    ) -> None:
        """
        Partially fit the model with new data.
        
        Args:
            features: Feature dictionary
            label: True label (fraud/legitimate)
        """
        # Convert features to array
        X = self._dict_to_array(features)
        y = 1 if label else 0
        
        # Add to batch
        self.batch_X.append(X)
        self.batch_y.append(y)
        
        # Update when batch is full
        if len(self.batch_X) >= self.batch_size:
            await self._update_model()
    
    async def _update_model(self) -> None:
        """Update model with batch."""
        if not self.batch_X:
            return
        
        X = np.array(self.batch_X)
        y = np.array(self.batch_y)
        
        if not self.is_fitted:
            # Initial fit
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.model.fit(X_scaled, y)
            self.is_fitted = True
        else:
            # Partial fit
            X_scaled = self.scaler.transform(X)
            self.model.partial_fit(X_scaled, y, classes=[0, 1])
        
        # Clear batch
        self.batch_X = []
        self.batch_y = []
        
        logger.debug(f"Model updated with batch of {len(y)} samples")
    
    async def predict(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """
        Predict fraud probability.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Tuple of (is_fraud, confidence)
        """
        if not self.is_fitted:
            return False, 0.5
        
        X = self._dict_to_array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        confidence = self.model.predict_proba(X_scaled)[0].max()
        
        return bool(prediction), float(confidence)
    
    def _dict_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to array."""
        return np.array([v for v in features.values()])
    
    def save_model(self, path: Path) -> None:
        """Save model to file."""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "is_fitted": self.is_fitted,
        }
        
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path) -> None:
        """Load model from file."""
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.is_fitted = model_data["is_fitted"]
        
        logger.info(f"Model loaded from {path}")


class ThresholdOptimizer:
    """
    Optimizes detection thresholds based on business objectives.
    """
    
    def __init__(
        self,
        false_positive_cost: float = 1.0,
        false_negative_cost: float = 10.0,
    ):
        """
        Initialize threshold optimizer.
        
        Args:
            false_positive_cost: Cost of false positive
            false_negative_cost: Cost of false negative
        """
        self.fp_cost = false_positive_cost
        self.fn_cost = false_negative_cost
        
        # Historical scores and labels
        self.scores_history = []
        self.labels_history = []
        
        logger.info("ThresholdOptimizer initialized")
    
    def add_result(self, score: float, true_label: bool) -> None:
        """Add detection result."""
        self.scores_history.append(score)
        self.labels_history.append(true_label)
    
    def find_optimal_threshold(
        self,
        metric: str = "cost",
        min_precision: Optional[float] = None,
        min_recall: Optional[float] = None,
    ) -> float:
        """
        Find optimal threshold.
        
        Args:
            metric: Optimization metric ("cost", "f1", "balanced")
            min_precision: Minimum precision constraint
            min_recall: Minimum recall constraint
            
        Returns:
            Optimal threshold
        """
        if not self.scores_history:
            return 0.5
        
        scores = np.array(self.scores_history)
        labels = np.array(self.labels_history)
        
        # Try different thresholds
        thresholds = np.linspace(0, 1, 101)
        best_threshold = 0.5
        best_value = float("inf") if metric == "cost" else 0
        
        for threshold in thresholds:
            predictions = scores > threshold
            
            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            tn = np.sum((predictions == 0) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))
            
            # Check constraints
            if tp + fp > 0:
                precision = tp / (tp + fp)
                if min_precision and precision < min_precision:
                    continue
            
            if tp + fn > 0:
                recall = tp / (tp + fn)
                if min_recall and recall < min_recall:
                    continue
            
            # Calculate metric
            if metric == "cost":
                cost = fp * self.fp_cost + fn * self.fn_cost
                if cost < best_value:
                    best_value = cost
                    best_threshold = threshold
            elif metric == "f1":
                if tp > 0:
                    precision = tp / (tp + fp) if tp + fp > 0 else 0
                    recall = tp / (tp + fn) if tp + fn > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
                    if f1 > best_value:
                        best_value = f1
                        best_threshold = threshold
            elif metric == "balanced":
                # Balance between precision and recall
                if tp + fp > 0 and tp + fn > 0:
                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    balanced = 2 * precision * recall / (precision + recall)
                    if balanced > best_value:
                        best_value = balanced
                        best_threshold = threshold
        
        return float(best_threshold)
    
    def get_threshold_analysis(self) -> Dict[str, Any]:
        """Get analysis of different thresholds."""
        if not self.scores_history:
            return {}
        
        scores = np.array(self.scores_history)
        labels = np.array(self.labels_history)
        
        analysis = {
            "thresholds": [],
            "metrics": [],
        }
        
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            predictions = scores > threshold
            
            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            tn = np.sum((predictions == 0) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))
            
            metrics = {
                "threshold": threshold,
                "precision": tp / (tp + fp) if tp + fp > 0 else 0,
                "recall": tp / (tp + fn) if tp + fn > 0 else 0,
                "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0,
                "cost": fp * self.fp_cost + fn * self.fn_cost,
            }
            
            analysis["thresholds"].append(threshold)
            analysis["metrics"].append(metrics)
        
        return analysis


class ABTestFramework:
    """
    A/B testing framework for model updates.
    """
    
    def __init__(self):
        """Initialize A/B testing framework."""
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.completed_tests: List[ABTestConfig] = []
        
        logger.info("ABTestFramework initialized")
    
    def create_test(
        self,
        name: str,
        description: str,
        variant_a: Dict[str, Any],
        variant_b: Dict[str, Any],
        traffic_split: float = 0.5,
        duration_days: int = 7,
    ) -> ABTestConfig:
        """
        Create new A/B test.
        
        Args:
            name: Test name
            description: Test description
            variant_a: Control configuration
            variant_b: Test configuration
            traffic_split: Traffic percentage for variant B
            duration_days: Test duration in days
            
        Returns:
            A/B test configuration
        """
        test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        test = ABTestConfig(
            test_id=test_id,
            name=name,
            description=description,
            variant_a=variant_a,
            variant_b=variant_b,
            traffic_split=traffic_split,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=duration_days),
        )
        
        self.active_tests[test_id] = test
        logger.info(f"Created A/B test: {name} ({test_id})")
        
        return test
    
    def assign_variant(self, test_id: str) -> str:
        """
        Assign variant for request.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Variant assignment ("A" or "B")
        """
        test = self.active_tests.get(test_id)
        if not test:
            return "A"
        
        # Check if test is still active
        if test.end_date and datetime.now() > test.end_date:
            self._complete_test(test_id)
            return "A"
        
        # Random assignment based on traffic split
        return "B" if np.random.random() < test.traffic_split else "A"
    
    def record_metric(
        self,
        test_id: str,
        variant: str,
        metric_name: str,
        value: float,
    ) -> None:
        """
        Record metric for variant.
        
        Args:
            test_id: Test identifier
            variant: Variant ("A" or "B")
            metric_name: Metric name
            value: Metric value
        """
        test = self.active_tests.get(test_id)
        if not test:
            return
        
        if variant not in test.metrics:
            test.metrics[variant] = {}
        
        if metric_name not in test.metrics[variant]:
            test.metrics[variant][metric_name] = []
        
        test.metrics[variant][metric_name].append(value)
    
    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get test results."""
        test = self.active_tests.get(test_id)
        if not test:
            # Check completed tests
            test = next((t for t in self.completed_tests if t.test_id == test_id), None)
            if not test:
                return {}
        
        results = {
            "test_id": test_id,
            "name": test.name,
            "status": "active" if test_id in self.active_tests else "completed",
            "variants": {},
        }
        
        for variant in ["A", "B"]:
            if variant in test.metrics:
                variant_metrics = {}
                for metric_name, values in test.metrics[variant].items():
                    if values:
                        variant_metrics[metric_name] = {
                            "mean": np.mean(values),
                            "std": np.std(values),
                            "count": len(values),
                        }
                results["variants"][variant] = variant_metrics
        
        # Statistical significance
        if "A" in test.metrics and "B" in test.metrics:
            results["significance"] = self._calculate_significance(test)
        
        return results
    
    def _complete_test(self, test_id: str) -> None:
        """Complete and archive test."""
        if test_id in self.active_tests:
            test = self.active_tests.pop(test_id)
            test.end_date = datetime.now()
            self.completed_tests.append(test)
            logger.info(f"Completed A/B test: {test.name}")
    
    def _calculate_significance(self, test: ABTestConfig) -> Dict[str, Any]:
        """Calculate statistical significance."""
        # Simple t-test for primary metric
        primary_metric = "accuracy"  # Or configurable
        
        if primary_metric in test.metrics.get("A", {}) and primary_metric in test.metrics.get("B", {}):
            from scipy import stats
            
            a_values = test.metrics["A"][primary_metric]
            b_values = test.metrics["B"][primary_metric]
            
            if len(a_values) > 1 and len(b_values) > 1:
                t_stat, p_value = stats.ttest_ind(a_values, b_values)
                
                return {
                    "metric": primary_metric,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                    "winner": "B" if np.mean(b_values) > np.mean(a_values) else "A",
                }
        
        return {}