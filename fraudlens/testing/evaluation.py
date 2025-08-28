"""
Evaluation suite for FraudLens with fraud detection metrics.

Author: Yobie Benjamin
Date: 2025
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, cross_val_score

from fraudlens.core.base.detector import DetectionResult, FraudType


@dataclass
class EvaluationMetrics:
    """Fraud detection evaluation metrics."""
    
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    auc_roc: float
    auc_pr: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]
    threshold: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def specificity(self) -> float:
        """Calculate specificity (true negative rate)."""
        if self.true_negatives + self.false_positives == 0:
            return 0.0
        return self.true_negatives / (self.true_negatives + self.false_positives)
    
    @property
    def fpr(self) -> float:
        """Calculate false positive rate."""
        return 1 - self.specificity
    
    @property
    def fnr(self) -> float:
        """Calculate false negative rate."""
        if self.false_negatives + self.true_positives == 0:
            return 0.0
        return self.false_negatives / (self.false_negatives + self.true_positives)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
            "specificity": self.specificity,
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "fpr": self.fpr,
            "fnr": self.fnr,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FalsePositiveAnalysis:
    """False positive analysis results."""
    
    total_false_positives: int
    fp_by_type: Dict[str, int]
    fp_patterns: List[str]
    fp_confidence_distribution: Dict[str, float]
    common_fp_features: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_false_positives": self.total_false_positives,
            "fp_by_type": self.fp_by_type,
            "fp_patterns": self.fp_patterns,
            "fp_confidence_distribution": self.fp_confidence_distribution,
            "common_fp_features": self.common_fp_features,
            "recommendations": self.recommendations,
        }


@dataclass
class PerformanceBenchmark:
    """Performance benchmark results."""
    
    avg_processing_time_ms: float
    min_processing_time_ms: float
    max_processing_time_ms: float
    p50_processing_time_ms: float
    p95_processing_time_ms: float
    p99_processing_time_ms: float
    throughput_samples_per_sec: float
    cpu_utilization_percent: float
    memory_usage_mb: float
    gpu_utilization_percent: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "avg_processing_time_ms": self.avg_processing_time_ms,
            "min_processing_time_ms": self.min_processing_time_ms,
            "max_processing_time_ms": self.max_processing_time_ms,
            "p50_processing_time_ms": self.p50_processing_time_ms,
            "p95_processing_time_ms": self.p95_processing_time_ms,
            "p99_processing_time_ms": self.p99_processing_time_ms,
            "throughput_samples_per_sec": self.throughput_samples_per_sec,
            "cpu_utilization_percent": self.cpu_utilization_percent,
            "memory_usage_mb": self.memory_usage_mb,
            "gpu_utilization_percent": self.gpu_utilization_percent,
        }


class FraudDetectionEvaluator:
    """Comprehensive evaluation suite for fraud detection."""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """
        Initialize evaluator.
        
        Args:
            output_dir: Directory for saving results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.metrics_history = []
        self.benchmark_history = []
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None,
        threshold: float = 0.5,
        class_names: Optional[List[str]] = None,
    ) -> EvaluationMetrics:
        """
        Calculate comprehensive fraud detection metrics.
        
        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            y_scores: Prediction scores/probabilities
            threshold: Classification threshold
            class_names: Names for each class
            
        Returns:
            Evaluation metrics
        """
        # Convert predictions based on threshold if scores provided
        if y_scores is not None:
            y_pred = (y_scores >= threshold).astype(int)
        
        # Basic metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # AUC scores
        auc_roc = 0.0
        auc_pr = 0.0
        
        if y_scores is not None and len(np.unique(y_true)) > 1:
            try:
                auc_roc = roc_auc_score(y_true, y_scores)
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
                auc_pr = auc(recall_curve, precision_curve)
            except:
                pass
        
        # Classification report
        if class_names is None:
            class_names = ["Normal", "Fraud"]
        
        clf_report = classification_report(
            y_true,
            y_pred,
            target_names=class_names[:len(np.unique(y_true))],
            output_dict=True,
            zero_division=0,
        )
        
        metrics = EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            confusion_matrix=cm,
            classification_report=clf_report,
            threshold=threshold,
        )
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def analyze_false_positives(
        self,
        false_positive_samples: List[Dict[str, Any]],
        detection_results: List[DetectionResult],
    ) -> FalsePositiveAnalysis:
        """
        Analyze false positive cases.
        
        Args:
            false_positive_samples: List of FP samples with features
            detection_results: Detection results for FP samples
            
        Returns:
            False positive analysis
        """
        fp_by_type = defaultdict(int)
        fp_patterns = []
        confidence_bins = {
            "0.5-0.6": 0,
            "0.6-0.7": 0,
            "0.7-0.8": 0,
            "0.8-0.9": 0,
            "0.9-1.0": 0,
        }
        common_features = defaultdict(int)
        
        for sample, result in zip(false_positive_samples, detection_results):
            # Count by fraud type
            for fraud_type in result.fraud_types:
                fp_by_type[fraud_type.value] += 1
            
            # Analyze confidence distribution
            score = result.fraud_score
            if 0.5 <= score < 0.6:
                confidence_bins["0.5-0.6"] += 1
            elif 0.6 <= score < 0.7:
                confidence_bins["0.6-0.7"] += 1
            elif 0.7 <= score < 0.8:
                confidence_bins["0.7-0.8"] += 1
            elif 0.8 <= score < 0.9:
                confidence_bins["0.8-0.9"] += 1
            elif 0.9 <= score <= 1.0:
                confidence_bins["0.9-1.0"] += 1
            
            # Extract common features
            if "features" in sample:
                for feature, value in sample["features"].items():
                    if value:  # Feature is present
                        common_features[feature] += 1
            
            # Identify patterns
            if result.explanation:
                fp_patterns.append(result.explanation[:100])
        
        # Sort and get top features
        top_features = sorted(
            common_features.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Generate recommendations
        recommendations = []
        
        # High confidence FPs
        high_conf_fps = confidence_bins.get("0.9-1.0", 0)
        if high_conf_fps > len(false_positive_samples) * 0.1:
            recommendations.append(
                "High confidence false positives detected. "
                "Consider adjusting detection thresholds or retraining models."
            )
        
        # Specific fraud type FPs
        for fraud_type, count in fp_by_type.items():
            if count > len(false_positive_samples) * 0.3:
                recommendations.append(
                    f"High false positive rate for {fraud_type}. "
                    f"Review detection rules for this fraud type."
                )
        
        # Common feature patterns
        if top_features and top_features[0][1] > len(false_positive_samples) * 0.5:
            recommendations.append(
                f"Feature '{top_features[0][0]}' appears in {top_features[0][1]} FPs. "
                "Consider adjusting feature weights."
            )
        
        if not recommendations:
            recommendations.append("False positive rate is within acceptable range.")
        
        analysis = FalsePositiveAnalysis(
            total_false_positives=len(false_positive_samples),
            fp_by_type=dict(fp_by_type),
            fp_patterns=fp_patterns[:10],  # Top 10 patterns
            fp_confidence_distribution=confidence_bins,
            common_fp_features=[f[0] for f in top_features],
            recommendations=recommendations,
        )
        
        return analysis
    
    def benchmark_processing_speed(
        self,
        test_samples: List[Any],
        process_func,
        warmup_runs: int = 10,
        benchmark_runs: int = 100,
    ) -> PerformanceBenchmark:
        """
        Benchmark processing speed.
        
        Args:
            test_samples: Test samples to process
            process_func: Function to process samples
            warmup_runs: Number of warmup runs
            benchmark_runs: Number of benchmark runs
            
        Returns:
            Performance benchmark results
        """
        import psutil
        
        process = psutil.Process()
        
        # Warmup
        print(f"Running {warmup_runs} warmup iterations...")
        for _ in range(warmup_runs):
            for sample in test_samples[:min(10, len(test_samples))]:
                _ = process_func(sample)
        
        # Benchmark
        print(f"Running {benchmark_runs} benchmark iterations...")
        processing_times = []
        cpu_usage = []
        memory_usage = []
        
        for i in range(benchmark_runs):
            sample = test_samples[i % len(test_samples)]
            
            # Measure processing time
            start_time = time.perf_counter()
            _ = process_func(sample)
            processing_time = (time.perf_counter() - start_time) * 1000  # ms
            
            processing_times.append(processing_time)
            
            # Measure resource usage
            cpu_usage.append(process.cpu_percent())
            memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
        
        # Calculate statistics
        processing_times_sorted = sorted(processing_times)
        
        benchmark = PerformanceBenchmark(
            avg_processing_time_ms=np.mean(processing_times),
            min_processing_time_ms=np.min(processing_times),
            max_processing_time_ms=np.max(processing_times),
            p50_processing_time_ms=np.percentile(processing_times_sorted, 50),
            p95_processing_time_ms=np.percentile(processing_times_sorted, 95),
            p99_processing_time_ms=np.percentile(processing_times_sorted, 99),
            throughput_samples_per_sec=1000 / np.mean(processing_times),
            cpu_utilization_percent=np.mean(cpu_usage),
            memory_usage_mb=np.mean(memory_usage),
        )
        
        self.benchmark_history.append(benchmark)
        
        return benchmark
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model,
        cv_folds: int = 5,
        scoring: str = "f1",
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            model: Model to evaluate
            cv_folds: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        
        # Train on each fold and collect detailed metrics
        fold_metrics = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            y_scores = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            # Calculate metrics
            fold_metric = self.calculate_metrics(y_val, y_pred, y_scores)
            fold_metrics.append(fold_metric)
        
        results = {
            "cv_scores": cv_scores.tolist(),
            "mean_score": np.mean(cv_scores),
            "std_score": np.std(cv_scores),
            "fold_metrics": [m.to_dict() for m in fold_metrics],
        }
        
        return results
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_scores: Prediction scores
            save_path: Path to save plot
            
        Returns:
            FPR, TPR, and AUC score
        """
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.output_dir / f"roc_curve_{datetime.now():%Y%m%d_%H%M%S}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        
        return fpr, tpr, roc_auc
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_scores: Prediction scores
            save_path: Path to save plot
            
        Returns:
            Precision, recall, and AUC score
        """
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.output_dir / f"pr_curve_{datetime.now():%Y%m%d_%H%M%S}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        
        return precision, recall, pr_auc
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: Names for each class
            save_path: Path to save plot
        """
        if class_names is None:
            class_names = ["Normal", "Fraud"]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.output_dir / f"confusion_matrix_{datetime.now():%Y%m%d_%H%M%S}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def generate_evaluation_report(
        self,
        metrics: EvaluationMetrics,
        fp_analysis: Optional[FalsePositiveAnalysis] = None,
        benchmark: Optional[PerformanceBenchmark] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            metrics: Evaluation metrics
            fp_analysis: False positive analysis
            benchmark: Performance benchmark
            save_path: Path to save report
            
        Returns:
            Evaluation report dictionary
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics.to_dict(),
        }
        
        if fp_analysis:
            report["false_positive_analysis"] = fp_analysis.to_dict()
        
        if benchmark:
            report["performance_benchmark"] = benchmark.to_dict()
        
        # Add summary
        report["summary"] = {
            "model_performance": "Good" if metrics.f1_score > 0.8 else "Needs Improvement",
            "false_positive_rate": metrics.fpr,
            "false_negative_rate": metrics.fnr,
            "recommended_threshold": metrics.threshold,
        }
        
        # Save report
        if save_path:
            report_path = Path(save_path)
        else:
            report_path = self.output_dir / f"evaluation_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"Evaluation report saved to: {report_path}")
        
        return report


if __name__ == "__main__":
    # Example usage
    evaluator = FraudDetectionEvaluator()
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # True labels (10% fraud)
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    
    # Simulated predictions (with some errors)
    y_scores = np.random.beta(2, 5, n_samples)  # Skewed towards lower scores
    y_scores[y_true == 1] += np.random.normal(0.3, 0.1, sum(y_true == 1))
    y_scores = np.clip(y_scores, 0, 1)
    
    y_pred = (y_scores >= 0.5).astype(int)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred, y_scores)
    
    # Plot curves
    evaluator.plot_roc_curve(y_true, y_scores)
    evaluator.plot_precision_recall_curve(y_true, y_scores)
    evaluator.plot_confusion_matrix(metrics.confusion_matrix)
    
    # Generate report
    report = evaluator.generate_evaluation_report(metrics)
    
    print("\nEvaluation Summary:")
    print(f"Precision: {metrics.precision:.3f}")
    print(f"Recall: {metrics.recall:.3f}")
    print(f"F1 Score: {metrics.f1_score:.3f}")
    print(f"AUC-ROC: {metrics.auc_roc:.3f}")
    print(f"False Positive Rate: {metrics.fpr:.3f}")
    print(f"False Negative Rate: {metrics.fnr:.3f}")