"""
Feedback loop for continuous improvement of fraud detection.

Author: Yobie Benjamin
Date: 2025-08-28
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from loguru import logger
import numpy as np
from collections import defaultdict


class FeedbackLoop:
    """
    Manages feedback collection and model improvement.
    """
    
    def __init__(self, 
                 feedback_path: Optional[Path] = None,
                 auto_retrain_threshold: int = 100):
        """
        Initialize feedback loop.
        
        Args:
            feedback_path: Path to store feedback data
            auto_retrain_threshold: Number of feedback items before auto-retraining
        """
        self.feedback_path = feedback_path or Path("data/feedback")
        self.feedback_path.mkdir(parents=True, exist_ok=True)
        
        self.auto_retrain_threshold = auto_retrain_threshold
        
        # Feedback storage
        self.feedback_queue = []
        self.feedback_stats = defaultdict(lambda: {"correct": 0, "incorrect": 0})
        
        # Performance tracking
        self.performance_history = []
        self.model_versions = {}
        
        # Load existing feedback
        self._load_feedback_history()
        
        logger.info(f"FeedbackLoop initialized with threshold: {auto_retrain_threshold}")
    
    def _load_feedback_history(self):
        """Load historical feedback data."""
        feedback_file = self.feedback_path / "feedback_history.json"
        
        if feedback_file.exists():
            try:
                with open(feedback_file, 'r') as f:
                    data = json.load(f)
                    self.feedback_queue = data.get("queue", [])
                    self.feedback_stats = defaultdict(
                        lambda: {"correct": 0, "incorrect": 0},
                        data.get("stats", {})
                    )
                    self.performance_history = data.get("performance", [])
                logger.info(f"Loaded {len(self.feedback_queue)} feedback items")
            except Exception as e:
                logger.error(f"Failed to load feedback history: {e}")
    
    def add_feedback(self, 
                     detection_result: Dict[str, Any],
                     ground_truth: bool,
                     confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Add user feedback on detection result.
        
        Args:
            detection_result: Original detection result
            ground_truth: Whether document was actually fraudulent
            confidence_threshold: Threshold for considering detection correct
            
        Returns:
            Feedback analysis
        """
        # Determine if detection was correct
        predicted_fraud = detection_result.get("fraud_score", 0) > confidence_threshold
        is_correct = predicted_fraud == ground_truth
        
        # Create feedback entry
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "detection_result": detection_result,
            "ground_truth": ground_truth,
            "predicted": predicted_fraud,
            "is_correct": is_correct,
            "confidence": detection_result.get("fraud_score", 0),
            "fraud_types": detection_result.get("fraud_types", [])
        }
        
        # Add to queue
        self.feedback_queue.append(feedback_entry)
        
        # Update statistics
        fraud_type = detection_result.get("primary_fraud_type", "unknown")
        if is_correct:
            self.feedback_stats[fraud_type]["correct"] += 1
        else:
            self.feedback_stats[fraud_type]["incorrect"] += 1
        
        # Calculate metrics
        total_feedback = len(self.feedback_queue)
        accuracy = self._calculate_accuracy()
        
        # Check if retraining is needed
        needs_retraining = total_feedback >= self.auto_retrain_threshold
        
        # Save feedback
        self._save_feedback()
        
        result = {
            "feedback_id": f"fb_{datetime.now().timestamp()}",
            "is_correct": is_correct,
            "accuracy": accuracy,
            "total_feedback": total_feedback,
            "needs_retraining": needs_retraining,
            "feedback_type": "false_positive" if predicted_fraud and not ground_truth else
                           "false_negative" if not predicted_fraud and ground_truth else
                           "true_positive" if predicted_fraud and ground_truth else
                           "true_negative"
        }
        
        logger.info(f"Added feedback: {result['feedback_type']}, Accuracy: {accuracy:.2%}")
        
        # Trigger retraining if needed
        if needs_retraining:
            asyncio.create_task(self._trigger_retraining())
        
        return result
    
    def _calculate_accuracy(self) -> float:
        """Calculate overall accuracy from feedback."""
        if not self.feedback_queue:
            return 0.0
        
        correct = sum(1 for fb in self.feedback_queue if fb["is_correct"])
        return correct / len(self.feedback_queue)
    
    def _save_feedback(self):
        """Save feedback data to file."""
        try:
            feedback_file = self.feedback_path / "feedback_history.json"
            
            with open(feedback_file, 'w') as f:
                json.dump({
                    "queue": self.feedback_queue[-1000:],  # Keep last 1000 items
                    "stats": dict(self.feedback_stats),
                    "performance": self.performance_history[-100:]  # Keep last 100 entries
                }, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
    
    async def _trigger_retraining(self):
        """Trigger model retraining based on feedback."""
        logger.info("Triggering automatic retraining based on feedback")
        
        try:
            # Prepare training data from feedback
            training_data = self._prepare_training_data()
            
            # Import fine-tuner
            from .fine_tuner import FineTuner
            
            # Initialize fine-tuner
            fine_tuner = FineTuner()
            
            # Retrain models with feedback data
            for model_name in ["document_forgery", "manipulation"]:
                # Save training data
                data_path = self.feedback_path / "training_data"
                data_path.mkdir(exist_ok=True)
                
                metadata_file = data_path / "metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(training_data, f, indent=2)
                
                # Fine-tune model
                results = fine_tuner.fine_tune_on_known_fakes(
                    data_path, 
                    model_name=model_name,
                    epochs=5  # Quick retraining
                )
                
                # Track performance
                self.performance_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "model": model_name,
                    "feedback_count": len(self.feedback_queue),
                    "accuracy_before": self._calculate_accuracy(),
                    "training_results": results
                })
            
            # Clear processed feedback
            self.feedback_queue = self.feedback_queue[-100:]  # Keep recent items
            
            logger.info("Automatic retraining completed")
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
    
    def _prepare_training_data(self) -> List[Dict[str, Any]]:
        """Prepare training data from feedback."""
        training_data = []
        
        # Convert feedback to training samples
        for feedback in self.feedback_queue:
            sample = {
                "id": feedback.get("timestamp", ""),
                "is_fake": feedback["ground_truth"],
                "confidence": abs(feedback["confidence"] - (1.0 if feedback["ground_truth"] else 0.0)),
                "fraud_types": feedback.get("fraud_types", []),
                "feedback_type": feedback.get("feedback_type", "unknown")
            }
            
            # Add pattern information if available
            if "text_content" in feedback.get("detection_result", {}):
                sample["patterns"] = self._extract_patterns(
                    feedback["detection_result"]["text_content"]
                )
            
            training_data.append(sample)
        
        return training_data
    
    def _extract_patterns(self, text: str) -> List[str]:
        """Extract patterns from text for training."""
        patterns = []
        
        # Common fake document indicators
        fake_indicators = [
            "McLovin", "MCLOVIN", "John Doe", "Jane Doe",
            "SPECIMEN", "SAMPLE", "TEST", "DEMO",
            "123 Main St", "Anytown", "12345",
            "000000000", "111111111", "123456789"
        ]
        
        for indicator in fake_indicators:
            if indicator.lower() in text.lower():
                patterns.append(indicator)
        
        return patterns
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.feedback_queue:
            return {
                "accuracy": 0.0,
                "total_feedback": 0,
                "fraud_types": {},
                "confusion_matrix": {
                    "true_positive": 0,
                    "true_negative": 0,
                    "false_positive": 0,
                    "false_negative": 0
                }
            }
        
        # Calculate confusion matrix
        confusion_matrix = {
            "true_positive": 0,
            "true_negative": 0, 
            "false_positive": 0,
            "false_negative": 0
        }
        
        for feedback in self.feedback_queue:
            predicted = feedback["predicted"]
            actual = feedback["ground_truth"]
            
            if predicted and actual:
                confusion_matrix["true_positive"] += 1
            elif not predicted and not actual:
                confusion_matrix["true_negative"] += 1
            elif predicted and not actual:
                confusion_matrix["false_positive"] += 1
            else:
                confusion_matrix["false_negative"] += 1
        
        # Calculate per-fraud-type accuracy
        fraud_type_metrics = {}
        for fraud_type, stats in self.feedback_stats.items():
            total = stats["correct"] + stats["incorrect"]
            if total > 0:
                fraud_type_metrics[fraud_type] = {
                    "accuracy": stats["correct"] / total,
                    "total": total
                }
        
        return {
            "accuracy": self._calculate_accuracy(),
            "total_feedback": len(self.feedback_queue),
            "fraud_types": fraud_type_metrics,
            "confusion_matrix": confusion_matrix,
            "precision": confusion_matrix["true_positive"] / 
                        (confusion_matrix["true_positive"] + confusion_matrix["false_positive"])
                        if (confusion_matrix["true_positive"] + confusion_matrix["false_positive"]) > 0 else 0,
            "recall": confusion_matrix["true_positive"] / 
                     (confusion_matrix["true_positive"] + confusion_matrix["false_negative"])
                     if (confusion_matrix["true_positive"] + confusion_matrix["false_negative"]) > 0 else 0
        }
    
    def get_improvement_suggestions(self) -> List[str]:
        """Get suggestions for model improvement based on feedback."""
        suggestions = []
        metrics = self.get_performance_metrics()
        
        # Check overall accuracy
        if metrics["accuracy"] < 0.8:
            suggestions.append("Overall accuracy is below 80%. Consider increasing training data.")
        
        # Check for high false positive rate
        if metrics["confusion_matrix"]["false_positive"] > metrics["confusion_matrix"]["true_positive"] * 0.3:
            suggestions.append("High false positive rate detected. Review confidence thresholds.")
        
        # Check for high false negative rate
        if metrics["confusion_matrix"]["false_negative"] > metrics["confusion_matrix"]["true_negative"] * 0.3:
            suggestions.append("High false negative rate detected. Consider adding more fraud patterns.")
        
        # Check specific fraud types
        for fraud_type, type_metrics in metrics["fraud_types"].items():
            if type_metrics["accuracy"] < 0.7:
                suggestions.append(f"Low accuracy for {fraud_type} detection ({type_metrics['accuracy']:.1%})")
        
        # Check if retraining is needed
        if len(self.feedback_queue) > self.auto_retrain_threshold * 0.8:
            suggestions.append(f"Approaching retraining threshold ({len(self.feedback_queue)}/{self.auto_retrain_threshold})")
        
        return suggestions
    
    def export_feedback_report(self) -> Dict[str, Any]:
        """Export comprehensive feedback report."""
        return {
            "generated_at": datetime.now().isoformat(),
            "metrics": self.get_performance_metrics(),
            "suggestions": self.get_improvement_suggestions(),
            "recent_feedback": self.feedback_queue[-10:],  # Last 10 items
            "performance_trend": self._calculate_performance_trend(),
            "model_versions": self.model_versions
        }
    
    def _calculate_performance_trend(self) -> Dict[str, Any]:
        """Calculate performance trend over time."""
        if len(self.performance_history) < 2:
            return {"trend": "insufficient_data"}
        
        # Get accuracy from last few checkpoints
        recent_accuracies = []
        for entry in self.performance_history[-5:]:
            if "accuracy_before" in entry:
                recent_accuracies.append(entry["accuracy_before"])
        
        if len(recent_accuracies) < 2:
            return {"trend": "insufficient_data"}
        
        # Calculate trend
        trend = np.polyfit(range(len(recent_accuracies)), recent_accuracies, 1)[0]
        
        return {
            "trend": "improving" if trend > 0.01 else "declining" if trend < -0.01 else "stable",
            "trend_value": float(trend),
            "recent_accuracies": recent_accuracies
        }