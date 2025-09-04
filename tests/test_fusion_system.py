"""
Comprehensive tests for multi-modal fusion system.

Author: Yobie Benjamin
Date: 2025
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from fraudlens.fusion.adaptive_learning import (
    ABTestFramework,
    AdaptiveLearner,
    Feedback,
    OnlineLearner,
    ThresholdOptimizer,
)
from fraudlens.fusion.explainer import ExplanationGenerator, ReportExporter
from fraudlens.fusion.fusion_engine import (
    FusedResult,
    FusionStrategy,
    ModalityWeight,
    MultiModalFraudFusion,
    RiskScore,
)
from fraudlens.fusion.pattern_matcher import (
    FraudPatternMatcher,
    Pattern,
    PatternLibrary,
    PatternType,
)
from fraudlens.fusion.risk_scorer import (
    AnomalyDetector,
    BayesianRiskAggregator,
    RiskScoringEngine,
    TimeSeriesAnalyzer,
)
from fraudlens.fusion.validators import ConsistencyReport, CrossModalValidator
from fraudlens.core.base.detector import DetectionResult, FraudType, Modality


class TestMultiModalFusion:
    """Test multi-modal fusion engine."""

    @pytest.fixture
    def fusion_engine(self):
        """Create fusion engine."""
        return MultiModalFraudFusion(
            strategy=FusionStrategy.HYBRID,
            enable_attention=True,
        )

    @pytest.fixture
    def mock_results(self):
        """Create mock detection results."""
        from datetime import datetime

        text_result = DetectionResult(
            fraud_score=0.7,
            confidence=0.85,
            fraud_types=[FraudType.PHISHING],
            explanation="Suspicious text patterns detected",
            evidence={"keywords": ["urgent", "payment"]},
            timestamp=datetime.now(),
            detector_id="text_detector_v1",
            modality=Modality.TEXT,
            processing_time_ms=50,
        )

        vision_result = DetectionResult(
            fraud_score=0.6,
            confidence=0.75,
            fraud_types=[FraudType.DOCUMENT_FORGERY],
            explanation="Document tampering detected",
            evidence={"manipulations": ["signature", "date"]},
            timestamp=datetime.now(),
            detector_id="vision_detector_v1",
            modality=Modality.IMAGE,
            processing_time_ms=100,
        )

        audio_result = DetectionResult(
            fraud_score=0.5,
            confidence=0.7,
            fraud_types=[FraudType.DEEPFAKE],
            explanation="Voice synthesis patterns detected",
            evidence={"anomalies": ["pitch_variation"]},
            timestamp=datetime.now(),
            detector_id="audio_detector_v1",
            modality=Modality.AUDIO,
            processing_time_ms=150,
        )

        return text_result, vision_result, audio_result

    @pytest.mark.asyncio
    async def test_late_fusion(self, fusion_engine, mock_results):
        """Test late fusion strategy."""
        fusion_engine.strategy = FusionStrategy.LATE
        text, vision, audio = mock_results

        result = await fusion_engine.fuse_modalities(
            text_result=text,
            vision_result=vision,
            audio_result=audio,
        )

        assert isinstance(result, FusedResult)
        assert 0 <= result.fraud_score <= 1
        assert result.fusion_strategy == FusionStrategy.LATE
        assert len(result.modality_scores) == 3
        assert result.fraud_types  # Should have fraud types

    @pytest.mark.asyncio
    async def test_early_fusion(self, fusion_engine, mock_results):
        """Test early fusion strategy."""
        fusion_engine.strategy = FusionStrategy.EARLY
        text, vision, audio = mock_results

        # Create mock features
        features = {
            "text": np.random.rand(10),
            "vision": np.random.rand(10),
            "audio": np.random.rand(10),
        }

        result = await fusion_engine.fuse_modalities(
            text_result=text,
            vision_result=vision,
            audio_result=audio,
            features=features,
        )

        assert isinstance(result, FusedResult)
        assert result.fusion_strategy == FusionStrategy.EARLY
        assert "features_shape" in result.evidence

    @pytest.mark.asyncio
    async def test_hybrid_fusion(self, fusion_engine, mock_results):
        """Test hybrid fusion with attention."""
        text, vision, audio = mock_results

        result = await fusion_engine.fuse_modalities(
            text_result=text,
            vision_result=vision,
            audio_result=audio,
        )

        assert result.fusion_strategy == FusionStrategy.HYBRID
        assert "early_score" in result.evidence
        assert "late_score" in result.evidence
        assert result.evidence["attention_enabled"]

    @pytest.mark.asyncio
    async def test_adaptive_strategy_selection(self, fusion_engine, mock_results):
        """Test adaptive strategy selection."""
        fusion_engine.strategy = FusionStrategy.ADAPTIVE
        text, vision, _ = mock_results

        # Test with few modalities
        result = await fusion_engine.fuse_modalities(
            text_result=text,
            vision_result=vision,
        )

        assert result.fusion_strategy in [FusionStrategy.LATE, FusionStrategy.EARLY]

    @pytest.mark.asyncio
    async def test_consistency_calculation(self, fusion_engine, mock_results):
        """Test consistency score calculation."""
        text, vision, audio = mock_results

        result = await fusion_engine.fuse_modalities(
            text_result=text,
            vision_result=vision,
            audio_result=audio,
        )

        assert 0 <= result.consistency_score <= 1
        # With different scores, consistency should be moderate
        assert 0.3 < result.consistency_score < 0.9

    @pytest.mark.asyncio
    async def test_weight_update(self, fusion_engine):
        """Test modality weight updates."""
        fusion_engine.update_weights("text", 0.5)

        assert fusion_engine.modality_weights["text"].base_weight == 0.5

    @pytest.mark.asyncio
    async def test_cache_functionality(self, fusion_engine, mock_results):
        """Test result caching."""
        text, vision, audio = mock_results

        # First call
        result1 = await fusion_engine.fuse_modalities(
            text_result=text,
            vision_result=vision,
        )

        # Second call with same inputs
        result2 = await fusion_engine.fuse_modalities(
            text_result=text,
            vision_result=vision,
        )

        # Should return cached result (same fraud score)
        assert result1.fraud_score == result2.fraud_score


class TestRiskScoring:
    """Test risk scoring engine."""

    @pytest.fixture
    def risk_engine(self):
        """Create risk scoring engine."""
        return RiskScoringEngine(
            enable_bayesian=True,
            enable_anomaly=True,
            enable_time_series=True,
            enable_graph=True,
        )

    @pytest.mark.asyncio
    async def test_comprehensive_risk_scoring(self, risk_engine):
        """Test comprehensive risk scoring."""
        modality_scores = {
            "text": 0.7,
            "vision": 0.6,
            "audio": 0.5,
        }

        historical_data = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        entity_features = {
            "transaction_count": 100,
            "account_age_days": 30,
            "failed_attempts": 5,
        }

        relationships = {
            "user1": ["user2", "user3"],
            "user2": ["user1", "user4"],
        }

        profile = await risk_engine.calculate_risk_score(
            fraud_score=0.65,
            confidence=0.8,
            modality_scores=modality_scores,
            historical_data=historical_data,
            entity_features=entity_features,
            relationships=relationships,
        )

        assert 0 <= profile.risk_score <= 1
        assert profile.risk_level in ["low", "medium", "high", "critical"]
        assert profile.trend in ["increasing", "stable", "decreasing"]
        assert len(profile.confidence_interval) == 2
        assert profile.factors  # Should have risk factors

    @pytest.mark.asyncio
    async def test_bayesian_aggregation(self):
        """Test Bayesian risk aggregation."""
        aggregator = BayesianRiskAggregator()

        scores = {
            "text": 0.8,
            "vision": 0.7,
            "audio": 0.6,
        }

        result = await aggregator.aggregate(scores, confidence=0.9)

        assert 0 <= result <= 1

    @pytest.mark.asyncio
    async def test_anomaly_detection(self):
        """Test anomaly detection."""
        detector = AnomalyDetector()

        # Normal features
        normal_features = {
            "transaction_amount": 100,
            "login_attempts": 1,
            "session_duration": 300,
        }

        normal_score = await detector.detect(normal_features)

        # Anomalous features
        anomaly_features = {
            "transaction_amount": 10000,
            "login_attempts": 50,
            "session_duration": 5,
        }

        anomaly_score = await detector.detect(anomaly_features)

        assert 0 <= normal_score <= 1
        assert 0 <= anomaly_score <= 1
        # Anomaly should have higher score
        assert anomaly_score >= normal_score

    @pytest.mark.asyncio
    async def test_time_series_analysis(self):
        """Test time series analysis."""
        analyzer = TimeSeriesAnalyzer()

        # Increasing trend
        increasing_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        score, trend = await analyzer.analyze(increasing_data)

        assert 0 <= score <= 1
        assert trend == "increasing"

        # Stable trend
        stable_data = [0.5, 0.48, 0.52, 0.49, 0.51, 0.50]
        score, trend = await analyzer.analyze(stable_data)

        assert trend == "stable"


class TestCrossModalValidation:
    """Test cross-modal validation."""

    @pytest.fixture
    def validator(self):
        """Create cross-modal validator."""
        return CrossModalValidator(
            similarity_threshold=0.7,
            sync_tolerance_ms=100,
        )

    @pytest.mark.asyncio
    async def test_text_image_consistency(self, validator):
        """Test text-image consistency validation."""
        text_result = DetectionResult(
            fraud_score=0.7,
            confidence=0.85,
            fraud_types=[FraudType.PHISHING],
            explanation="",
            evidence={"entities": ["PayPal", "account", "payment"]},
            modality=Modality.TEXT,
            processing_time_ms=50,
            timestamp=datetime.now(),
            detector_id="text_detector",
        )

        image_result = DetectionResult(
            fraud_score=0.6,
            confidence=0.75,
            fraud_types=[FraudType.PHISHING],
            explanation="",
            evidence={"objects": ["logo", "form"], "text_in_image": ["PayPal"]},
            modality=Modality.IMAGE,
            processing_time_ms=100,
            timestamp=datetime.now(),
            detector_id="image_detector",
        )

        report = await validator.validate_consistency(
            text_result=text_result,
            image_result=image_result,
        )

        assert isinstance(report, ConsistencyReport)
        assert report.text_image_consistency is not None
        assert report.overall_consistency > 0.5  # Should have some consistency

    @pytest.mark.asyncio
    async def test_metadata_validation(self, validator):
        """Test metadata-content validation."""
        content = "Document created by John Doe" * 50
        raw_data = {
            "metadata": {
                "creation_date": "2024-01-01",
                "modification_date": "2024-01-02",
                "author": "John Doe",
                "file_size": len(content.encode()),  # Match actual content size in bytes
            },
            "content": content,
        }

        report = await validator.validate_consistency(raw_data=raw_data)

        assert report.metadata_content_validation is not None
        assert report.metadata_content_validation.is_valid  # Should be valid

    @pytest.mark.asyncio
    async def test_temporal_consistency(self, validator):
        """Test temporal consistency checking."""
        result1 = DetectionResult(
            fraud_score=0.5,
            confidence=0.8,
            fraud_types=[],
            explanation="",
            evidence={},
            modality=Modality.TEXT,
            processing_time_ms=50,
            timestamp=datetime.now(),
            detector_id="detector1",
        )

        result2 = DetectionResult(
            fraud_score=0.5,
            confidence=0.8,
            fraud_types=[],
            explanation="",
            evidence={},
            modality=Modality.IMAGE,
            processing_time_ms=55,
            timestamp=datetime.now() + timedelta(seconds=1),
            detector_id="detector2",
        )

        report = await validator.validate_consistency(
            text_result=result1,
            image_result=result2,
        )

        assert report.temporal_consistency is not None


class TestPatternMatching:
    """Test fraud pattern matching."""

    @pytest.fixture
    def pattern_library(self, tmp_path):
        """Create pattern library."""
        library_path = tmp_path / "test_patterns.json"
        return PatternLibrary(library_path)

    @pytest.fixture
    def pattern_matcher(self, pattern_library):
        """Create pattern matcher."""
        return FraudPatternMatcher(pattern_library)

    @pytest.mark.asyncio
    async def test_text_pattern_matching(self, pattern_matcher):
        """Test text regex pattern matching."""
        data = {
            "text": "URGENT: Your account will be suspended immediately unless you make a payment now!"
        }

        matches = await pattern_matcher.match_patterns(
            data,
            pattern_types=[PatternType.TEXT_REGEX],
        )

        assert len(matches) > 0
        assert matches[0].pattern.type == PatternType.TEXT_REGEX
        assert matches[0].confidence > 0.7

    @pytest.mark.asyncio
    async def test_behavioral_pattern_matching(self, pattern_matcher):
        """Test behavioral pattern matching."""
        data = {
            "transactions": [
                {"amount": 9500, "timestamp": 1000},
                {"amount": 9400, "timestamp": 1100},
                {"amount": 9600, "timestamp": 1200},
                {"amount": 9300, "timestamp": 1300},
            ]
        }

        matches = await pattern_matcher.match_patterns(
            data,
            pattern_types=[PatternType.BEHAVIORAL],
            tags=["structuring"],
        )

        assert len(matches) > 0
        assert "structured_count" in matches[0].evidence
        assert matches[0].evidence["structured_count"] >= 3

    @pytest.mark.asyncio
    async def test_sequence_pattern_matching(self, pattern_matcher):
        """Test sequence pattern matching."""
        data = {
            "events": [
                {"type": "password_reset", "timestamp": 1000},
                {"type": "email_change", "timestamp": 2000},
                {"type": "large_withdrawal", "timestamp": 3000},
            ]
        }

        matches = await pattern_matcher.match_patterns(
            data,
            pattern_types=[PatternType.SEQUENCE],
        )

        # Should match account takeover sequence
        assert any(m.pattern.pattern_id == "seq_fraud_001" for m in matches)

    def test_pattern_performance_evaluation(self, pattern_library):
        """Test pattern performance evaluation."""
        pattern = pattern_library.get_pattern("txt_phish_001")
        pattern.match_count = 100
        pattern.false_positive_count = 10

        matcher = FraudPatternMatcher(pattern_library)
        perf = matcher.evaluate_pattern_performance("txt_phish_001")

        assert perf["precision"] == 0.9
        assert perf["total_matches"] == 100


class TestExplanationGeneration:
    """Test explanation generation."""

    @pytest.fixture
    def explainer(self):
        """Create explanation generator."""
        return ExplanationGenerator(language="en", detail_level="medium")

    @pytest.fixture
    def mock_fused_result(self):
        """Create mock fused result."""
        return FusedResult(
            fraud_score=0.75,
            confidence=0.85,
            fraud_types=[FraudType.PHISHING, FraudType.MONEY_LAUNDERING],
            modality_scores={"text": 0.8, "vision": 0.7},
            fusion_strategy=FusionStrategy.HYBRID,
            consistency_score=0.9,
            explanation="Multi-modal analysis detected fraud",
            evidence={"key_indicators": ["suspicious_text", "altered_image"]},
            processing_time_ms=200,
        )

    @pytest.fixture
    def mock_risk_score(self):
        """Create mock risk score."""
        return RiskScore(
            overall_risk=0.75,
            risk_level="high",
            risk_factors=[
                {"factor": "base_fraud_score", "value": 0.75, "weight": 0.3},
                {"factor": "anomaly_detection", "value": 0.8, "weight": 0.2},
            ],
            confidence_intervals=(0.7, 0.8),
            modality_contributions={"text": 0.4, "vision": 0.35},
            anomaly_score=0.8,
            trend="increasing",
        )

    @pytest.mark.asyncio
    async def test_explanation_generation(self, explainer, mock_fused_result, mock_risk_score):
        """Test comprehensive explanation generation."""
        explanation = await explainer.generate_explanation(
            mock_fused_result,
            mock_risk_score,
        )

        assert explanation.summary
        assert len(explanation.risk_factors) > 0
        assert explanation.confidence_explanation
        assert len(explanation.recommendations) > 0
        assert explanation.visual_highlights
        assert len(explanation.audit_trail) > 0

    @pytest.mark.asyncio
    async def test_report_export_json(self, tmp_path, mock_fused_result, mock_risk_score):
        """Test JSON report export."""
        exporter = ReportExporter(output_dir=tmp_path)
        explainer = ExplanationGenerator()

        explanation = await explainer.generate_explanation(
            mock_fused_result,
            mock_risk_score,
        )

        report_path = await exporter.export_report(
            case_id="TEST001",
            fused_result=mock_fused_result,
            risk_score=mock_risk_score,
            explanation=explanation,
            format="json",
        )

        assert report_path.exists()

        with open(report_path) as f:
            report = json.load(f)

        assert report["case_id"] == "TEST001"
        assert "risk_score" in report
        assert "explanation" in report


class TestAdaptiveLearning:
    """Test adaptive learning system."""

    @pytest.fixture
    def adaptive_learner(self):
        """Create adaptive learner."""
        return AdaptiveLearner(
            learning_rate=0.01,
            min_feedback_for_update=5,
        )

    @pytest.mark.asyncio
    async def test_feedback_processing(self, adaptive_learner):
        """Test feedback processing."""
        feedback = Feedback(
            case_id="CASE001",
            true_label=True,
            detection_result={"fraud_score": 0.8},
            confidence=0.9,
            feedback_type="manual",
        )

        await adaptive_learner.process_feedback(feedback)

        assert len(adaptive_learner.feedback_buffer) == 1
        assert adaptive_learner.performance_metrics["true_positives"] == 1

    @pytest.mark.asyncio
    async def test_threshold_update(self, adaptive_learner):
        """Test threshold updates based on feedback."""
        # Add multiple feedback samples
        for i in range(10):
            feedback = Feedback(
                case_id=f"CASE{i:03d}",
                true_label=i % 2 == 0,  # Alternate
                detection_result={"fraud_score": 0.6},
                confidence=0.8,
                feedback_type="manual",
            )
            await adaptive_learner.process_feedback(feedback)

        # Should have updated thresholds
        text_threshold = adaptive_learner.get_threshold("text")
        assert text_threshold.performance_metrics  # Should have metrics

    @pytest.mark.asyncio
    async def test_online_learning(self):
        """Test online learning."""
        learner = OnlineLearner(model_type="sgd")

        # Train with batch
        for i in range(50):
            features = {
                "feature1": np.random.rand(),
                "feature2": np.random.rand(),
                "feature3": np.random.rand(),
            }
            label = np.random.choice([True, False])

            await learner.partial_fit(features, label)

        # Make prediction
        test_features = {
            "feature1": 0.5,
            "feature2": 0.5,
            "feature3": 0.5,
        }

        is_fraud, confidence = await learner.predict(test_features)

        assert isinstance(is_fraud, bool)
        assert 0 <= confidence <= 1

    def test_threshold_optimization(self):
        """Test threshold optimization."""
        optimizer = ThresholdOptimizer(
            false_positive_cost=1.0,
            false_negative_cost=10.0,
        )

        # Add sample results
        for i in range(100):
            score = np.random.rand()
            true_label = score > 0.5  # Simple threshold
            optimizer.add_result(score, true_label)

        # Find optimal threshold
        optimal = optimizer.find_optimal_threshold(metric="cost")

        assert 0 <= optimal <= 1

        # Get analysis
        analysis = optimizer.get_threshold_analysis()
        assert "thresholds" in analysis
        assert "metrics" in analysis

    def test_ab_testing_framework(self):
        """Test A/B testing framework."""
        framework = ABTestFramework()

        # Create test
        test = framework.create_test(
            name="New Model Test",
            description="Testing improved fraud detection model",
            variant_a={"model": "v1", "threshold": 0.5},
            variant_b={"model": "v2", "threshold": 0.6},
            traffic_split=0.5,
            duration_days=7,
        )

        assert test.test_id in framework.active_tests

        # Assign variants and record metrics
        for i in range(100):
            variant = framework.assign_variant(test.test_id)
            assert variant in ["A", "B"]

            # Record metric
            framework.record_metric(
                test.test_id,
                variant,
                "accuracy",
                0.8 + np.random.rand() * 0.1,
            )

        # Get results
        results = framework.get_test_results(test.test_id)

        assert "variants" in results
        assert "A" in results["variants"]
        assert "B" in results["variants"]


class TestIntegration:
    """Integration tests for complete fusion system."""

    @pytest.mark.asyncio
    async def test_end_to_end_fusion_pipeline(self, tmp_path):
        """Test complete fusion pipeline."""
        # Initialize components
        fusion = MultiModalFraudFusion(strategy=FusionStrategy.HYBRID)
        risk_scorer = RiskScoringEngine()
        validator = CrossModalValidator()
        explainer = ExplanationGenerator()
        pattern_matcher = FraudPatternMatcher()
        adaptive_learner = AdaptiveLearner()

        # Create mock detection results
        text_result = DetectionResult(
            fraud_score=0.8,
            confidence=0.9,
            fraud_types=[FraudType.PHISHING],
            explanation="Phishing detected",
            evidence={"keywords": ["urgent", "verify"]},
            modality=Modality.TEXT,
            processing_time_ms=50,
            timestamp=datetime.now(),
            detector_id="text_detector",
        )

        vision_result = DetectionResult(
            fraud_score=0.7,
            confidence=0.85,
            fraud_types=[FraudType.DOCUMENT_FORGERY],
            explanation="Document tampering detected",
            evidence={"alterations": ["signature"]},
            modality=Modality.IMAGE,
            processing_time_ms=100,
            timestamp=datetime.now(),
            detector_id="vision_detector",
        )

        # Step 1: Fusion
        fused_result = await fusion.fuse_modalities(
            text_result=text_result,
            vision_result=vision_result,
        )

        assert fused_result.fraud_score > 0.5

        # Step 2: Risk scoring
        risk_profile = await risk_scorer.calculate_risk_score(
            fraud_score=fused_result.fraud_score,
            confidence=fused_result.confidence,
            modality_scores=fused_result.modality_scores,
        )

        assert risk_profile.risk_level in ["low", "medium", "high", "critical"]

        # Step 3: Cross-modal validation
        consistency_report = await validator.validate_consistency(
            text_result=text_result,
            image_result=vision_result,
        )

        assert consistency_report.overall_consistency >= 0

        # Step 4: Pattern matching
        data = {
            "text": "Urgent: Verify your account immediately!",
        }

        matches = await pattern_matcher.match_patterns(data)
        assert isinstance(matches, list)

        # Step 5: Generate explanation
        explanation = await explainer.generate_explanation(
            fused_result,
            risk_profile,
            consistency_report,
        )

        assert explanation.summary
        assert explanation.recommendations

        # Step 6: Export report
        exporter = ReportExporter(output_dir=tmp_path)
        report_path = await exporter.export_report(
            case_id="INT_TEST_001",
            fused_result=fused_result,
            risk_score=risk_profile,
            explanation=explanation,
            consistency_report=consistency_report,
            format="json",
        )

        assert report_path.exists()

        # Step 7: Process feedback
        feedback = Feedback(
            case_id="INT_TEST_001",
            true_label=True,  # Was actually fraud
            detection_result=fused_result.to_dict(),
            confidence=0.9,
            feedback_type="verified",
        )

        await adaptive_learner.process_feedback(feedback)

        # Verify complete pipeline execution
        assert adaptive_learner.performance_metrics["true_positives"] == 1

    @pytest.mark.asyncio
    async def test_ablation_study(self):
        """Test system performance with different components disabled."""
        # Test with different fusion strategies
        strategies = [
            FusionStrategy.EARLY,
            FusionStrategy.LATE,
            FusionStrategy.HYBRID,
            FusionStrategy.HIERARCHICAL,
        ]

        results = {}

        for strategy in strategies:
            fusion = MultiModalFraudFusion(strategy=strategy)

            # Create consistent test data
            text_result = DetectionResult(
                fraud_score=0.7,
                confidence=0.85,
                fraud_types=[FraudType.PHISHING],
                explanation="",
                evidence={},
                modality=Modality.TEXT,
                processing_time_ms=50,
                timestamp=datetime.now(),
                detector_id="text_detector",
            )

            vision_result = DetectionResult(
                fraud_score=0.6,
                confidence=0.75,
                fraud_types=[FraudType.PHISHING],
                explanation="",
                evidence={},
                modality=Modality.IMAGE,
                processing_time_ms=100,
                timestamp=datetime.now(),
                detector_id="vision_detector",
            )

            fused = await fusion.fuse_modalities(
                text_result=text_result,
                vision_result=vision_result,
            )

            results[strategy.value] = {
                "fraud_score": fused.fraud_score,
                "confidence": fused.confidence,
                "consistency": fused.consistency_score,
            }

        # Verify all strategies produce valid results
        for strategy_name, metrics in results.items():
            assert 0 <= metrics["fraud_score"] <= 1
            assert 0 <= metrics["confidence"] <= 1
            assert 0 <= metrics["consistency"] <= 1

    def test_performance_metrics(self):
        """Test system performance metrics."""
        fusion = MultiModalFraudFusion()
        risk_scorer = RiskScoringEngine()
        pattern_library = PatternLibrary()

        # Get performance stats
        fusion_stats = fusion.get_performance_stats()
        assert "avg_fusion_time_ms" in fusion_stats
        assert "total_fusions" in fusion_stats

        pattern_stats = pattern_library.get_performance_stats()
        assert "total_patterns" in pattern_stats
        assert "patterns_by_type" in pattern_stats

        # Verify metrics are reasonable
        assert fusion_stats["avg_fusion_time_ms"] >= 0
        assert pattern_stats["total_patterns"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
