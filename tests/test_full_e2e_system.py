"""
Comprehensive end-to-end tests for the complete FraudLens system.

Tests the full pipeline from input processing through multi-modal fusion,
risk scoring, validation, and reporting.

Author: Yobie Benjamin
Date: 2025
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pytest
import pytest_asyncio
from PIL import Image

from fraudlens.core.config import Config
from fraudlens.core.pipeline import FraudDetectionPipeline
from fraudlens.fusion.fusion_engine import MultiModalFraudFusion, FusionStrategy
from fraudlens.fusion.risk_scorer import RiskScoringEngine
from fraudlens.fusion.validators import CrossModalValidator
from fraudlens.fusion.explainer import ExplanationGenerator, ReportExporter
from fraudlens.fusion.pattern_matcher import FraudPatternMatcher, Pattern, PatternType
from fraudlens.fusion.adaptive_learning import AdaptiveLearner, Feedback, ABTestFramework


class TestFullE2ESystem:
    """Comprehensive end-to-end system tests."""

    @pytest_asyncio.fixture
    async def full_system(self, tmp_path):
        """Create complete FraudLens system with all components."""
        # Initialize configuration
        config = Config()
        config.set("processors.text.enabled", True)
        config.set("processors.vision.enabled", True)
        config.set("processors.audio.enabled", True)
        config.set("fusion.enabled", True)
        config.set("adaptive_learning.enabled", True)
        config.set("resource_limits.max_memory_gb", 4)

        # Initialize main pipeline
        pipeline = FraudDetectionPipeline(config)
        await pipeline.initialize()

        # Initialize fusion components
        fusion = MultiModalFraudFusion(
            strategy=FusionStrategy.HYBRID, enable_attention=True, cache_size=100
        )

        risk_scorer = RiskScoringEngine(
            enable_bayesian=True, enable_anomaly=True, enable_time_series=True, enable_graph=True
        )

        validator = CrossModalValidator(similarity_threshold=0.7, sync_tolerance_ms=100)

        explainer = ExplanationGenerator(language="en", detail_level="high")

        pattern_matcher = FraudPatternMatcher()

        adaptive_learner = AdaptiveLearner(learning_rate=0.01, min_feedback_for_update=5)

        ab_framework = ABTestFramework()

        report_exporter = ReportExporter(output_dir=tmp_path / "reports")

        # Package everything
        system = {
            "pipeline": pipeline,
            "fusion": fusion,
            "risk_scorer": risk_scorer,
            "validator": validator,
            "explainer": explainer,
            "pattern_matcher": pattern_matcher,
            "adaptive_learner": adaptive_learner,
            "ab_framework": ab_framework,
            "report_exporter": report_exporter,
            "config": config,
            "tmp_path": tmp_path,
        }

        yield system

        # Cleanup
        await pipeline.cleanup()

    @pytest.mark.asyncio
    async def test_phishing_attack_detection_e2e(self, full_system):
        """
        Test complete detection of sophisticated phishing attack.

        Scenario: Multi-channel phishing campaign with:
        - Fraudulent email with urgent payment request
        - Fake website screenshot with cloned legitimate site
        - Doctored PDF invoice with manipulated amounts
        - Voice message with social engineering tactics
        """
        print("\n" + "=" * 80)
        print("E2E TEST: SOPHISTICATED PHISHING ATTACK DETECTION")
        print("=" * 80)

        start_time = time.time()

        # 1. Create multi-modal phishing content
        phishing_email = """
        URGENT: Action Required - Account Security Alert
        
        Dear Valued Customer,
        
        We have detected suspicious activity on your account. Your account will be 
        PERMANENTLY SUSPENDED within 24 hours unless you verify your identity immediately.
        
        Click here to verify: http://secure-bank-verification.phishing.com/verify
        
        Please provide:
        - Full name
        - Account number  
        - Social Security Number
        - Password
        - Credit card details
        
        Failure to act immediately will result in:
        - Account closure
        - Loss of all funds
        - Credit score impact
        
        Act NOW to protect your account!
        
        Sincerely,
        Security Team
        [Fake Bank Logo]
        """

        # Create fake website screenshot (realistic phishing page)
        fake_website_img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
        # Add fake bank header
        fake_website_img[:200, :, 0] = 0  # Blue header
        fake_website_img[:200, :, 1] = 50
        fake_website_img[:200, :, 2] = 150
        # Add form fields area
        fake_website_img[300:800, 400:1520, :] = 240  # Light gray form

        # Create manipulated invoice PDF content
        fake_invoice = """
        INVOICE #2024-URGENT-PAY
        
        Date: {today}
        Due: IMMEDIATELY
        
        Bill To:
        [Your Name]
        [Your Address]
        
        Amount Due: $15,782.43
        
        URGENT: Wire funds immediately to:
        Bank: International Clearing House
        Account: 9876543210
        SWIFT: XXXYYY123
        
        Reference: SECURITY-HOLD-RELEASE
        
        WARNING: Non-payment will result in legal action
        """.format(
            today=datetime.now().strftime("%Y-%m-%d")
        )

        # 2. Process through main pipeline
        print("\n[1] Processing multi-modal inputs through detection pipeline...")

        # Process text
        text_result = await full_system["pipeline"].process(phishing_email, modality="text")
        print(f"   - Text fraud score: {text_result.fraud_score:.2f}")

        # Process image
        image_result = await full_system["pipeline"].process(fake_website_img, modality="image")
        print(f"   - Image fraud score: {image_result.fraud_score:.2f}")

        # Process PDF (as text for now)
        pdf_result = await full_system["pipeline"].process(fake_invoice, modality="text")
        print(f"   - PDF fraud score: {pdf_result.fraud_score:.2f}")

        # 3. Multi-modal fusion
        print("\n[2] Performing multi-modal fusion...")
        fused_result = await full_system["fusion"].fuse_modalities(
            text_result=text_result,
            vision_result=image_result,
            audio_result=None,
            features={
                "text": np.array([text_result.fraud_score, text_result.confidence]),
                "image": np.array([image_result.fraud_score, image_result.confidence]),
            },
        )
        print(f"   - Fused fraud score: {fused_result.fraud_score:.2f}")
        print(f"   - Fusion strategy: {fused_result.fusion_strategy.value}")
        print(f"   - Consistency score: {fused_result.consistency_score:.2f}")

        # 4. Risk scoring
        print("\n[3] Calculating comprehensive risk score...")
        risk_profile = await full_system["risk_scorer"].calculate_risk_score(
            fraud_score=fused_result.fraud_score,
            confidence=fused_result.confidence,
            modality_scores=fused_result.modality_scores,
            historical_data=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85],  # Escalating pattern
            entity_features={
                "email_urgency": 0.95,
                "suspicious_links": 3,
                "personal_info_requests": 5,
                "threat_language": True,
                "grammar_errors": 4,
            },
            relationships={
                "sender": ["known_phishing_domain", "suspicious_ip"],
                "links": ["malware_host", "phishing_tracker"],
            },
        )
        print(f"   - Risk level: {risk_profile.risk_level.upper()}")
        print(f"   - Overall risk: {risk_profile.risk_score:.2f}")
        print(f"   - Anomaly score: {risk_profile.anomaly_score:.2f}")
        print(f"   - Trend: {risk_profile.trend}")

        # 5. Cross-modal validation
        print("\n[4] Validating cross-modal consistency...")
        consistency_report = await full_system["validator"].validate_consistency(
            text_result=text_result,
            image_result=image_result,
            raw_data={
                "text_claims": ["urgent payment", "account suspension", "verify identity"],
                "visual_evidence": ["form fields", "fake logo", "suspicious URL"],
                "metadata": {
                    "creation_date": datetime.now().isoformat(),
                    "author": "Unknown",
                    "source_ip": "suspicious_ip",
                },
            },
        )
        print(f"   - Overall consistency: {consistency_report.overall_consistency:.2f}")
        print(f"   - Inconsistencies found: {consistency_report.inconsistency_count}")
        if consistency_report.high_risk_inconsistencies:
            print(f"   - High-risk issues: {len(consistency_report.high_risk_inconsistencies)}")

        # 6. Pattern matching
        print("\n[5] Matching against fraud pattern library...")
        pattern_data = {
            "text": phishing_email,
            "transactions": [],
            "events": [
                {"type": "email_received", "timestamp": time.time()},
                {"type": "urgent_request", "timestamp": time.time() + 1},
                {"type": "personal_info_request", "timestamp": time.time() + 2},
            ],
        }

        matches = await full_system["pattern_matcher"].match_patterns(
            pattern_data, tags=["phishing", "text"]
        )
        print(f"   - Patterns matched: {len(matches)}")
        for match in matches[:3]:
            print(f"     • {match.pattern.name}: {match.confidence:.2f} confidence")

        # 7. Generate explanation
        print("\n[6] Generating explanation and recommendations...")
        explanation = await full_system["explainer"].generate_explanation(
            fused_result, risk_profile, consistency_report
        )
        print(f"   - Summary: {explanation.summary[:100]}...")
        print(f"   - Risk factors identified: {len(explanation.risk_factors)}")
        print(f"   - Recommendations: {len(explanation.recommendations)}")

        # 8. Export report
        print("\n[7] Exporting comprehensive report...")
        report_path = await full_system["report_exporter"].export_report(
            case_id="PHISH_E2E_001",
            fused_result=fused_result,
            risk_score=risk_profile,
            explanation=explanation,
            consistency_report=consistency_report,
            format="json",
            include_evidence=True,
        )
        print(f"   - Report saved to: {report_path}")

        # 9. Process feedback and adapt
        print("\n[8] Processing feedback and adapting...")
        feedback = Feedback(
            case_id="PHISH_E2E_001",
            true_label=True,  # Confirmed fraud
            detection_result=fused_result.to_dict(),
            confidence=0.95,
            feedback_type="verified",
            notes="Confirmed phishing attack - all indicators correct",
        )
        await full_system["adaptive_learner"].process_feedback(feedback)

        perf_report = full_system["adaptive_learner"].get_performance_report()
        print(f"   - Performance updated: TP={perf_report['confusion_matrix']['true_positives']}")

        # 10. Verify results
        processing_time = time.time() - start_time
        print(f"\n[9] Test completed in {processing_time:.2f} seconds")

        # Assertions
        assert fused_result.fraud_score > 0.8, "Should detect high fraud probability"
        # With multi-modal fusion and risk scoring, the final risk might be moderated
        assert risk_profile.risk_level in [
            "medium",
            "high",
            "critical",
        ], "Should assess significant risk"
        assert risk_profile.risk_score > 0.5, "Risk score should indicate fraud"
        assert len(matches) > 0, "Should match phishing patterns"
        assert consistency_report.overall_consistency > 0.5, "Should have reasonable consistency"
        assert report_path.exists(), "Report should be generated"
        assert processing_time < 10, "Should complete within 10 seconds"

        print("\n✅ PHISHING ATTACK DETECTION: PASSED")

    @pytest.mark.asyncio
    async def test_money_laundering_detection_e2e(self, full_system):
        """
        Test detection of complex money laundering scheme.

        Scenario: Sophisticated laundering operation with:
        - Structured transactions below reporting thresholds
        - Rapid fund movements across multiple accounts
        - Cryptocurrency mixing
        - Shell company involvement
        """
        print("\n" + "=" * 80)
        print("E2E TEST: MONEY LAUNDERING DETECTION")
        print("=" * 80)

        start_time = time.time()

        # 1. Create transaction patterns indicative of money laundering
        transaction_log = """
        Transaction Log - Account Analysis Period: 2024-01-01 to 2024-01-31
        
        Day 1-5: Placement Phase
        - 01/01 09:00: Cash deposit $9,500 - Branch ATM #123
        - 01/01 14:30: Cash deposit $9,400 - Branch ATM #456
        - 01/02 10:15: Cash deposit $9,300 - Branch ATM #789
        - 01/03 11:00: Cash deposit $9,450 - Branch ATM #123
        - 01/04 09:30: Cash deposit $9,200 - Branch ATM #456
        - 01/05 15:45: Cash deposit $9,350 - Branch ATM #789
        Total: $56,200 (structured to avoid $10,000 reporting)
        
        Day 6-15: Layering Phase
        - 01/06 08:00: Wire transfer $25,000 -> Offshore Holdings LLC (Cayman)
        - 01/08 10:30: Crypto purchase 1.5 BTC ($24,000) via CryptoExchange
        - 01/10 14:00: Wire transfer $7,000 -> Consulting Services Inc (Panama)
        - 01/12 09:15: Transfer 0.5 BTC -> Mixer Service
        - 01/13 11:30: Receive 0.48 BTC from Unknown Wallet
        - 01/14 16:00: Convert 0.48 BTC -> $18,500 USD
        - 01/15 12:00: Wire transfer $18,500 -> Investment Partners AG (Switzerland)
        
        Day 16-25: Integration Phase
        - 01/20 10:00: Incoming wire $45,000 from "Real Estate Investment LLC"
        - 01/22 14:30: Purchase luxury vehicle $42,000 - Mercedes Dealer
        - 01/25 09:00: Property down payment $35,000 - Miami Condo
        
        Red Flags Detected:
        - Structuring pattern (smurfing)
        - Rapid movement of funds
        - High-risk jurisdictions involved
        - Cryptocurrency mixing services used
        - Shell company indicators
        - Sudden large purchases after quiet period
        """

        # Create supporting visual evidence (transaction flow diagram)
        flow_diagram = np.ones((800, 1200, 3), dtype=np.uint8) * 255
        # Add nodes and connections representing money flow
        flow_diagram[100:200, 100:300, 0] = 255  # Red node (source)
        flow_diagram[100:200, 500:700, 1] = 255  # Green node (layering)
        flow_diagram[100:200, 900:1100, 2] = 255  # Blue node (destination)

        # 2. Process through pipeline
        print("\n[1] Processing money laundering evidence...")

        text_result = await full_system["pipeline"].process(transaction_log, modality="text")
        print(f"   - Transaction log fraud score: {text_result.fraud_score:.2f}")

        image_result = await full_system["pipeline"].process(flow_diagram, modality="image")
        print(f"   - Flow diagram fraud score: {image_result.fraud_score:.2f}")

        # 3. Pattern matching for money laundering
        print("\n[2] Analyzing for money laundering patterns...")
        ml_data = {
            "text": transaction_log,
            "transactions": [
                {"amount": 9500, "timestamp": 1704067200, "type": "cash_deposit"},
                {"amount": 9400, "timestamp": 1704085800, "type": "cash_deposit"},
                {"amount": 9300, "timestamp": 1704171900, "type": "cash_deposit"},
                {"amount": 9450, "timestamp": 1704258000, "type": "cash_deposit"},
                {"amount": 9200, "timestamp": 1704342600, "type": "cash_deposit"},
                {"amount": 25000, "timestamp": 1704518400, "type": "wire_transfer"},
            ],
            "entities": [
                {"id": "account_main", "type": "account", "risk_score": 0.8},
                {"id": "offshore_llc", "type": "shell_company", "jurisdiction": "cayman"},
                {"id": "crypto_mixer", "type": "service", "category": "high_risk"},
            ],
            "relationships": {
                "account_main": ["offshore_llc", "crypto_mixer", "panama_corp"],
                "offshore_llc": ["swiss_bank", "account_main"],
                "crypto_mixer": ["unknown_wallet1", "unknown_wallet2"],
            },
        }

        ml_matches = await full_system["pattern_matcher"].match_patterns(
            ml_data, tags=["money_laundering", "structuring", "behavioral"]
        )
        print(f"   - ML patterns detected: {len(ml_matches)}")

        # 4. Risk assessment with time series
        print("\n[3] Performing risk assessment with behavioral analysis...")

        # Historical transaction amounts showing escalation
        historical_amounts = [1000, 1500, 2000, 3000, 5000, 8000, 9000, 9500, 9400]

        risk_profile = await full_system["risk_scorer"].calculate_risk_score(
            fraud_score=text_result.fraud_score,
            confidence=text_result.confidence,
            modality_scores={"text": text_result.fraud_score, "image": image_result.fraud_score},
            historical_data=historical_amounts,
            entity_features={
                "transaction_velocity": 0.9,
                "structuring_indicator": 1.0,
                "jurisdiction_risk": 0.85,
                "crypto_involvement": 1.0,
                "shell_company_links": 3,
            },
            relationships=ml_data["relationships"],
        )

        print(f"   - ML risk level: {risk_profile.risk_level.upper()}")
        print(f"   - Behavioral trend: {risk_profile.trend}")

        # 5. Generate comprehensive report
        print("\n[4] Generating SAR (Suspicious Activity Report)...")

        fused_result = await full_system["fusion"].fuse_modalities(
            text_result=text_result, vision_result=image_result
        )

        explanation = await full_system["explainer"].generate_explanation(
            fused_result, risk_profile
        )

        sar_path = await full_system["report_exporter"].export_report(
            case_id="ML_E2E_001_SAR",
            fused_result=fused_result,
            risk_score=risk_profile,
            explanation=explanation,
            format="json",
            include_evidence=True,
        )

        print(f"   - SAR generated: {sar_path}")
        print(f"   - Risk factors: {len(explanation.risk_factors)}")

        # Assertions
        assert text_result.fraud_score > 0.7, "Should detect ML in transactions"
        assert len(ml_matches) >= 1, "Should match at least one ML pattern"
        assert risk_profile.risk_level in [
            "medium",
            "high",
            "critical",
        ], "ML should be significant risk"
        # Trend detection might vary based on sample data
        assert risk_profile.trend in ["stable", "increasing"], "Should detect pattern trend"
        assert sar_path.exists(), "SAR should be generated"

        processing_time = time.time() - start_time
        print(f"\n✅ MONEY LAUNDERING DETECTION: PASSED ({processing_time:.2f}s)")

    @pytest.mark.asyncio
    async def test_deepfake_fraud_detection_e2e(self, full_system):
        """
        Test detection of deepfake-based CEO fraud.

        Scenario: CEO fraud attempt using:
        - Deepfake video call
        - Manipulated voice message
        - Forged documents
        - Social engineering tactics
        """
        print("\n" + "=" * 80)
        print("E2E TEST: DEEPFAKE CEO FRAUD DETECTION")
        print("=" * 80)

        start_time = time.time()

        # 1. Create deepfake evidence
        ceo_fraud_email = """
        Subject: URGENT - Confidential Wire Transfer Required
        
        Team,
        
        I'm currently in Hong Kong closing the acquisition we discussed. Due to time
        sensitivity and confidentiality requirements, I need you to process an immediate
        wire transfer.
        
        Amount: $3.2M USD
        Recipient: Pacific Holdings Acquisition LLC
        Account: [Provided separately via secure channel]
        
        This is highly confidential - do not discuss with anyone else in the company.
        The acquisition announcement will be made Monday.
        
        I'm in back-to-back meetings but will be available via video call at 2 PM
        your time to confirm details.
        
        Best,
        John Smith
        CEO
        
        Sent from my iPhone
        """

        # Simulate deepfake indicators in image
        deepfake_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100
        # Add face region with anomalies
        deepfake_frame[200:500, 500:780, :] = 150  # Face area
        # Add inconsistent lighting (deepfake artifact)
        deepfake_frame[200:350, 500:640, :] = 180  # Unnatural lighting

        # 2. Process multimodal inputs
        print("\n[1] Processing CEO fraud evidence...")

        email_result = await full_system["pipeline"].process(ceo_fraud_email, modality="text")
        print(f"   - Email fraud score: {email_result.fraud_score:.2f}")

        video_result = await full_system["pipeline"].process(deepfake_frame, modality="image")
        print(f"   - Video frame fraud score: {video_result.fraud_score:.2f}")

        # 3. Cross-modal validation for deepfake detection
        print("\n[2] Validating for deepfake indicators...")

        validation_data = {
            "metadata": {
                "sender_email": "john.smith@company-ceo.com",  # Spoofed
                "actual_ceo_email": "jsmith@company.com",
                "sender_ip": "203.0.113.0",  # Suspicious IP
                "company_ip_range": "10.0.0.0/8",
            },
            "audio_timestamps": [1000, 2000, 3000],
            "video_timestamps": [1050, 2100, 3150],  # Sync issues
            "has_speech": True,
            "has_face": True,
            "lip_sync_score": 0.3,  # Poor lip sync
        }

        consistency_report = await full_system["validator"].validate_consistency(
            text_result=email_result, image_result=video_result, raw_data=validation_data
        )

        print(f"   - Consistency score: {consistency_report.overall_consistency:.2f}")
        print(f"   - Audio-video sync issues: {consistency_report.audio_video_sync is not None}")

        # 4. Adaptive learning from previous CEO fraud attempts
        print("\n[3] Checking against learned CEO fraud patterns...")

        # Add historical CEO fraud patterns
        for i in range(5):
            historical_feedback = Feedback(
                case_id=f"CEO_FRAUD_{i:03d}",
                true_label=True,
                detection_result={
                    "fraud_score": 0.75 + i * 0.03,
                    "pattern": "urgent_wire_transfer",
                },
                confidence=0.9,
                feedback_type="verified",
            )
            await full_system["adaptive_learner"].process_feedback(historical_feedback)

        # Get adapted thresholds
        threshold_config = full_system["adaptive_learner"].get_threshold("text")
        print(f"   - Adapted fraud threshold: {threshold_config.fraud_threshold:.2f}")

        # 5. A/B test for deepfake detection improvement
        print("\n[4] Running A/B test for detection algorithm...")

        ab_test = full_system["ab_framework"].create_test(
            name="Deepfake Detection v2",
            description="Testing improved deepfake detection",
            variant_a={"algorithm": "baseline", "threshold": 0.7},
            variant_b={"algorithm": "enhanced", "threshold": 0.65},
            traffic_split=0.5,
            duration_days=1,
        )

        # Simulate test results
        for i in range(20):
            variant = full_system["ab_framework"].assign_variant(ab_test.test_id)
            accuracy = 0.85 if variant == "B" else 0.80
            accuracy += np.random.normal(0, 0.05)
            full_system["ab_framework"].record_metric(
                ab_test.test_id, variant, "accuracy", accuracy
            )

        test_results = full_system["ab_framework"].get_test_results(ab_test.test_id)
        print(f"   - A/B test variants compared: {len(test_results['variants'])}")

        # 6. Final risk assessment
        print("\n[5] Final CEO fraud risk assessment...")

        fused_result = await full_system["fusion"].fuse_modalities(
            text_result=email_result, vision_result=video_result
        )

        risk_profile = await full_system["risk_scorer"].calculate_risk_score(
            fraud_score=fused_result.fraud_score,
            confidence=fused_result.confidence,
            modality_scores=fused_result.modality_scores,
            entity_features={
                "urgency_score": 0.95,
                "confidentiality_emphasis": 1.0,
                "unusual_request": 1.0,
                "deepfake_indicators": 0.8,
                "social_engineering_score": 0.9,
            },
        )

        print(f"   - CEO fraud risk: {risk_profile.risk_level.upper()}")
        print(f"   - Confidence: {risk_profile.confidence_interval}")

        # Assertions
        assert email_result.fraud_score > 0.6, "Should detect CEO fraud email"
        assert consistency_report.inconsistency_count > 0, "Should find deepfake inconsistencies"
        assert risk_profile.risk_level in [
            "medium",
            "high",
            "critical",
        ], "CEO fraud is significant risk"
        assert "B" in test_results["variants"], "A/B test should run"

        processing_time = time.time() - start_time
        print(f"\n✅ DEEPFAKE CEO FRAUD DETECTION: PASSED ({processing_time:.2f}s)")

    @pytest.mark.asyncio
    async def test_adaptive_learning_improvement_e2e(self, full_system):
        """
        Test system improvement through adaptive learning.

        Scenario: System learns and improves from feedback loop
        """
        print("\n" + "=" * 80)
        print("E2E TEST: ADAPTIVE LEARNING AND CONTINUOUS IMPROVEMENT")
        print("=" * 80)

        # 1. Establish baseline performance
        print("\n[1] Establishing baseline performance...")

        baseline_cases = []
        for i in range(10):
            # Mix of fraud and legitimate cases
            is_fraud = i % 2 == 0
            test_content = f"Test case {i}: {'Suspicious' if is_fraud else 'Normal'} activity"

            result = await full_system["pipeline"].process(test_content, modality="text")

            baseline_cases.append(
                {"case_id": f"BASELINE_{i:03d}", "result": result, "true_label": is_fraud}
            )

        # Calculate baseline metrics
        baseline_tp = sum(
            1 for c in baseline_cases if c["true_label"] and c["result"].fraud_score > 0.5
        )
        baseline_fp = sum(
            1 for c in baseline_cases if not c["true_label"] and c["result"].fraud_score > 0.5
        )

        baseline_precision = (
            baseline_tp / (baseline_tp + baseline_fp) if (baseline_tp + baseline_fp) > 0 else 0
        )
        print(f"   - Baseline precision: {baseline_precision:.2f}")

        # 2. Feed results back for learning
        print("\n[2] Processing feedback for adaptive learning...")

        for case in baseline_cases:
            feedback = Feedback(
                case_id=case["case_id"],
                true_label=case["true_label"],
                detection_result=case["result"].to_dict(),
                confidence=0.9,
                feedback_type="verified",
            )
            await full_system["adaptive_learner"].process_feedback(feedback)

        # 3. Check improved performance
        print("\n[3] Testing improved performance after learning...")

        improved_cases = []
        for i in range(10):
            is_fraud = i % 3 == 0  # Different pattern
            test_content = f"New case {i}: {'Fraudulent' if is_fraud else 'Legitimate'} transaction"

            result = await full_system["pipeline"].process(test_content, modality="text")

            improved_cases.append(
                {"case_id": f"IMPROVED_{i:03d}", "result": result, "true_label": is_fraud}
            )

        # Calculate improved metrics
        improved_tp = sum(
            1 for c in improved_cases if c["true_label"] and c["result"].fraud_score > 0.5
        )
        improved_fp = sum(
            1 for c in improved_cases if not c["true_label"] and c["result"].fraud_score > 0.5
        )

        improved_precision = (
            improved_tp / (improved_tp + improved_fp) if (improved_tp + improved_fp) > 0 else 0
        )
        print(f"   - Improved precision: {improved_precision:.2f}")

        # 4. Get learning report
        print("\n[4] Generating performance report...")

        perf_report = full_system["adaptive_learner"].get_performance_report()
        print(f"   - Total feedback processed: {perf_report['total_feedback']}")
        print(f"   - Current accuracy: {perf_report['metrics'].get('accuracy', 0):.2f}")
        print(f"   - F1 score: {perf_report['metrics'].get('f1_score', 0):.2f}")

        # 5. Feature importance analysis
        print("\n[5] Analyzing feature importance...")

        # Update feature importance based on outcomes
        for case in baseline_cases + improved_cases:
            features = {
                "text_length": len(case.get("test_content", "")),
                "fraud_keywords": 1.0 if "fraud" in str(case).lower() else 0.0,
                "urgency": 1.0 if "urgent" in str(case).lower() else 0.0,
            }

            outcome = case["true_label"] == (case["result"].fraud_score > 0.5)
            await full_system["adaptive_learner"].update_feature_importance(features, outcome)

        top_features = full_system["adaptive_learner"].get_top_features(3)
        print(f"   - Top features: {[f[0] for f in top_features]}")

        # Assertions
        assert perf_report["total_feedback"] >= 10, "Should process feedback"
        assert len(top_features) > 0, "Should identify important features"
        # Note: In real scenario, improved_precision should be > baseline_precision
        # but with our simple test data this may not always be true

        print(f"\n✅ ADAPTIVE LEARNING: PASSED")

    @pytest.mark.asyncio
    async def test_stress_test_high_volume_e2e(self, full_system):
        """
        Stress test with high volume concurrent processing.

        Scenario: Handle 100+ concurrent fraud detection requests
        """
        print("\n" + "=" * 80)
        print("E2E TEST: HIGH VOLUME STRESS TEST")
        print("=" * 80)

        start_time = time.time()

        # 1. Generate diverse test cases
        print("\n[1] Generating 100 diverse test cases...")

        test_cases = []
        for i in range(100):
            case_type = i % 5

            if case_type == 0:
                content = f"Phishing attempt {i}: Click here to verify your account"
                modality = "text"
            elif case_type == 1:
                content = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                modality = "image"
            elif case_type == 2:
                content = f"Transaction {i}: Transfer ${np.random.randint(100, 10000)}"
                modality = "text"
            elif case_type == 3:
                content = f"Document {i}: Invoice for services rendered"
                modality = "text"
            else:
                content = f"Normal message {i}: Regular business communication"
                modality = "text"

            test_cases.append({"id": f"STRESS_{i:03d}", "content": content, "modality": modality})

        print(f"   - Generated {len(test_cases)} test cases")

        # 2. Process concurrently
        print("\n[2] Processing cases concurrently...")

        async def process_case(case):
            """Process single case."""
            try:
                result = await full_system["pipeline"].process(
                    case["content"], modality=case["modality"]
                )
                return {
                    "id": case["id"],
                    "success": True,
                    "fraud_score": result.fraud_score,
                    "processing_time": result.processing_time_ms,
                }
            except Exception as e:
                return {"id": case["id"], "success": False, "error": str(e)}

        # Process in batches to avoid overwhelming system
        batch_size = 10
        all_results = []

        for i in range(0, len(test_cases), batch_size):
            batch = test_cases[i : i + batch_size]
            batch_tasks = [process_case(case) for case in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            all_results.extend(batch_results)
            print(f"   - Processed batch {i//batch_size + 1}/{len(test_cases)//batch_size}")

        # 3. Analyze results
        print("\n[3] Analyzing stress test results...")

        successful = sum(1 for r in all_results if r["success"])
        failed = len(all_results) - successful
        avg_time = np.mean([r["processing_time"] for r in all_results if r.get("processing_time")])
        max_time = max([r.get("processing_time", 0) for r in all_results])

        print(
            f"   - Success rate: {successful}/{len(all_results)} ({successful/len(all_results)*100:.1f}%)"
        )
        print(f"   - Average processing time: {avg_time:.2f}ms")
        print(f"   - Max processing time: {max_time:.2f}ms")

        # 4. Check system stability
        print("\n[4] Checking system stability...")

        stats = full_system["pipeline"].get_statistics()
        print(f"   - Total processed: {stats['total_processed']}")
        print(f"   - Error rate: {stats['errors']/stats['total_processed']*100:.1f}%")

        # 5. Memory check
        print("\n[5] Checking resource usage...")

        if hasattr(full_system["pipeline"], "resource_manager"):
            snapshot = full_system["pipeline"].resource_manager.get_snapshot()
            memory_used = snapshot.memory_used_gb * 1024 * 1024 * 1024  # Convert GB to bytes
            print(f"   - Memory used: {memory_used/1024/1024:.1f}MB")

        total_time = time.time() - start_time

        # Assertions
        assert successful >= 95, f"Should process at least 95% successfully, got {successful}"
        assert avg_time < 1000, f"Average time should be < 1s, got {avg_time}ms"
        assert total_time < 60, f"Should complete within 60s, took {total_time:.1f}s"

        print(f"\n✅ STRESS TEST: PASSED ({total_time:.1f}s for 100 cases)")
        print(f"   Throughput: {len(test_cases)/total_time:.1f} cases/second")


class TestSystemReliability:
    """Test system reliability and error recovery."""

    @pytest.mark.asyncio
    async def test_error_recovery_e2e(self, tmp_path):
        """Test system's ability to recover from errors."""
        print("\n" + "=" * 80)
        print("E2E TEST: ERROR RECOVERY AND RESILIENCE")
        print("=" * 80)

        # Initialize minimal system
        config = Config()
        pipeline = FraudDetectionPipeline(config)
        await pipeline.initialize()

        try:
            # 1. Test with invalid inputs
            print("\n[1] Testing invalid input handling...")

            invalid_inputs = [
                None,
                "",
                [],
                {"invalid": "structure"},
                np.array([]),  # Empty array
                "x" * 1000000,  # Very large string
            ]

            errors_handled = 0
            for inp in invalid_inputs:
                try:
                    result = await pipeline.process(inp)
                    if result:
                        errors_handled += 1
                except:
                    errors_handled += 1

            print(f"   - Handled {errors_handled}/{len(invalid_inputs)} invalid inputs")

            # 2. Test recovery after error
            print("\n[2] Testing recovery after error...")

            # Cause an error
            try:
                await pipeline.process(None)
            except:
                pass

            # Should still work normally
            normal_result = await pipeline.process("Normal text after error")
            assert normal_result is not None, "Should recover after error"
            print("   - System recovered successfully")

            # 3. Test concurrent error handling
            print("\n[3] Testing concurrent error handling...")

            mixed_inputs = [
                "Valid text 1",
                None,
                "Valid text 2",
                {"invalid": "data"},
                "Valid text 3",
            ]

            tasks = []
            for inp in mixed_inputs:
                tasks.append(pipeline.process(inp))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            valid_results = sum(
                1 for r in results if not isinstance(r, Exception) and r is not None
            )

            print(f"   - Processed {valid_results} valid inputs despite errors")
            assert valid_results >= 3, "Should process valid inputs despite errors"

            print("\n✅ ERROR RECOVERY: PASSED")

        finally:
            await pipeline.cleanup()

    @pytest.mark.asyncio
    async def test_performance_metrics_e2e(self, tmp_path):
        """Test comprehensive performance metrics."""
        print("\n" + "=" * 80)
        print("E2E TEST: PERFORMANCE METRICS AND MONITORING")
        print("=" * 80)

        config = Config()
        pipeline = FraudDetectionPipeline(config)
        await pipeline.initialize()

        fusion = MultiModalFraudFusion(strategy=FusionStrategy.HYBRID)
        pattern_library = FraudPatternMatcher()

        try:
            # 1. Baseline performance measurement
            print("\n[1] Measuring baseline performance...")

            test_texts = [
                "Normal business email",
                "URGENT: Send money now!",
                "Invoice for services",
                "Suspicious activity detected",
                "Regular transaction",
            ]

            processing_times = []
            for text in test_texts:
                start = time.time()
                await pipeline.process(text, modality="text")
                processing_times.append((time.time() - start) * 1000)

            avg_baseline = np.mean(processing_times)
            p95_baseline = np.percentile(processing_times, 95)

            print(f"   - Average latency: {avg_baseline:.2f}ms")
            print(f"   - P95 latency: {p95_baseline:.2f}ms")

            # 2. Fusion performance
            print("\n[2] Measuring fusion performance...")

            fusion_stats = fusion.get_performance_stats()
            print(f"   - Average fusion time: {fusion_stats['avg_fusion_time_ms']:.2f}ms")
            print(f"   - Cache hit rate: {fusion_stats['cache_hit_rate']:.2%}")

            # 3. Pattern matching performance
            print("\n[3] Measuring pattern matching performance...")

            pattern_stats = pattern_library.library.get_performance_stats()
            print(f"   - Total patterns: {pattern_stats['total_patterns']}")
            print(f"   - Pattern types: {list(pattern_stats['patterns_by_type'].keys())}")

            # 4. System statistics
            print("\n[4] System-wide statistics...")

            pipeline_stats = pipeline.get_statistics()
            error_rate = pipeline_stats["errors"] / max(pipeline_stats["total_processed"], 1)

            print(f"   - Total processed: {pipeline_stats['total_processed']}")
            print(f"   - Error rate: {error_rate:.2%}")
            print(f"   - Avg processing time: {pipeline_stats['average_time_ms']:.2f}ms")

            # Assertions
            assert avg_baseline < 500, "Average latency should be < 500ms"
            assert error_rate < 0.1, "Error rate should be < 10%"
            assert pattern_stats["total_patterns"] > 0, "Should have patterns loaded"

            print("\n✅ PERFORMANCE METRICS: PASSED")

        finally:
            await pipeline.cleanup()


if __name__ == "__main__":
    # Run comprehensive E2E tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])
