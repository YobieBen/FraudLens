#!/usr/bin/env python3
"""
Comprehensive End-to-End Test Suite for FraudLens
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fraudlens.core.pipeline import FraudDetectionPipeline


class ComprehensiveE2ETester:
    """Comprehensive end-to-end tester for FraudLens."""
    
    def __init__(self):
        self.pipeline = None
        self.results = []
        self.test_cases = {
            "text": [
                {
                    "input": "Your account has been compromised. Click here to secure it immediately.",
                    "expected_fraud": True,
                    "expected_types": ["phishing", "social_engineering"],
                    "description": "Classic phishing attempt"
                },
                {
                    "input": "Congratulations! You've won $1,000,000! Click here to claim your prize now!",
                    "expected_fraud": True,
                    "expected_types": ["scam", "phishing"],
                    "description": "Lottery scam"
                },
                {
                    "input": "Dear Sir/Madam, I am a Nigerian prince with $50 million inheritance...",
                    "expected_fraud": True,
                    "expected_types": ["scam"],
                    "description": "Nigerian prince scam"
                },
                {
                    "input": "Please provide your SSN and date of birth to verify your identity.",
                    "expected_fraud": True,
                    "expected_types": ["identity_theft"],
                    "description": "Identity theft attempt"
                },
                {
                    "input": "This is a normal business email about the upcoming meeting on Monday at 2 PM.",
                    "expected_fraud": False,
                    "expected_types": [],
                    "description": "Legitimate business email"
                },
                {
                    "input": "Invoice #12345 for services rendered. Payment due in 30 days. Total: $500.00",
                    "expected_fraud": False,
                    "expected_types": [],
                    "description": "Legitimate invoice"
                }
            ]
        }
        
    async def initialize(self):
        """Initialize the fraud detection pipeline."""
        print("\n" + "="*80)
        print("🚀 Initializing FraudLens Pipeline...")
        print("="*80)
        
        self.pipeline = FraudDetectionPipeline()
        await self.pipeline.initialize()
        
        print("✅ Pipeline initialized successfully\n")
        
    async def test_text_fraud(self):
        """Test text fraud detection."""
        print("\n" + "="*80)
        print("📝 TESTING TEXT FRAUD DETECTION")
        print("="*80)
        
        text_results = []
        
        for i, test_case in enumerate(self.test_cases["text"], 1):
            print(f"\n[Test {i}/{len(self.test_cases['text'])}] {test_case['description']}")
            print(f"Input: {test_case['input'][:100]}...")
            
            start_time = time.time()
            result = await self.pipeline.process(test_case["input"], modality="text")
            processing_time = (time.time() - start_time) * 1000
            
            # Analyze results
            fraud_detected = result.fraud_score > 0.5 if result else False
            fraud_types = []
            
            if result:
                # Extract fraud types from result
                if hasattr(result, 'fraud_types'):
                    fraud_types = [str(ft) for ft in result.fraud_types] if result.fraud_types else []
                
                print(f"  🎯 Fraud Score: {result.fraud_score:.2%}")
                print(f"  📊 Confidence: {result.confidence:.2%}")
                print(f"  🏷️  Types Detected: {fraud_types}")
                print(f"  💬 Explanation: {result.explanation[:200] if result.explanation else 'None'}...")
                print(f"  ⏱️  Processing Time: {processing_time:.2f}ms")
                
                # Check if detection matches expectation
                expected_fraud = test_case["expected_fraud"]
                detection_correct = fraud_detected == expected_fraud
                
                if detection_correct:
                    print(f"  ✅ Detection CORRECT (Expected: {'Fraud' if expected_fraud else 'Clean'})")
                else:
                    print(f"  ❌ Detection WRONG (Expected: {'Fraud' if expected_fraud else 'Clean'}, Got: {'Fraud' if fraud_detected else 'Clean'})")
                
                # Store result
                text_results.append({
                    "test_case": test_case["description"],
                    "input": test_case["input"][:50] + "...",
                    "expected_fraud": expected_fraud,
                    "detected_fraud": fraud_detected,
                    "fraud_score": result.fraud_score,
                    "confidence": result.confidence,
                    "fraud_types": fraud_types,
                    "correct": detection_correct,
                    "processing_time_ms": processing_time
                })
            else:
                print(f"  ❌ ERROR: No result returned")
                text_results.append({
                    "test_case": test_case["description"],
                    "error": "No result returned"
                })
        
        return text_results
    
    async def test_known_fakes(self):
        """Test known fake document detection (McLovin)."""
        print("\n" + "="*80)
        print("🆔 TESTING KNOWN FAKE DETECTION")
        print("="*80)
        
        # Create a mock McLovin document text
        mclovin_text = """
        HAWAII DRIVER LICENSE
        Name: McLovin
        Date of Birth: 06/03/1981
        License Number: 01-47-87695
        Address: 892 Momona St, Honolulu, HI 96820
        """
        
        print("\n[Testing McLovin License Detection]")
        print(f"Input: McLovin fake ID text")
        
        start_time = time.time()
        result = await self.pipeline.process(mclovin_text, modality="text")
        processing_time = (time.time() - start_time) * 1000
        
        if result:
            print(f"  🎯 Fraud Score: {result.fraud_score:.2%}")
            print(f"  📊 Confidence: {result.confidence:.2%}")
            print(f"  🏷️  Types: {result.fraud_types}")
            print(f"  ⏱️  Processing Time: {processing_time:.2f}ms")
            
            if result.fraud_score > 0.9:
                print(f"  ✅ McLovin fake detected successfully!")
            else:
                print(f"  ⚠️  McLovin fake not properly detected (score too low)")
        
        return {"mclovin_detected": result.fraud_score > 0.9 if result else False}
    
    async def test_integrations(self):
        """Test external integrations."""
        print("\n" + "="*80)
        print("🔌 TESTING INTEGRATIONS")
        print("="*80)
        
        integration_results = {}
        
        # Test threat intelligence
        print("\n[Testing Threat Intelligence]")
        try:
            from fraudlens.integrations.threat_intel import ThreatIntelligenceManager
            threat_intel = ThreatIntelligenceManager()
            stats = threat_intel.get_statistics()
            print(f"  ✅ Threat Intelligence: Active ({stats['total_indicators']} indicators)")
            integration_results["threat_intel"] = "active"
        except Exception as e:
            print(f"  ❌ Threat Intelligence: Failed ({str(e)})")
            integration_results["threat_intel"] = "failed"
        
        # Test phishing database
        print("\n[Testing Phishing Database]")
        try:
            from fraudlens.integrations.phishing_db import PhishingDatabaseConnector
            phishing_db = PhishingDatabaseConnector()
            await phishing_db.initialize()
            test_result = await phishing_db.check_domain("example-phishing.com")
            print(f"  ✅ Phishing Database: Active")
            integration_results["phishing_db"] = "active"
        except Exception as e:
            print(f"  ❌ Phishing Database: Failed ({str(e)})")
            integration_results["phishing_db"] = "failed"
        
        # Test document validator
        print("\n[Testing Document Validator]")
        try:
            from fraudlens.integrations.document_validator import DocumentValidator
            doc_validator = DocumentValidator()
            validation = doc_validator.validate_document_structure("driver_license", {})
            print(f"  ✅ Document Validator: Active")
            integration_results["document_validator"] = "active"
        except Exception as e:
            print(f"  ❌ Document Validator: Failed ({str(e)})")
            integration_results["document_validator"] = "failed"
        
        return integration_results
    
    async def generate_report(self, results: Dict[str, Any]):
        """Generate comprehensive test report."""
        print("\n" + "="*80)
        print("📊 TEST REPORT")
        print("="*80)
        
        # Text fraud detection results
        if "text_fraud" in results:
            text_results = results["text_fraud"]
            correct = sum(1 for r in text_results if r.get("correct", False))
            total = len(text_results)
            accuracy = (correct / total) * 100 if total > 0 else 0
            
            print(f"\n📝 Text Fraud Detection:")
            print(f"  • Total Tests: {total}")
            print(f"  • Correct: {correct}")
            print(f"  • Accuracy: {accuracy:.1f}%")
            
            # Show failed cases
            failed = [r for r in text_results if not r.get("correct", False)]
            if failed:
                print(f"\n  ⚠️  Failed Cases:")
                for case in failed:
                    print(f"    - {case['test_case']}: Expected {'Fraud' if case['expected_fraud'] else 'Clean'}, "
                          f"Got {'Fraud' if case.get('detected_fraud', False) else 'Clean'} "
                          f"(Score: {case.get('fraud_score', 0):.2%})")
        
        # Known fakes detection
        if "known_fakes" in results:
            print(f"\n🆔 Known Fakes Detection:")
            print(f"  • McLovin Detection: {'✅ PASSED' if results['known_fakes']['mclovin_detected'] else '❌ FAILED'}")
        
        # Integrations
        if "integrations" in results:
            print(f"\n🔌 Integrations:")
            for integration, status in results["integrations"].items():
                symbol = "✅" if status == "active" else "❌"
                print(f"  • {integration}: {symbol} {status.upper()}")
        
        # Overall summary
        print("\n" + "="*80)
        print("📈 OVERALL SUMMARY")
        print("="*80)
        
        all_passed = True
        
        if "text_fraud" in results:
            text_accuracy = accuracy
            if text_accuracy < 80:
                all_passed = False
                print(f"  ⚠️  Text Detection Accuracy: {text_accuracy:.1f}% (Below 80% threshold)")
            else:
                print(f"  ✅ Text Detection Accuracy: {text_accuracy:.1f}%")
        
        if "known_fakes" in results:
            if not results["known_fakes"]["mclovin_detected"]:
                all_passed = False
                print(f"  ❌ Known Fakes: Failed to detect McLovin")
            else:
                print(f"  ✅ Known Fakes: Detection working")
        
        if "integrations" in results:
            failed_integrations = [k for k, v in results["integrations"].items() if v != "active"]
            if failed_integrations:
                all_passed = False
                print(f"  ⚠️  Integrations: {len(failed_integrations)} failed")
            else:
                print(f"  ✅ Integrations: All active")
        
        print("\n" + "="*80)
        if all_passed:
            print("✅ ALL TESTS PASSED")
        else:
            print("❌ SOME TESTS FAILED - Review issues above")
        print("="*80)
        
        return all_passed
    
    async def run_all_tests(self):
        """Run all comprehensive tests."""
        try:
            await self.initialize()
            
            results = {}
            
            # Run text fraud tests
            results["text_fraud"] = await self.test_text_fraud()
            
            # Run known fakes test
            results["known_fakes"] = await self.test_known_fakes()
            
            # Run integration tests
            results["integrations"] = await self.test_integrations()
            
            # Generate report
            all_passed = await self.generate_report(results)
            
            # Save results to file
            with open("test_results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n📁 Detailed results saved to test_results.json")
            
            return all_passed
            
        except Exception as e:
            print(f"\n❌ ERROR during testing: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """Main test runner."""
    print("\n" + "="*80)
    print("🔬 FRAUDLENS COMPREHENSIVE E2E TEST SUITE")
    print("="*80)
    
    tester = ComprehensiveE2ETester()
    success = await tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())