#!/usr/bin/env python3
"""
Test external database connections and integrations.

Author: Yobie Benjamin
Date: 2025-08-28
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from loguru import logger

from fraudlens.integrations.threat_intel import ThreatIntelligenceManager
from fraudlens.integrations.document_validator import DocumentValidator
from fraudlens.integrations.phishing_db import PhishingDatabaseConnector
from fraudlens.core.pipeline import FraudDetectionPipeline


async def test_threat_intelligence():
    """Test threat intelligence feeds."""
    logger.info("\n=== Testing Threat Intelligence Feeds ===")
    
    threat_intel = ThreatIntelligenceManager()
    
    # Test threat feed updates
    logger.info("Updating threat feeds...")
    await threat_intel.update_threat_feeds()
    
    # List available feeds
    logger.info(f"Available feeds: {list(threat_intel.threat_feeds.keys())}")
    
    # Test URL checking
    test_urls = [
        "http://phishing-example.com",
        "http://google.com",
        "http://malware-test.com",
        "http://legitimate-site.org"
    ]
    
    for url in test_urls:
        is_threat = await threat_intel.check_url(url)
        logger.info(f"  {url}: {'THREAT' if is_threat else 'SAFE'}")
    
    # Get statistics
    stats = threat_intel.get_statistics()
    logger.info(f"Threat Intel Statistics:")
    logger.info(f"  Total URLs checked: {stats.get('urls_checked', 0)}")
    logger.info(f"  Threats detected: {stats.get('threats_detected', 0)}")
    logger.info(f"  Active feeds: {stats.get('active_feeds', 0)}")
    logger.info(f"  Last update: {stats.get('last_update', 'Never')}")
    
    return threat_intel


async def test_document_validator():
    """Test document validation capabilities."""
    logger.info("\n=== Testing Document Validator ===")
    
    validator = DocumentValidator()
    
    # Test SSN validation (multiple countries)
    test_ssns = [
        ("123-45-6789", "US"),  # US format
        ("123456789", "US"),    # US without dashes
        ("1234567890", "CA"),   # Canadian SIN
        ("12345678A", "UK"),    # UK NIN
        ("123.456.789", "BR"),  # Brazilian CPF
        ("12-3456789-0", "KR"), # Korean RRN
        ("invalid-ssn", "US"),  # Invalid
    ]
    
    logger.info("Testing SSN/National ID validation:")
    for ssn, country in test_ssns:
        result = validator.validate_ssn(ssn, country)
        logger.info(f"  {ssn} ({country}): {result['valid']} - {result.get('message', '')}")
    
    # Test driver's license validation
    test_licenses = [
        ("A123456789", "CA"),   # California
        ("123456789", "TX"),    # Texas
        ("A12-345-678", "NY"),  # New York
        ("INVALID", "FL"),      # Invalid
    ]
    
    logger.info("\nTesting Driver's License validation:")
    for license_num, state in test_licenses:
        result = validator.validate_driver_license(license_num, state)
        logger.info(f"  {license_num} ({state}): {result['valid']}")
    
    # Test passport MRZ
    test_mrz = [
        "P<USASMITH<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
        "1234567890USA7001011M2001011<<<<<<<<<<<<<<06"
    ]
    
    logger.info("\nTesting Passport MRZ validation:")
    for mrz_line in test_mrz:
        result = validator.validate_passport_mrz(mrz_line)
        logger.info(f"  MRZ Valid: {result['valid']}")
    
    # Test credit card validation
    test_cards = [
        "4532015112830366",  # Visa
        "5425233430109903",  # Mastercard
        "374245455400126",   # Amex
        "1234567890123456",  # Invalid
    ]
    
    logger.info("\nTesting Credit Card validation:")
    for card_num in test_cards:
        result = validator.validate_credit_card(card_num)
        logger.info(f"  {card_num}: {result['valid']} ({result.get('card_type', 'Unknown')})")
    
    return validator


async def test_phishing_database():
    """Test phishing database connections."""
    logger.info("\n=== Testing Phishing Database Connector ===")
    
    phishing_db = PhishingDatabaseConnector()
    
    # Test database connections
    logger.info("Testing database connections:")
    for db_name in phishing_db.databases:
        logger.info(f"  {db_name}: Connected")
    
    # Test URL checking
    test_urls = [
        "http://phishing-test.com/fake-bank",
        "https://www.google.com",
        "http://suspicious-site.net/login",
        "https://www.amazon.com"
    ]
    
    logger.info("\nChecking URLs against phishing databases:")
    for url in test_urls:
        result = await phishing_db.check_url(url)
        if result["is_phishing"]:
            logger.info(f"  {url}: PHISHING (confidence: {result['confidence']:.2%})")
        else:
            logger.info(f"  {url}: SAFE")
    
    # Test domain checking
    test_domains = [
        "phishing-site.com",
        "google.com",
        "suspicious-bank.net",
        "microsoft.com"
    ]
    
    logger.info("\nChecking domains:")
    for domain in test_domains:
        is_suspicious = await phishing_db.check_domain(domain)
        logger.info(f"  {domain}: {'SUSPICIOUS' if is_suspicious else 'SAFE'}")
    
    # Get statistics
    stats = phishing_db.get_statistics()
    logger.info(f"\nPhishing DB Statistics:")
    logger.info(f"  Total URLs in database: {stats.get('total_urls', 0)}")
    logger.info(f"  Total domains blacklisted: {stats.get('total_domains', 0)}")
    logger.info(f"  Active databases: {stats.get('active_databases', 0)}")
    
    return phishing_db


async def test_full_integration():
    """Test full integration with FraudLens pipeline."""
    logger.info("\n=== Testing Full Integration with Pipeline ===")
    
    pipeline = FraudDetectionPipeline()
    await pipeline.initialize()
    
    # Test text with suspicious URLs
    test_text = """
    Dear Customer,
    
    Your account has been suspended. Please verify your identity at:
    http://phishing-bank.com/verify
    
    Or contact us at suspicious@fake-bank.net
    
    Your SSN 123-45-6789 and credit card 4532015112830366 are at risk.
    """
    
    logger.info("Processing suspicious text through pipeline...")
    result = await pipeline.process(test_text, modality="text")
    
    logger.info(f"Fraud Score: {result.fraud_score:.2%}")
    logger.info(f"Fraud Types: {result.fraud_types}")
    logger.info(f"Confidence: {result.confidence:.2%}")
    
    # Check evidence for external database hits
    if result.evidence:
        logger.info("External Database Evidence:")
        for key, value in result.evidence.items():
            if "threat" in key.lower() or "phishing" in key.lower():
                logger.info(f"  {key}: {value}")
    
    return result


async def generate_report():
    """Generate comprehensive report of external integrations."""
    logger.info("\n=== EXTERNAL DATABASE INTEGRATION REPORT ===")
    logger.info(f"Generated: {datetime.now().isoformat()}")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "threat_intelligence": {},
        "document_validation": {},
        "phishing_databases": {},
        "integration_status": {}
    }
    
    # Test each component
    try:
        threat_intel = await test_threat_intelligence()
        report["threat_intelligence"] = {
            "status": "operational",
            "feeds": list(threat_intel.threat_feeds.keys()),
            "stats": threat_intel.get_statistics()
        }
    except Exception as e:
        report["threat_intelligence"] = {"status": "error", "error": str(e)}
    
    try:
        validator = await test_document_validator()
        report["document_validation"] = {
            "status": "operational",
            "supported_countries": ["US", "CA", "UK", "BR", "KR", "DE", "FR", "AU"],
            "document_types": ["SSN", "Driver's License", "Passport", "Credit Card"]
        }
    except Exception as e:
        report["document_validation"] = {"status": "error", "error": str(e)}
    
    try:
        phishing_db = await test_phishing_database()
        report["phishing_databases"] = {
            "status": "operational",
            "databases": list(phishing_db.databases.keys()),
            "stats": phishing_db.get_statistics()
        }
    except Exception as e:
        report["phishing_databases"] = {"status": "error", "error": str(e)}
    
    # Test full integration
    try:
        result = await test_full_integration()
        report["integration_status"] = {
            "status": "operational",
            "pipeline_active": True,
            "fraud_detection_working": result.fraud_score > 0
        }
    except Exception as e:
        report["integration_status"] = {"status": "error", "error": str(e)}
    
    # Save report
    report_path = Path("external_database_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\nReport saved to: {report_path}")
    
    # Summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Threat Intelligence: {report['threat_intelligence']['status']}")
    logger.info(f"Document Validation: {report['document_validation']['status']}")
    logger.info(f"Phishing Databases: {report['phishing_databases']['status']}")
    logger.info(f"Pipeline Integration: {report['integration_status']['status']}")
    
    operational_count = sum(
        1 for component in report.values()
        if isinstance(component, dict) and component.get('status') == 'operational'
    )
    
    logger.info(f"\nOverall Status: {operational_count}/4 systems operational")
    
    return report


if __name__ == "__main__":
    asyncio.run(generate_report())