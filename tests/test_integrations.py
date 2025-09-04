"""
Test external database integrations.

Author: Yobie Benjamin
Date: 2025-08-28
"""

import asyncio

from fraudlens.core.pipeline import FraudDetectionPipeline
from fraudlens.integrations import (
    DocumentValidator,
    PhishingDatabaseConnector,
    ThreatIntelligenceManager,
)


async def test_document_validation():
    """Test document validation capabilities."""
    print("\n=== Testing Document Validation ===")

    validator = DocumentValidator()

    # Test US SSN validation
    test_ssns = [
        ("123-45-6789", False),  # Invalid (known fake)
        ("000-12-3456", False),  # Invalid area
        ("123-00-4567", False),  # Invalid group
        ("987-65-4321", True),  # Valid format (not checking if issued)
    ]

    for ssn, expected_valid in test_ssns:
        result = validator.validate_ssn(ssn, "US")
        status = "✓" if (result["valid"] == expected_valid) else "✗"
        print(f"{status} SSN {ssn}: {result.get('error', 'Valid')}")

    # Test credit card validation (Luhn algorithm)
    test_cards = [
        ("4532015112830366", True),  # Valid Visa
        ("5425233430109903", True),  # Valid Mastercard
        ("374245455400126", True),  # Valid Amex
        ("4532015112830367", False),  # Invalid (bad check digit)
        ("1234567812345678", False),  # Invalid
    ]

    for card, expected_valid in test_cards:
        result = validator.validate_credit_card(card)
        status = "✓" if (result["valid"] == expected_valid) else "✗"
        card_type = result.get("type", "Unknown")
        print(f"{status} Card {card[:4]}...{card[-4:]}: {card_type}")

    # Test driver's license validation
    test_licenses = [
        ("A1234567", "CA", True),  # Valid California format
        ("123456789", "TX", False),  # Invalid Texas (should be 7-8 digits)
        ("S12345678", "MA", True),  # Valid Massachusetts
    ]

    for license, state, expected_valid in test_licenses:
        result = validator.validate_driver_license(license, state)
        status = "✓" if (result["valid"] == expected_valid) else "✗"
        print(f"{status} License {license} ({state}): {result.get('error', 'Valid')}")

    # Test passport MRZ
    mrz_lines = [
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
        "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
    ]

    result = validator.validate_passport_mrz(mrz_lines)
    print(f"Passport MRZ: {'Valid' if result['valid'] else 'Invalid'}")

    # Test international documents
    print("\n--- International Documents ---")

    # Canadian SIN (uses Luhn)
    result = validator.validate_ssn("046-454-286", "CA")
    print(f"Canadian SIN: {'Valid' if result['valid'] else result.get('error', 'Invalid')}")

    # UK NINO
    result = validator.validate_ssn("AB123456C", "UK")
    print(f"UK NINO: {'Valid' if result['valid'] else result.get('error', 'Invalid')}")

    # Indian PAN
    result = validator.validate_document("ABCDE1234F", "IN_PAN")
    print(f"Indian PAN: {'Valid' if result['valid'] else result.get('error', 'Invalid')}")

    print(
        f"\nSupported documents: {len(validator.get_supported_documents()['social_security'])} countries"
    )


async def test_phishing_detection():
    """Test phishing database integration."""
    print("\n=== Testing Phishing Detection ===")

    phishing_db = PhishingDatabaseConnector()

    test_urls = [
        "https://www.paypal.com",  # Legitimate
        "https://paypal-secure.fake.com",  # Impersonation
        "http://192.168.1.1/login",  # IP address
        "https://bit.ly/3xY9zK2",  # URL shortener
        "https://amaz0n.com",  # Typosquatting
        "https://microsoft-verify.tk",  # Suspicious TLD
        "https://www.google.com",  # Legitimate
        "https://pаypal.com",  # Homograph (Cyrillic 'a')
    ]

    for url in test_urls:
        result = await phishing_db.check_url(url)
        threat_level = result["threat_level"]
        confidence = result["confidence"]
        status = "⚠️" if result["is_phishing"] else "✓"

        print(f"{status} {url[:50]}")
        print(f"   Threat: {threat_level} (confidence: {confidence:.1%})")
        if result["threats"]:
            print(f"   Issues: {', '.join(result['threats'][:2])}")
        if result.get("target_brand"):
            print(f"   Target: {result['target_brand']} impersonation")

    # Test email checking
    print("\n--- Email Validation ---")

    test_emails = [
        "john.doe@gmail.com",  # Legitimate
        "admin123456@tempmail.com",  # Disposable
        "security-alert-9876543@fake.com",  # Suspicious pattern
        "support@microsoft.com",  # Legitimate (but check context)
    ]

    threat_intel = ThreatIntelligenceManager()

    for email in test_emails:
        result = await threat_intel.check_email(email)
        threat_score = result["threat_score"]
        status = "⚠️" if threat_score > 0.5 else "✓"

        print(f"{status} {email}: Score {threat_score:.1%}")
        if result["threats"]:
            print(f"   Issues: {', '.join(result['threats'])}")


async def test_threat_intelligence():
    """Test threat intelligence integration."""
    print("\n=== Testing Threat Intelligence ===")

    threat_intel = ThreatIntelligenceManager()
    await threat_intel.initialize()

    # Check URL threats
    test_data = [
        ("https://malicious.example.com/phishing", "url"),
        ("evil-domain.tk", "domain"),
        ("192.168.1.100", "ip"),
        ("d41d8cd98f00b204e9800998ecf8427e", "hash"),  # MD5 of empty file
    ]

    for data, data_type in test_data:
        if data_type == "url":
            result = await threat_intel.check_url(data)
            print(f"URL Check: {data[:30]}")
            print(f"   Threat Score: {result['threat_score']:.1%}")
            print(f"   Threats: {result.get('threats', [])}")
        elif data_type == "ip":
            result = await threat_intel.check_ip(data)
            print(f"IP Check: {data}")
            print(f"   Threat Score: {result['threat_score']:.1%}")
        elif data_type == "hash":
            result = await threat_intel.check_hash(data)
            print(f"Hash Check: {data[:16]}...")
            print(f"   Malicious: {result['malicious']}")

    summary = await threat_intel.get_threat_summary()
    print(f"\nThreat Intelligence Summary:")
    print(f"   Known bad URLs: {summary['known_bad_urls']}")
    print(f"   Known bad domains: {summary['known_bad_domains']}")
    print(f"   Known bad IPs: {summary['known_bad_ips']}")
    print(f"   Active feeds: {summary['feeds_configured']}")


async def test_pipeline_integration():
    """Test full pipeline with integrations."""
    print("\n=== Testing Pipeline Integration ===")

    pipeline = FraudDetectionPipeline()
    await pipeline.initialize()

    # Test document validation through pipeline
    print("\n--- Document Validation via Pipeline ---")
    result = await pipeline.validate_document("123-45-6789", "ssn")
    print(f"SSN Validation: {'Valid' if result['valid'] else 'Invalid'}")
    print(f"   Fraud Score: {result['fraud_score']:.1%}")

    # Test URL threat checking through pipeline
    print("\n--- URL Threat Check via Pipeline ---")
    result = await pipeline.check_url_threat("https://paypal-verify.suspicious.com")
    print(f"URL: {result['url']}")
    print(f"   Malicious: {result['is_malicious']}")
    print(f"   Threat Score: {result['threat_score']:.1%}")
    if result["recommendations"]:
        print(f"   Recommendations: {result['recommendations'][0]}")

    # Test email threat checking
    print("\n--- Email Threat Check via Pipeline ---")
    result = await pipeline.check_email_threat(
        "security@tempmail.com",
        "Your account has been suspended. Click here to verify: http://bit.ly/verify",
    )
    print(f"Email: {result['email']}")
    print(f"   Suspicious: {result['is_suspicious']}")
    print(f"   Fraud Score: {result['fraud_score']:.1%}")
    print(f"   Threats: {', '.join(result['threats'][:2]) if result['threats'] else 'None'}")

    # Test text with URL detection
    print("\n--- Text with URL Enhancement ---")
    text = "Please verify your PayPal account at https://paypal-secure.fake.com"
    result = await pipeline.process(text, modality="text")

    if result:
        print(f"Text Fraud Detection:")
        print(f"   Fraud Score: {result.fraud_score:.0f}%")
        print(f"   Fraud Types: {', '.join(str(ft.value if hasattr(ft, 'value') else ft) for ft in result.fraud_types) if result.fraud_types else 'None'}")

        # Enhance with threat intel
        enhanced = await pipeline.enhance_detection_with_intel(result)
        if enhanced.fraud_score != result.fraud_score:
            print(f"   Enhanced Score: {enhanced.fraud_score:.0f}%")
            print(
                f"   New threats: {[t for t in enhanced.fraud_types if t not in result.fraud_types]}"
            )

    await pipeline.cleanup()
    print("\n✓ Pipeline integration test complete")


async def main():
    """Run all integration tests."""
    print("=" * 60)
    print("FRAUDLENS EXTERNAL DATABASE INTEGRATION TESTS")
    print("=" * 60)

    # Run tests
    await test_document_validation()
    await test_phishing_detection()
    await test_threat_intelligence()
    await test_pipeline_integration()

    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
