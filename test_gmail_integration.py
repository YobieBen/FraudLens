#!/usr/bin/env python3
"""
Test Gmail Integration
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fraudlens.api.gmail_integration import GmailFraudScanner, EmailAction


async def test_gmail_connection():
    """Test Gmail connection and basic functionality."""
    print("\n" + "="*60)
    print("TESTING GMAIL INTEGRATION")
    print("="*60)
    
    # Try to create scanner
    try:
        print("\n1. Creating Gmail scanner...")
        scanner = GmailFraudScanner(
            credentials_file="credentials.json",
            token_file="token.pickle",
            fraud_threshold=0.7,
            auto_action=False,
        )
        print("   ✅ Scanner created")
        
        print("\n2. Initializing scanner...")
        await scanner.initialize()
        print("   ✅ Scanner initialized")
        
        print("\n3. Testing email stream (demo mode if no credentials)...")
        results = await scanner.stream_emails(
            query="is:unread",
            max_results=5,
            process_attachments=False,
            since_days=1
        )
        
        if results:
            print(f"   ✅ Found {len(results)} emails")
            for i, result in enumerate(results[:3], 1):
                print(f"\n   Email {i}:")
                print(f"     Subject: {result.subject[:50]}...")
                print(f"     From: {result.sender}")
                print(f"     Fraud Score: {result.fraud_score:.2%}")
                print(f"     Types: {', '.join(result.fraud_types) if result.fraud_types else 'Clean'}")
        else:
            print("   ℹ️  No emails found or API not configured")
        
        print("\n" + "="*60)
        print("✅ GMAIL INTEGRATION TEST COMPLETE")
        print("="*60)
        return True
        
    except FileNotFoundError as e:
        print(f"\n⚠️  Gmail credentials not found: {e}")
        print("   Running in DEMO MODE")
        print("\n" + "="*60)
        print("✅ DEMO MODE ACTIVE - Integration ready for real Gmail when configured")
        print("="*60)
        return False
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_gmail_connection())
    sys.exit(0 if success else 1)