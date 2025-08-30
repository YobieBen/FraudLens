#!/usr/bin/env python3
"""
Test Gmail IMAP SSL connection with multiple fallback methods
"""

import ssl
import imaplib
import certifi
import sys
import os

def test_gmail_ssl_connection():
    """Test SSL connection to Gmail IMAP server with different methods."""
    
    print("=" * 60)
    print("Gmail IMAP SSL Connection Test")
    print("=" * 60)
    print()
    
    # Get credentials
    email = input("Enter your Gmail address: ")
    app_password = input("Enter your app password: ")
    
    print("\nTesting SSL connection methods...")
    print("-" * 40)
    
    # Method 1: Certifi certificates
    print("\n1. Testing with certifi certificates...")
    try:
        context = ssl.create_default_context(cafile=certifi.where())
        imap = imaplib.IMAP4_SSL("imap.gmail.com", 993, ssl_context=context)
        imap.login(email, app_password)
        print("✅ SUCCESS: Connected with certifi certificates")
        imap.logout()
        return True
    except ssl.SSLError as e:
        print(f"❌ SSL Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Method 2: System default certificates
    print("\n2. Testing with system certificates...")
    try:
        context = ssl.create_default_context()
        imap = imaplib.IMAP4_SSL("imap.gmail.com", 993, ssl_context=context)
        imap.login(email, app_password)
        print("✅ SUCCESS: Connected with system certificates")
        imap.logout()
        return True
    except ssl.SSLError as e:
        print(f"❌ SSL Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Method 3: Unverified context (less secure)
    print("\n3. Testing with unverified context (less secure)...")
    try:
        context = ssl._create_unverified_context()
        imap = imaplib.IMAP4_SSL("imap.gmail.com", 993, ssl_context=context)
        imap.login(email, app_password)
        print("⚠️  SUCCESS: Connected with unverified context")
        print("   Note: This method bypasses SSL verification")
        imap.logout()
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("❌ All connection methods failed")
    print("\nTroubleshooting tips:")
    print("1. Ensure 2-factor authentication is enabled on your Gmail")
    print("2. Generate an app-specific password at:")
    print("   https://myaccount.google.com/apppasswords")
    print("3. Check your internet connection")
    print("4. Try updating certifi: pip install --upgrade certifi")
    print("=" * 60)
    
    return False

if __name__ == "__main__":
    # Set environment variables for SSL
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    
    success = test_gmail_ssl_connection()
    sys.exit(0 if success else 1)