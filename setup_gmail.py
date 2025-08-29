#!/usr/bin/env python3
"""
Setup script for Gmail OAuth2 authentication
Run this to configure Gmail API access
"""

import os
import json
from pathlib import Path


def setup_gmail_auth():
    """Interactive setup for Gmail authentication."""
    
    print("\n" + "="*60)
    print("FraudLens Gmail Setup")
    print("="*60)
    
    print("\nThis script will help you set up Gmail API access for FraudLens.")
    print("\nYou'll need to:")
    print("1. Enable Gmail API in Google Cloud Console")
    print("2. Create OAuth2 credentials")
    print("3. Download the credentials JSON file")
    
    print("\n" + "-"*60)
    print("Step 1: Enable Gmail API")
    print("-"*60)
    print("\n1. Go to: https://console.cloud.google.com/")
    print("2. Create a new project or select existing")
    print("3. Go to 'APIs & Services' > 'Library'")
    print("4. Search for 'Gmail API' and enable it")
    
    input("\nPress Enter when you've enabled Gmail API...")
    
    print("\n" + "-"*60)
    print("Step 2: Create OAuth2 Credentials")
    print("-"*60)
    print("\n1. Go to 'APIs & Services' > 'Credentials'")
    print("2. Click 'Create Credentials' > 'OAuth client ID'")
    print("3. Choose 'Desktop app' as application type")
    print("4. Name it 'FraudLens Gmail Scanner'")
    print("5. Download the credentials JSON file")
    
    creds_path = input("\nEnter path to downloaded credentials JSON file: ").strip()
    
    if not os.path.exists(creds_path):
        print(f"Error: File not found: {creds_path}")
        return False
    
    # Copy to project directory
    dest_path = Path("credentials.json")
    with open(creds_path, 'r') as src:
        creds_data = json.load(src)
    
    with open(dest_path, 'w') as dst:
        json.dump(creds_data, dst, indent=2)
    
    print(f"\n✅ Credentials saved to {dest_path}")
    
    print("\n" + "-"*60)
    print("Step 3: Configure Settings")
    print("-"*60)
    
    # Get configuration
    config = {
        "fraud_threshold": 0.7,
        "auto_action": False,
        "action_thresholds": {
            "flag": 0.5,
            "spam": 0.7,
            "trash": 0.95,
            "quarantine": 0.8,
        },
        "monitoring": {
            "enabled": False,
            "interval_seconds": 60,
            "query": "is:unread",
        },
    }
    
    print("\nDefault configuration:")
    print(f"  Fraud threshold: {config['fraud_threshold']}")
    print(f"  Auto-action: {config['auto_action']}")
    
    customize = input("\nCustomize settings? (y/n): ").lower() == 'y'
    
    if customize:
        try:
            threshold = float(input("Fraud threshold (0.0-1.0) [0.7]: ") or "0.7")
            config["fraud_threshold"] = max(0.0, min(1.0, threshold))
        except:
            pass
        
        config["auto_action"] = input("Enable auto-action? (y/n) [n]: ").lower() == 'y'
        
        if config["auto_action"]:
            print("\nAction thresholds (0.0-1.0):")
            try:
                config["action_thresholds"]["flag"] = float(
                    input(f"  Flag threshold [{config['action_thresholds']['flag']}]: ") 
                    or config["action_thresholds"]["flag"]
                )
                config["action_thresholds"]["spam"] = float(
                    input(f"  Spam threshold [{config['action_thresholds']['spam']}]: ")
                    or config["action_thresholds"]["spam"]
                )
                config["action_thresholds"]["trash"] = float(
                    input(f"  Trash threshold [{config['action_thresholds']['trash']}]: ")
                    or config["action_thresholds"]["trash"]
                )
            except:
                pass
    
    # Save configuration
    config_path = Path("gmail_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Configuration saved to {config_path}")
    
    print("\n" + "-"*60)
    print("Step 4: Test Connection")
    print("-"*60)
    
    test = input("\nTest Gmail connection now? (y/n): ").lower() == 'y'
    
    if test:
        print("\nTesting connection...")
        try:
            import asyncio
            from fraudlens.api.gmail_integration import GmailFraudScanner
            
            async def test_connection():
                scanner = GmailFraudScanner(
                    credentials_file="credentials.json",
                    fraud_threshold=config["fraud_threshold"],
                    auto_action=config["auto_action"],
                )
                
                await scanner.initialize()
                print("✅ Successfully connected to Gmail!")
                
                # Get email count
                results = scanner.service.users().messages().list(
                    userId='me',
                    maxResults=1
                ).execute()
                
                total = results.get('resultSizeEstimate', 0)
                print(f"📧 Found approximately {total} emails in mailbox")
                
                return True
            
            success = asyncio.run(test_connection())
            
            if success:
                print("\n" + "="*60)
                print("✅ Gmail setup complete!")
                print("="*60)
                print("\nYou can now:")
                print("1. Run the API server: python -m fraudlens.api.email_api")
                print("2. Use the Gmail scanner in your code")
                print("3. Access the API at http://localhost:8000/docs")
                
                return True
                
        except Exception as e:
            print(f"\n❌ Connection test failed: {e}")
            print("\nPlease check your credentials and try again.")
            return False
    
    return True


def create_example_script():
    """Create an example usage script."""
    
    example = '''#!/usr/bin/env python3
"""
Example Gmail fraud scanning script
"""

import asyncio
from fraudlens.api.gmail_integration import GmailFraudScanner, EmailAction


async def main():
    # Initialize scanner
    scanner = GmailFraudScanner(
        fraud_threshold=0.7,
        auto_action=True,  # Automatically take action
        action_threshold={
            EmailAction.FLAG: 0.5,
            EmailAction.SPAM: 0.7,
            EmailAction.TRASH: 0.95,
        }
    )
    
    await scanner.initialize()
    
    # Scan unread emails from last 7 days
    print("\\nScanning unread emails...")
    results = await scanner.stream_emails(
        query="is:unread",
        max_results=50,
        process_attachments=True,
        since_days=7
    )
    
    # Display results
    print(f"\\nProcessed {len(results)} emails\\n")
    
    fraud_count = 0
    for result in results:
        if result.fraud_score > scanner.fraud_threshold:
            fraud_count += 1
            print(f"⚠️  FRAUD DETECTED:")
            print(f"   Subject: {result.subject}")
            print(f"   From: {result.sender}")
            print(f"   Score: {result.fraud_score:.2%}")
            print(f"   Types: {', '.join(result.fraud_types)}")
            print(f"   Action: {result.action_taken.value}")
            print()
    
    print(f"\\nSummary:")
    print(f"  Total emails: {len(results)}")
    print(f"  Fraudulent: {fraud_count}")
    print(f"  Clean: {len(results) - fraud_count}")
    
    # Get statistics
    stats = scanner.get_statistics()
    print(f"\\nStatistics:")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Fraud detected: {stats['fraud_detected']}")
    print(f"  Fraud rate: {stats['fraud_rate']:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
'''
    
    with open("example_gmail_scan.py", "w") as f:
        f.write(example)
    
    print("\n📝 Created example_gmail_scan.py")


if __name__ == "__main__":
    success = setup_gmail_auth()
    
    if success:
        create_example_script()
        print("\n✨ Setup complete! Run 'python example_gmail_scan.py' to test.")
    else:
        print("\n❌ Setup incomplete. Please try again.")