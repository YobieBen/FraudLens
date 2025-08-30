# Gmail Integration Setup Guide

## Prerequisites

Before connecting FraudLens to your Gmail account, you need to:

1. **Enable 2-Factor Authentication (2FA)**
   - Go to your Google Account settings: https://myaccount.google.com
   - Navigate to Security → 2-Step Verification
   - Follow the setup process

2. **Generate an App-Specific Password**
   - Visit: https://myaccount.google.com/apppasswords
   - Sign in if prompted
   - Select "Mail" from the app dropdown
   - Select your device or choose "Other" and enter "FraudLens"
   - Click "Generate"
   - Copy the 16-character password (spaces don't matter)

## Connecting to FraudLens

1. **Launch the integrated dashboard:**
   ```bash
   ./launch_integrated.sh
   ```

2. **Navigate to Gmail Integration tab**

3. **Enter your credentials:**
   - Email: Your full Gmail address (e.g., yourname@gmail.com)
   - Password: The 16-character app password (NOT your regular password)

4. **Click "Connect to Gmail"**

## Troubleshooting SSL Certificate Errors

If you encounter SSL certificate verification errors on macOS:

### Quick Fix
The launch script automatically sets SSL environment variables. If issues persist:

```bash
# Install/update certifi
pip3 install --upgrade certifi

# Test connection
python3 test_gmail_ssl.py
```

### Manual SSL Fix
```bash
# Set environment variables
export SSL_CERT_FILE=$(python3 -m certifi)
export REQUESTS_CA_BUNDLE=$(python3 -m certifi)

# Then launch the dashboard
python3 demo/gradio_app_integrated.py
```

### System-wide Fix (macOS)
```bash
# Update certificates
brew install ca-certificates
brew install certifi

# Or with pip
pip3 install --upgrade certifi
```

## Security Notes

- **Never share your app password**
- App passwords bypass 2FA, so keep them secure
- You can revoke app passwords anytime from Google Account settings
- FraudLens uses secure IMAP over SSL/TLS (port 993)

## Features Available

Once connected, you can:
- Scan unread emails for fraud
- Search emails with queries like:
  - `is:unread` - Unread emails
  - `from:sender@example.com` - Emails from specific sender
  - `subject:invoice` - Emails with specific subject
- Analyze up to 20 emails at once
- View fraud detection results with confidence scores
- Mark suspicious emails as spam
- Delete fraudulent emails

## Permissions

FraudLens only requires:
- Read access to your emails
- Ability to mark emails as read/spam
- Move emails to trash

It does NOT:
- Send emails on your behalf
- Access your contacts
- Modify email settings
- Store your password (only session-based)

## Revoking Access

To revoke FraudLens access:
1. Go to https://myaccount.google.com/apppasswords
2. Find the FraudLens app password
3. Click the trash icon to delete it

## Support

If you experience issues:
1. Verify 2FA is enabled
2. Regenerate the app password
3. Check internet connection
4. Run the SSL test script: `python3 test_gmail_ssl.py`
5. Update dependencies: `pip3 install --upgrade -r requirements.txt`