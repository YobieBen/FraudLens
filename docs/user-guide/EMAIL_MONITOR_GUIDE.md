# FraudLens Email Monitor User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Email Monitor Features](#email-monitor-features)
4. [Gmail Integration Setup](#gmail-integration-setup)
5. [Using the Email Monitor](#using-the-email-monitor)
6. [Understanding Fraud Detection](#understanding-fraud-detection)
7. [Managing Fraudulent Emails](#managing-fraudulent-emails)
8. [Real-time Monitoring](#real-time-monitoring)
9. [Reports and Analytics](#reports-and-analytics)
10. [Best Practices](#best-practices)

## Introduction

The FraudLens Email Monitor is a powerful tool designed to protect your inbox from fraudulent emails, phishing attempts, and scams. Using advanced AI and machine learning, it automatically scans your emails and identifies potential threats in real-time.

### Key Benefits
- 🛡️ **Real-time Protection**: Continuous monitoring of incoming emails
- 🤖 **AI-Powered Detection**: Advanced algorithms identify sophisticated fraud attempts
- 📊 **Detailed Analytics**: Comprehensive reports on email security
- 🔄 **Automated Actions**: Automatically move suspicious emails to spam/trash
- 🔍 **Deep Analysis**: Examines content, headers, attachments, and sender reputation

## Getting Started

### Prerequisites
- Gmail account with API access enabled
- Python 3.8 or higher
- FraudLens installed and configured
- Valid Google Cloud credentials

### Quick Start
1. Launch the FraudLens application
2. Navigate to the Email Monitor tab
3. Connect your Gmail account
4. Start monitoring

## Email Monitor Features

### 1. Dashboard Overview
The Email Monitor dashboard provides a comprehensive view of your email security status:

```
┌─────────────────────────────────────────────┐
│         Email Monitor Dashboard             │
├─────────────────────────────────────────────┤
│ Status: 🟢 Active                           │
│ Emails Scanned Today: 247                  │
│ Threats Detected: 12                       │
│ Protection Level: High                     │
└─────────────────────────────────────────────┘
```

### 2. Feature Overview

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Auto-Scan** | Automatically scans new emails | Continuous protection |
| **Batch Processing** | Scan multiple emails simultaneously | Fast analysis |
| **Smart Filters** | Create rules based on fraud patterns | Proactive blocking |
| **Whitelist/Blacklist** | Manage trusted and blocked senders | Customized security |
| **Real-time Alerts** | Instant notifications for threats | Immediate awareness |
| **Detailed Reports** | Comprehensive security analytics | Informed decisions |

## Gmail Integration Setup

### Step 1: Enable Gmail API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Gmail API:
   ```
   APIs & Services → Library → Search "Gmail API" → Enable
   ```

### Step 2: Create Credentials

1. Navigate to **APIs & Services → Credentials**
2. Click **Create Credentials → OAuth 2.0 Client ID**
3. Configure consent screen:
   - Application name: FraudLens Email Monitor
   - Authorized domains: Your domain
   - Scopes: Gmail API scopes

4. Download credentials as `credentials.json`

### Step 3: Configure FraudLens

1. Place `credentials.json` in the FraudLens config directory:
   ```bash
   cp credentials.json ~/.fraudlens/config/
   ```

2. Set environment variables:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=~/.fraudlens/config/credentials.json
   export GMAIL_USER_EMAIL=your-email@gmail.com
   ```

### Step 4: Authorize Access

1. Run the authorization script:
   ```bash
   python -m fraudlens.auth.gmail_auth
   ```

2. Follow the browser prompt to authorize access
3. Token will be saved automatically

## Using the Email Monitor

### Scanning Your Inbox

#### Manual Scan
1. Click **"Scan Inbox"** button
2. Select scan options:
   - Number of emails to scan
   - Date range
   - Specific labels/folders
3. Click **"Start Scan"**

#### Automatic Scanning
1. Enable **"Auto-Scan"** toggle
2. Configure settings:
   ```
   Scan Interval: [Every 5 minutes ▼]
   Max Emails per Scan: [50 ▼]
   Actions for Fraud: [Move to Spam ▼]
   ```

### Viewing Scan Results

The results table shows:

| Email Subject | From | Date | Risk Level | Fraud Type | Actions |
|--------------|------|------|------------|------------|---------|
| URGENT: Verify Account | scammer@fake.com | Today | 🔴 High | Phishing | [Spam] [Trash] [Report] |
| You've won $1M! | lottery@scam.net | Yesterday | 🔴 High | Scam | [Spam] [Trash] [Report] |
| Newsletter | news@legitimate.com | Today | 🟢 Low | None | [Safe] |

### Understanding Risk Levels

- 🟢 **Low Risk (0-30%)**: Likely legitimate
- 🟡 **Medium Risk (30-70%)**: Requires review
- 🔴 **High Risk (70-100%)**: Likely fraudulent

## Understanding Fraud Detection

### Detection Techniques

#### 1. Content Analysis
- **Keyword Detection**: Identifies suspicious phrases
- **Urgency Indicators**: Detects pressure tactics
- **Link Analysis**: Examines URLs for phishing
- **Grammar Check**: Poor grammar often indicates fraud

#### 2. Sender Verification
- **Domain Reputation**: Checks sender domain history
- **SPF/DKIM Validation**: Verifies email authentication
- **Blacklist Check**: Compares against known bad actors
- **Display Name Spoofing**: Detects impersonation attempts

#### 3. Pattern Recognition
- **Template Matching**: Identifies known fraud templates
- **Behavioral Analysis**: Detects unusual patterns
- **Machine Learning**: Learns from new fraud attempts

### Fraud Types Detected

| Type | Description | Common Signs |
|------|-------------|--------------|
| **Phishing** | Attempts to steal credentials | Fake login pages, urgent requests |
| **Scam** | Financial fraud attempts | Too-good-to-be-true offers |
| **Malware** | Contains malicious attachments | Suspicious files, executables |
| **Impersonation** | Pretends to be legitimate entity | Spoofed addresses, logos |
| **BEC** | Business Email Compromise | Wire transfer requests, invoice fraud |

## Managing Fraudulent Emails

### Quick Actions

#### Mark as Spam
```python
# Automatically moves to spam folder
email_monitor.mark_as_spam(email_id)
```

#### Move to Trash
```python
# Permanently delete after 30 days
email_monitor.move_to_trash(email_id)
```

#### Report Fraud
```python
# Report to authorities and blocklist
email_monitor.report_fraud(email_id, details)
```

### Bulk Operations

1. Select multiple emails using checkboxes
2. Choose bulk action from dropdown:
   - Mark all as spam
   - Delete selected
   - Add senders to blacklist
   - Export for review

### Creating Filters

Create custom filters to automatically handle fraudulent emails:

```yaml
Filter Name: Block Lottery Scams
Conditions:
  - Subject contains: "lottery", "winner", "congratulations"
  - From domain: ends with ".tk", ".ml"
  - Has attachment: yes
Actions:
  - Mark as spam: yes
  - Delete: no
  - Notify: yes
```

## Real-time Monitoring

### Setting Up Monitoring

1. Navigate to **Settings → Monitoring**
2. Configure monitoring parameters:

```json
{
  "enabled": true,
  "check_interval": 300,  // seconds
  "max_emails_per_check": 50,
  "auto_actions": {
    "high_risk": "move_to_spam",
    "medium_risk": "flag_for_review",
    "low_risk": "allow"
  },
  "notifications": {
    "email": true,
    "desktop": true,
    "webhook": "https://your-webhook.com/fraud-alert"
  }
}
```

### Monitoring Dashboard

```
┌─────────────────────────────────────────────┐
│        Real-time Monitoring Status          │
├─────────────────────────────────────────────┤
│ 🟢 Active Monitoring                        │
│                                             │
│ Last Check: 2 minutes ago                  │
│ Next Check: in 3 minutes                   │
│                                             │
│ Today's Activity:                          │
│ ├─ Emails Checked: 523                     │
│ ├─ Threats Blocked: 27                     │
│ └─ False Positives: 2                      │
│                                             │
│ [Pause] [Settings] [View Logs]             │
└─────────────────────────────────────────────┘
```

### Alert Configuration

Configure how you receive alerts:

| Alert Type | Email | Desktop | SMS | Webhook |
|------------|-------|---------|-----|---------|
| High Risk Detected | ✅ | ✅ | ✅ | ✅ |
| Multiple Threats | ✅ | ✅ | ❌ | ✅ |
| New Fraud Pattern | ✅ | ❌ | ❌ | ✅ |
| System Issues | ✅ | ✅ | ✅ | ✅ |

## Reports and Analytics

### Dashboard Analytics

View comprehensive analytics about your email security:

```
📊 Email Security Analytics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Time Period: Last 30 Days

📈 Fraud Detection Trend
    High ████████░░ 42%
  Medium ██████░░░░ 31%
     Low ███░░░░░░░ 27%

🎯 Top Fraud Types
1. Phishing (45%)
2. Scams (28%)
3. Malware (15%)
4. BEC (12%)

📧 Email Statistics
• Total Scanned: 15,234
• Fraudulent: 892 (5.9%)
• Auto-Blocked: 743
• False Positives: 12

👥 Top Risk Senders
1. no-reply@suspicious-domain.tk (23 emails)
2. winner@lottery-scam.ml (19 emails)
3. security@fake-bank.com (17 emails)
```

### Generating Reports

#### Quick Report
1. Click **"Generate Report"**
2. Select report type:
   - Daily Summary
   - Weekly Analysis
   - Monthly Overview
   - Custom Range

#### Detailed Report Options

```python
report = email_monitor.generate_report(
    start_date="2024-01-01",
    end_date="2024-01-31",
    include_sections=[
        "executive_summary",
        "threat_analysis",
        "sender_reputation",
        "action_taken",
        "recommendations"
    ],
    format="pdf"  # or "csv", "json", "html"
)
```

### Exporting Data

Export options for further analysis:

| Format | Use Case | Contains |
|--------|----------|----------|
| **CSV** | Spreadsheet analysis | Raw data, statistics |
| **PDF** | Management reports | Charts, summaries |
| **JSON** | API integration | Structured data |
| **HTML** | Web viewing | Interactive charts |

## Best Practices

### 1. Regular Monitoring
- ✅ Enable automatic scanning
- ✅ Review daily summaries
- ✅ Check false positives weekly
- ✅ Update filters monthly

### 2. Whitelist Management
Maintain a whitelist of trusted senders:
```
Trusted Senders:
- *@yourcompany.com
- newsletter@trustedservice.com
- support@knownvendor.com
```

### 3. Security Settings

Recommended security configuration:

```yaml
Security Level: High
Auto-Actions: Enabled
Scan Attachments: Yes
Check Links: Yes
Verify SPF/DKIM: Yes
Machine Learning: Enabled
Real-time Updates: Yes
```

### 4. Response Plan

When fraud is detected:

1. **Immediate Actions**
   - Don't click any links
   - Don't download attachments
   - Mark as fraud
   - Report to IT/Security team

2. **Follow-up**
   - Change passwords if compromised
   - Alert colleagues about the threat
   - Update filters to prevent recurrence

### 5. Training and Awareness

- Review fraud patterns monthly
- Share threat intelligence with team
- Test security awareness with simulations
- Keep detection models updated

## Troubleshooting Common Issues

### Email Not Scanning

**Problem**: Emails are not being scanned automatically

**Solutions**:
1. Check Gmail API connection
2. Verify credentials are valid
3. Ensure monitoring is enabled
4. Check rate limits

### High False Positive Rate

**Problem**: Legitimate emails marked as fraud

**Solutions**:
1. Adjust sensitivity settings
2. Add senders to whitelist
3. Report false positives for model improvement
4. Review and update filters

### Performance Issues

**Problem**: Slow scanning or timeouts

**Solutions**:
1. Reduce batch size
2. Increase scan interval
3. Check network connectivity
4. Optimize filter rules

## FAQ

**Q: How accurate is the fraud detection?**
A: FraudLens achieves 95%+ accuracy with less than 2% false positive rate.

**Q: Can I customize detection rules?**
A: Yes, you can create custom filters and adjust sensitivity settings.

**Q: Is my email data secure?**
A: All data is encrypted in transit and at rest. We don't store email content.

**Q: How often should I scan my inbox?**
A: We recommend continuous monitoring with 5-minute intervals for optimal protection.

**Q: Can I use this with multiple email accounts?**
A: Yes, you can add multiple Gmail accounts to monitor.

## Support

For additional help:
- 📧 Email: support@fraudlens.com
- 📚 Documentation: https://docs.fraudlens.com
- 💬 Community Forum: https://community.fraudlens.com
- 🐛 Report Issues: https://github.com/fraudlens/issues