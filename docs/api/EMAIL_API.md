# FraudLens Email API Documentation

## Overview

The FraudLens Email API provides comprehensive email fraud detection capabilities with Gmail integration. It can process emails in bulk, analyze attachments (images, videos, documents), and automatically manage fraudulent emails.

## Features

- **Real-time Email Streaming**: Process emails directly from Gmail
- **Attachment Analysis**: Analyze images, videos, and documents
- **Bulk Processing**: Handle multiple email queries in parallel
- **Automatic Actions**: Move fraudulent emails to spam/trash
- **Continuous Monitoring**: Monitor inbox for new threats
- **Multi-modal Analysis**: Combined text and attachment fraud detection

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Gmail OAuth2

Run the setup script to configure Gmail API access:

```bash
python setup_gmail.py
```

This will guide you through:
- Enabling Gmail API in Google Cloud Console
- Creating OAuth2 credentials
- Downloading credentials file
- Testing the connection

### 3. Start the API Server

```bash
python -m fraudlens.api.email_api
```

Or using Docker:

```bash
docker-compose -f docker-compose.email.yml up
```

## API Endpoints

### Base URL
```
http://localhost:8000
```

### Authentication
The API uses Gmail OAuth2 for authentication. Credentials are managed server-side.

## Endpoints

### 1. Stream and Scan Emails

**POST** `/api/v1/scan/stream`

Stream emails from Gmail and analyze for fraud.

#### Request Body
```json
{
  "query": "is:unread",
  "max_results": 100,
  "process_attachments": true,
  "since_days": 7
}
```

#### Response
```json
{
  "success": true,
  "count": 25,
  "results": [
    {
      "message_id": "18b7c9d8e5f4a3b2",
      "subject": "Urgent: Verify Your Account",
      "sender": "noreply@suspicious-site.com",
      "date": "2025-08-29T10:30:00Z",
      "fraud_score": 0.89,
      "fraud_types": ["phishing", "social_engineering"],
      "confidence": 0.92,
      "explanation": "High risk indicators for phishing...",
      "attachments_analyzed": [
        {
          "filename": "document.pdf",
          "fraud_score": 0.75,
          "fraud_types": ["document_fraud"]
        }
      ],
      "action_taken": "spam",
      "combined_score": 0.89,
      "flagged": true
    }
  ],
  "statistics": {
    "total_processed": 25,
    "fraud_detected": 3,
    "fraud_rate": 0.12
  }
}
```

### 2. Bulk Process Multiple Queries

**POST** `/api/v1/scan/bulk`

Process multiple Gmail queries in parallel.

#### Request Body
```json
{
  "queries": [
    "from:noreply",
    "subject:urgent",
    "has:attachment larger:5M"
  ],
  "parallel": true,
  "max_workers": 5
}
```

#### Response
```json
{
  "success": true,
  "total_processed": 150,
  "total_fraud_detected": 23,
  "queries": {
    "from:noreply": {
      "count": 50,
      "fraud_count": 12,
      "results": [...]
    },
    "subject:urgent": {
      "count": 75,
      "fraud_count": 8,
      "results": [...]
    }
  }
}
```

### 3. Scan Single Email

**GET** `/api/v1/scan/email/{message_id}?process_attachments=true`

Analyze a specific email by message ID.

#### Response
```json
{
  "success": true,
  "result": {
    "message_id": "18b7c9d8e5f4a3b2",
    "fraud_score": 0.89,
    "fraud_types": ["phishing"],
    "attachments_analyzed": 2,
    "action_taken": "spam"
  }
}
```

### 4. Start Monitoring

**POST** `/api/v1/monitor/start`

Begin continuous inbox monitoring.

#### Request Body
```json
{
  "enabled": true,
  "interval_seconds": 60,
  "query": "is:unread",
  "auto_action": true
}
```

### 5. Execute Manual Action

**POST** `/api/v1/action/execute`

Manually execute actions on emails.

#### Request Body
```json
{
  "message_ids": ["msg1", "msg2", "msg3"],
  "action": "spam"
}
```

Available actions:
- `flag` - Add warning label
- `spam` - Move to spam
- `trash` - Move to trash
- `quarantine` - Move to quarantine folder

### 6. Update Configuration

**PUT** `/api/v1/config/update`

Update fraud detection thresholds and settings.

#### Request Body
```json
{
  "fraud_threshold": 0.7,
  "auto_action": true,
  "action_thresholds": {
    "flag": 0.5,
    "spam": 0.7,
    "trash": 0.95,
    "quarantine": 0.8
  }
}
```

### 7. Get Statistics

**GET** `/api/v1/statistics`

Get processing statistics.

#### Response
```json
{
  "total_processed": 1250,
  "fraud_detected": 187,
  "attachments_processed": 423,
  "fraud_rate": 0.15,
  "avg_processing_time_ms": 245.3,
  "actions_taken": {
    "spam": 150,
    "trash": 12,
    "flag": 25
  }
}
```

### 8. Export Results

**GET** `/api/v1/export/results?format=json&since_days=7`

Export scan results in JSON or CSV format.

## Gmail Labels

FraudLens automatically creates and manages these Gmail labels:

- `FraudLens/Analyzed` - All processed emails
- `FraudLens/Safe` - Clean emails (score < 0.5)
- `FraudLens/Suspicious` - Moderate risk (0.5-0.8)
- `FraudLens/Fraud` - High risk (> 0.8)
- `FraudLens/Quarantine` - Isolated suspicious emails

## Fraud Detection Process

### 1. Email Content Analysis
- Subject line patterns
- Sender reputation
- Body text for phishing indicators
- URL analysis
- Urgency and pressure tactics

### 2. Attachment Processing
- **Images**: Forgery detection, deepfake analysis
- **Videos**: Manipulation detection, deepfake analysis
- **Documents**: Document fraud, tampering detection
- **PDFs**: Text extraction and analysis

### 3. Combined Scoring
- Content score: 0.0-1.0
- Attachment scores: 0.0-1.0 each
- Combined score: Maximum of all scores
- Confidence level: Model confidence

### 4. Automatic Actions
Based on combined score:
- **0.5-0.7**: Flag with warning label
- **0.7-0.8**: Move to spam
- **0.8-0.95**: Quarantine
- **> 0.95**: Move to trash

## Example Usage

### Python Client

```python
import asyncio
from fraudlens.api.gmail_integration import GmailFraudScanner

async def scan_emails():
    scanner = GmailFraudScanner(
        fraud_threshold=0.7,
        auto_action=True
    )
    
    await scanner.initialize()
    
    # Scan unread emails
    results = await scanner.stream_emails(
        query="is:unread",
        max_results=50,
        process_attachments=True
    )
    
    for result in results:
        if result.fraud_score > 0.7:
            print(f"Fraud detected: {result.subject}")
            print(f"Score: {result.fraud_score:.2%}")
            print(f"Action: {result.action_taken.value}")

asyncio.run(scan_emails())
```

### cURL Examples

Stream and scan emails:
```bash
curl -X POST http://localhost:8000/api/v1/scan/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "is:unread",
    "max_results": 10,
    "process_attachments": true
  }'
```

Start monitoring:
```bash
curl -X POST http://localhost:8000/api/v1/monitor/start \
  -H "Content-Type: application/json" \
  -d '{
    "enabled": true,
    "interval_seconds": 60,
    "auto_action": true
  }'
```

## Performance

- **Processing Speed**: ~200-500ms per email (without attachments)
- **Attachment Processing**: +500-2000ms per attachment
- **Batch Processing**: Up to 100 emails per request
- **Monitoring Interval**: Minimum 10 seconds
- **Concurrent Workers**: Maximum 20 for bulk processing

## Security Considerations

1. **OAuth2 Credentials**: Store securely, never commit to repository
2. **Token Storage**: `token.pickle` contains access tokens
3. **Rate Limits**: Gmail API has quotas (250 quota units/user/second)
4. **Privacy**: Email content is processed locally, not sent to external services
5. **Actions**: Irreversible actions (trash) require high confidence

## Error Handling

The API returns standard HTTP status codes:

- **200**: Success
- **400**: Bad request (invalid parameters)
- **403**: Authentication failed
- **429**: Rate limit exceeded
- **500**: Internal server error
- **503**: Service unavailable (scanner not initialized)

Error response format:
```json
{
  "detail": "Error message",
  "status_code": 500
}
```

## Monitoring & Alerts

### Webhook Integration

Configure Gmail push notifications:

1. Set up Cloud Pub/Sub topic
2. Configure Gmail to send notifications
3. Point webhook to `/api/v1/webhook/gmail`

### Real-time Monitoring

Use WebSocket connection for real-time updates:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.fraud_detected) {
    alert(`Fraud detected: ${data.subject}`);
  }
};
```

## Limitations

- **Gmail Quotas**: 250 quota units/user/second
- **Attachment Size**: Maximum 25MB per attachment
- **Processing Time**: Large videos may timeout (10 minute limit)
- **Concurrent Requests**: Maximum 100 concurrent API calls
- **Storage**: Temporary files cleaned after processing

## Support

For issues or questions:
- GitHub Issues: https://github.com/YobieBen/FraudLens/issues
- Documentation: https://yobieben.github.io/FraudLens/
- API Reference: http://localhost:8000/docs (Swagger UI)