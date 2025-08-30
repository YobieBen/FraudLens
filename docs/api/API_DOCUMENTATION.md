# FraudLens API Documentation

## Base URL
```
Production: https://api.fraudlens.com/v2
Staging: https://staging-api.fraudlens.com/v2
Development: http://localhost:8000
```

## Authentication

FraudLens API uses JWT (JSON Web Tokens) for authentication. You need to obtain an access token before making API requests.

### Getting Started

#### 1. Register a New User
```bash
curl -X POST https://api.fraudlens.com/v2/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "johndoe",
    "email": "john@example.com",
    "password": "SecurePassword123!",
    "role": "user"
  }'
```

**Response:**
```json
{
  "id": 123,
  "username": "johndoe",
  "email": "john@example.com",
  "role": "user",
  "is_active": true,
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### 2. Login to Get Access Token
```bash
curl -X POST https://api.fraudlens.com/v2/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=john@example.com&password=SecurePassword123!"
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### 3. Use the Access Token
Include the access token in the Authorization header for all subsequent requests:
```bash
curl -X GET https://api.fraudlens.com/v2/email/stats \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..."
```

## Email Fraud Detection

### Scan Inbox for Fraudulent Emails

Scan your Gmail inbox for potential fraud, phishing, and scam emails.

**Endpoint:** `POST /email/scan-inbox`

**Request:**
```bash
curl -X POST https://api.fraudlens.com/v2/email/scan-inbox \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "max_emails": 100,
    "query": "is:unread",
    "date_from": "2024-01-01",
    "date_to": "2024-01-31"
  }'
```

**Response:**
```json
{
  "results": [
    {
      "message_id": "msg_123abc",
      "subject": "URGENT: Verify Your Account",
      "from_address": "security@fake-bank.com",
      "date": "2024-01-15T08:30:00Z",
      "is_fraud": true,
      "confidence": 0.95,
      "fraud_type": "phishing",
      "risk_score": 9.2,
      "risk_level": "high",
      "explanation": "Email contains suspicious links and urgency indicators typical of phishing attempts",
      "recommended_action": "block"
    }
  ],
  "total_scanned": 100,
  "fraud_detected": 12,
  "scan_duration": 5.3
}
```

### Scan Specific Email

Analyze a specific email by its Gmail message ID.

**Endpoint:** `GET /email/scan/{message_id}`

**Request:**
```bash
curl -X GET https://api.fraudlens.com/v2/email/scan/msg_123abc \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Batch Email Scanning

Scan multiple emails in a single request.

**Endpoint:** `POST /email/batch-scan`

**Request:**
```bash
curl -X POST https://api.fraudlens.com/v2/email/batch-scan \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message_ids": ["msg_123", "msg_456", "msg_789"],
    "batch_size": 50
  }'
```

### Mark Emails as Spam

Move suspicious emails to spam folder.

**Endpoint:** `POST /email/mark-spam`

**Request:**
```bash
curl -X POST https://api.fraudlens.com/v2/email/mark-spam \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message_ids": ["msg_123", "msg_456"]
  }'
```

### Create Email Filter

Create automated filters to handle fraudulent emails (Admin only).

**Endpoint:** `POST /email/create-filter`

**Request:**
```bash
curl -X POST https://api.fraudlens.com/v2/email/create-filter \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "from_address": "*@suspicious-domain.com",
    "subject_contains": "lottery winner",
    "action": "delete"
  }'
```

## Real-time Monitoring

### Start Email Monitoring

Enable real-time monitoring of your inbox.

**Endpoint:** `POST /email/monitor/start`

**Request:**
```bash
curl -X POST https://api.fraudlens.com/v2/email/monitor/start \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "check_interval": 300,
    "callback_url": "https://your-webhook.com/fraud-alert",
    "auto_actions": {
      "mark_spam": true,
      "delete_high_risk": false
    }
  }'
```

**Response:**
```json
{
  "monitoring": true,
  "monitor_id": "mon_abc123",
  "next_check": "2024-01-15T10:35:00Z"
}
```

### Webhook Payload

When fraud is detected, the following payload is sent to your webhook:

```json
{
  "event": "fraud_detected",
  "timestamp": "2024-01-15T10:30:00Z",
  "monitor_id": "mon_abc123",
  "email": {
    "message_id": "msg_789",
    "subject": "You've won $1,000,000",
    "from": "lottery@scam.com",
    "fraud_type": "scam",
    "risk_level": "high",
    "confidence": 0.98
  },
  "action_taken": "moved_to_spam"
}
```

## Text Analysis

### Analyze Text for Fraud

Detect fraud in text content.

**Endpoint:** `POST /analyze/text`

**Request:**
```bash
curl -X POST https://api.fraudlens.com/v2/analyze/text \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Congratulations! You have won $1,000,000. Click here to claim your prize immediately!",
    "content_type": "text"
  }'
```

**Response:**
```json
{
  "is_fraud": true,
  "confidence": 0.97,
  "fraud_types": ["scam", "phishing"],
  "risk_level": "high",
  "explanation": "Content contains lottery scam indicators and suspicious urgency",
  "indicators": [
    "Unrealistic monetary promise",
    "Urgency indicator: 'immediately'",
    "Call-to-action with suspicious link"
  ],
  "request_id": "req_xyz789",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Image Analysis

### Detect Image Manipulation

Check if an image has been manipulated or is a deepfake.

**Endpoint:** `POST /analyze/image`

**Request:**
```bash
curl -X POST https://api.fraudlens.com/v2/analyze/image \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "image=@/path/to/image.jpg" \
  -F "check_deepfake=true" \
  -F "check_manipulation=true"
```

**Response:**
```json
{
  "is_manipulated": true,
  "manipulation_confidence": 0.89,
  "is_deepfake": false,
  "deepfake_confidence": 0.12,
  "manipulation_types": ["splicing", "cloning"],
  "authenticity_score": 0.23,
  "metadata": {
    "original_camera": "Unknown",
    "editing_software": "Photoshop CC 2023"
  }
}
```

## Document Validation

### Validate Document Authenticity

Check if a document is authentic or fraudulent.

**Endpoint:** `POST /analyze/document`

**Request:**
```bash
curl -X POST https://api.fraudlens.com/v2/analyze/document \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "document=@/path/to/document.pdf" \
  -F "document_type=invoice"
```

**Response:**
```json
{
  "is_authentic": false,
  "confidence": 0.76,
  "document_type": "invoice",
  "issues_found": [
    "Inconsistent formatting",
    "Invalid company registration number",
    "Suspicious payment details"
  ],
  "verification_status": "fraudulent"
}
```

## Reports and Analytics

### Get Email Security Statistics

Retrieve statistics about email security.

**Endpoint:** `GET /email/stats`

**Request:**
```bash
curl -X GET "https://api.fraudlens.com/v2/email/stats?period=month" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**
```json
{
  "total_scanned": 5234,
  "fraud_detected": 423,
  "fraud_rate": 0.081,
  "common_fraud_types": {
    "phishing": 245,
    "scam": 132,
    "malware": 46
  },
  "high_risk_senders": [
    "noreply@suspicious-bank.com",
    "winner@lottery-scam.net"
  ],
  "blocked_count": 389,
  "false_positives": 12
}
```

### Generate Fraud Report

Generate detailed fraud detection report.

**Endpoint:** `POST /reports/generate`

**Request:**
```bash
curl -X POST https://api.fraudlens.com/v2/reports/generate \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "format": "pdf",
    "include_details": true
  }'
```

**Response:**
```json
{
  "report_id": "rpt_abc123",
  "download_url": "https://api.fraudlens.com/v2/reports/download/rpt_abc123",
  "expires_at": "2024-02-01T10:30:00Z"
}
```

## Rate Limiting

API rate limits are enforced based on user role:

| Role | Rate Limit | Time Window |
|------|------------|-------------|
| Admin | 5000 requests | per minute |
| User | 1000 requests | per minute |
| Viewer | 500 requests | per minute |
| API User | 2000 requests | per minute |

Rate limit information is included in response headers:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1705315200
```

## Error Handling

### Error Response Format

All errors follow a consistent format:

```json
{
  "error": "ValidationError",
  "message": "Invalid email format",
  "status_code": 400,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Common Error Codes

| Status Code | Description | Common Causes |
|-------------|-------------|---------------|
| 400 | Bad Request | Invalid parameters, malformed JSON |
| 401 | Unauthorized | Missing or invalid token |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource doesn't exist |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily down |

## SDKs and Code Examples

### Python SDK

```python
from fraudlens import FraudLensClient

# Initialize client
client = FraudLensClient(
    api_key="your_api_key",
    base_url="https://api.fraudlens.com/v2"
)

# Scan inbox
results = client.email.scan_inbox(
    max_emails=100,
    query="is:unread"
)

# Analyze text
analysis = client.analyze.text(
    content="Suspicious message content",
    content_type="email"
)

# Start monitoring
monitor = client.monitor.start_email_monitoring(
    check_interval=300,
    callback_url="https://your-webhook.com"
)
```

### JavaScript/Node.js SDK

```javascript
const FraudLens = require('fraudlens-sdk');

// Initialize client
const client = new FraudLens({
  apiKey: 'your_api_key',
  baseURL: 'https://api.fraudlens.com/v2'
});

// Scan inbox
const results = await client.email.scanInbox({
  maxEmails: 100,
  query: 'is:unread'
});

// Analyze text
const analysis = await client.analyze.text({
  content: 'Suspicious message content',
  contentType: 'email'
});
```

### cURL Examples

See individual endpoint sections above for cURL examples.

## Postman Collection

Download our Postman collection for easy API testing:
[Download Postman Collection](https://api.fraudlens.com/postman-collection.json)

## API Changelog

### Version 2.0.0 (Current)
- Added Gmail integration
- Enhanced real-time monitoring
- Improved fraud detection accuracy
- Added batch processing endpoints

### Version 1.5.0
- Added document validation
- Improved image analysis
- Added webhook support

## Support

For API support:
- Email: api-support@fraudlens.com
- Documentation: https://docs.fraudlens.com
- Status Page: https://status.fraudlens.com