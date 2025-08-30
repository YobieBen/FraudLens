# FraudLens Documentation

Welcome to the FraudLens documentation! This comprehensive guide covers everything you need to know about using, deploying, and maintaining FraudLens.

## 📚 Documentation Overview

### [User Guide](user-guide/EMAIL_MONITOR_GUIDE.md)
Complete guide for using the FraudLens Email Monitor
- Getting started with Email Monitor
- Gmail integration setup
- Understanding fraud detection
- Managing fraudulent emails
- Real-time monitoring
- Reports and analytics
- Best practices

### [API Documentation](api/API_DOCUMENTATION.md)
RESTful API reference and examples
- Authentication (JWT/API Keys)
- Email fraud detection endpoints
- Text, image, and document analysis
- Real-time monitoring APIs
- Batch processing
- Webhooks and callbacks
- Rate limiting

### [OpenAPI Specification](api/openapi.yaml)
Interactive API documentation
- Swagger/OpenAPI 3.0 specification
- Request/response schemas
- Authentication methods
- Error codes
- Try it out functionality

### [Deployment Guide](deployment/DEPLOYMENT_GUIDE.md)
Production deployment instructions
- Docker deployment
- Kubernetes orchestration
- Cloud deployment (AWS/GCP/Azure)
- Configuration management
- Security best practices
- Scaling strategies
- Monitoring setup

### [Troubleshooting Guide](troubleshooting/TROUBLESHOOTING_GUIDE.md)
Common issues and solutions
- Quick diagnostics
- Error messages reference
- Performance tuning
- Recovery procedures
- Getting help

## 🚀 Quick Start

### 1. Installation

#### Using Docker (Recommended)
```bash
# Clone repository
git clone https://github.com/fraudlens/fraudlens.git
cd fraudlens

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start services
docker-compose up -d
```

#### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Run application
python -m fraudlens.api.main
```

### 2. First-Time Setup

#### Configure Gmail Integration
1. Enable Gmail API in Google Cloud Console
2. Download credentials.json
3. Run authentication:
```bash
python -m fraudlens.auth.gmail_auth
```

#### Create Admin User
```bash
python -m fraudlens.db.create_admin \
  --username admin \
  --email admin@example.com \
  --password SecurePassword123!
```

### 3. Access the Application

- **Web UI**: http://localhost:8501
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📖 Documentation Structure

```
docs/
├── README.md                 # This file
├── user-guide/
│   └── EMAIL_MONITOR_GUIDE.md   # User guide for Email Monitor
├── api/
│   ├── API_DOCUMENTATION.md     # API reference
│   └── openapi.yaml             # OpenAPI specification
├── deployment/
│   └── DEPLOYMENT_GUIDE.md      # Deployment instructions
├── troubleshooting/
│   └── TROUBLESHOOTING_GUIDE.md # Troubleshooting guide
└── images/                      # Documentation images
```

## 🔑 Key Features

### Email Fraud Detection
- **Gmail Integration**: Direct integration with Gmail API
- **Real-time Monitoring**: Continuous inbox monitoring
- **Batch Processing**: Scan multiple emails simultaneously
- **Smart Filters**: Create rules based on fraud patterns
- **Automated Actions**: Auto-move to spam/trash

### Fraud Analysis
- **Text Analysis**: Detect phishing, scams, and suspicious content
- **Image Analysis**: Identify manipulated images and deepfakes
- **Document Validation**: Verify document authenticity
- **Link Analysis**: Check for malicious URLs
- **Attachment Scanning**: Detect malware in attachments

### Security & Performance
- **JWT Authentication**: Secure API access
- **Rate Limiting**: Prevent abuse
- **Role-Based Access**: Admin, User, Viewer roles
- **Redis Caching**: High-performance caching
- **Parallel Processing**: Batch operations
- **Progress Tracking**: Real-time progress updates

## 📊 Architecture Overview

```mermaid
graph TB
    Client[Client Applications]
    
    Client --> LB[Load Balancer]
    LB --> API1[API Server 1]
    LB --> API2[API Server 2]
    LB --> API3[API Server 3]
    
    API1 --> Cache[Redis Cache]
    API2 --> Cache
    API3 --> Cache
    
    API1 --> DB[(PostgreSQL)]
    API2 --> DB
    API3 --> DB
    
    API1 --> Queue[Task Queue]
    API2 --> Queue
    API3 --> Queue
    
    Queue --> Worker1[Worker 1]
    Queue --> Worker2[Worker 2]
    
    Worker1 --> Gmail[Gmail API]
    Worker2 --> Gmail
    
    Worker1 --> ML[ML Models]
    Worker2 --> ML
```

## 🛠️ Configuration

### Environment Variables

Key configuration variables:

```env
# Application
FRAUDLENS_ENV=production
SECRET_KEY=your-secret-key
JWT_SECRET_KEY=your-jwt-secret

# Database
DATABASE_URL=postgresql://user:pass@host:5432/fraudlens

# Redis
REDIS_URL=redis://:password@host:6379/0

# Gmail API
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# Performance
WORKERS=4
CACHE_TTL=3600
```

See [.env.example](.env.example) for complete configuration options.

## 📈 Monitoring

### Metrics Endpoints
- `/metrics` - Prometheus metrics
- `/health` - Health check
- `/health/database` - Database health
- `/health/redis` - Redis health

### Dashboards
- Grafana dashboards in `monitoring/grafana/`
- Prometheus configuration in `monitoring/prometheus.yml`

## 🧪 Testing

### Run Tests
```bash
# All tests
make test

# Unit tests
make test-unit

# Integration tests
make test-integration

# Performance benchmarks
make test-performance

# Coverage report
make coverage
```

### Test Coverage
- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- Performance tests: `tests/performance/`

## 🔐 Security

### Best Practices
- Use HTTPS in production
- Rotate secrets regularly
- Enable rate limiting
- Keep dependencies updated
- Regular security audits
- Use least privilege principle

### Security Features
- JWT authentication
- API key management
- Role-based access control
- Rate limiting
- Input validation
- SQL injection prevention
- XSS protection

## 📝 API Examples

### Authenticate
```bash
curl -X POST http://localhost:8000/auth/token \
  -d "username=user@example.com&password=password"
```

### Scan Inbox
```bash
curl -X POST http://localhost:8000/email/scan-inbox \
  -H "Authorization: Bearer TOKEN" \
  -d '{"max_emails": 100}'
```

### Analyze Text
```bash
curl -X POST http://localhost:8000/analyze/text \
  -H "Authorization: Bearer TOKEN" \
  -d '{"content": "Suspicious message"}'
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Clone repository
git clone https://github.com/fraudlens/fraudlens.git

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Format code
black fraudlens/ tests/

# Lint code
flake8 fraudlens/ tests/
```

## 📄 License

FraudLens is licensed under the MIT License. See [LICENSE](../LICENSE) for details.

## 🆘 Support

### Getting Help
- **Documentation**: You're here!
- **GitHub Issues**: [Create an issue](https://github.com/fraudlens/fraudlens/issues)
- **Email**: support@fraudlens.com
- **Community Forum**: [community.fraudlens.com](https://community.fraudlens.com)
- **Slack**: [fraudlens.slack.com](https://fraudlens.slack.com)

### Reporting Issues
When reporting issues, please include:
1. Error messages and stack traces
2. Steps to reproduce
3. Environment details
4. Configuration (sanitized)
5. Relevant logs

## 📅 Changelog

### Version 2.0.0 (Latest)
- Gmail integration for email monitoring
- Real-time fraud detection
- Enhanced API with JWT authentication
- Performance optimizations
- Comprehensive documentation

### Version 1.5.0
- Document validation
- Image manipulation detection
- Webhook support
- Batch processing

See [CHANGELOG.md](../CHANGELOG.md) for complete version history.

## 🙏 Acknowledgments

FraudLens uses these excellent open-source projects:
- FastAPI for the REST API
- PostgreSQL for data storage
- Redis for caching
- Docker for containerization
- Kubernetes for orchestration

---

**Need more help?** Check our detailed guides above or contact support@fraudlens.com