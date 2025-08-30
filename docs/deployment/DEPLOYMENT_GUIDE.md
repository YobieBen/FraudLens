# FraudLens Deployment Guide

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Deployment Options](#deployment-options)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Cloud Deployment](#cloud-deployment)
7. [Configuration](#configuration)
8. [Security](#security)
9. [Monitoring](#monitoring)
10. [Backup and Recovery](#backup-and-recovery)
11. [Scaling](#scaling)
12. [Maintenance](#maintenance)

## Overview

FraudLens can be deployed in various environments, from single-server setups to distributed cloud architectures. This guide covers all deployment scenarios.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         Load Balancer                        │
│                         (Nginx/ALB)                          │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┬─────────────┐
        │                         │             │
┌───────▼────────┐   ┌───────────▼──────┐   ┌─▼──────────┐
│   API Server   │   │   API Server     │   │ API Server │
│   (Node 1)     │   │   (Node 2)       │   │ (Node 3)   │
└───────┬────────┘   └───────────┬──────┘   └─┬──────────┘
        │                         │             │
        └────────────┬────────────┴─────────────┘
                     │
        ┌────────────┴────────────┬─────────────┐
        │                         │             │
┌───────▼────────┐   ┌───────────▼──────┐   ┌─▼──────────┐
│     Redis      │   │   PostgreSQL     │   │   Workers  │
│    (Cache)     │   │   (Database)     │   │  (Queue)   │
└────────────────┘   └──────────────────┘   └────────────┘
```

## Prerequisites

### System Requirements

#### Minimum Requirements
- **CPU**: 2 cores
- **RAM**: 4 GB
- **Storage**: 20 GB SSD
- **OS**: Ubuntu 20.04+, CentOS 8+, or Docker-compatible OS

#### Recommended Requirements
- **CPU**: 4+ cores
- **RAM**: 8+ GB
- **Storage**: 100 GB SSD
- **Network**: 100 Mbps+

### Software Dependencies
```bash
# Docker
Docker Engine 20.10+
Docker Compose 2.0+

# Kubernetes (optional)
kubectl 1.24+
Helm 3.0+

# Database
PostgreSQL 13+
Redis 6.0+

# Python
Python 3.8+
pip 20.0+
```

## Deployment Options

### 1. Single Server (Development/Small Scale)
Best for: Development, testing, small organizations
- All components on one server
- Docker Compose orchestration
- Local PostgreSQL and Redis

### 2. Multi-Server (Medium Scale)
Best for: Medium organizations, production
- Separate database server
- Multiple API instances
- Dedicated worker nodes

### 3. Kubernetes (Large Scale)
Best for: Enterprise, high availability
- Container orchestration
- Auto-scaling
- Self-healing

### 4. Cloud Managed (Recommended for Production)
Best for: Production, scalability
- AWS/GCP/Azure services
- Managed databases
- Auto-scaling groups

## Docker Deployment

### Quick Start with Docker Compose

#### 1. Clone Repository
```bash
git clone https://github.com/fraudlens/fraudlens.git
cd fraudlens
```

#### 2. Configure Environment
```bash
cp .env.example .env.production
nano .env.production
```

Update critical values:
```env
# Security - MUST CHANGE
SECRET_KEY=your-secure-secret-key-min-32-chars
JWT_SECRET_KEY=your-jwt-secret-key-min-32-chars

# Database
POSTGRES_PASSWORD=secure-database-password
DATABASE_URL=postgresql://fraudlens:password@postgres:5432/fraudlens

# Redis
REDIS_PASSWORD=secure-redis-password

# Email
GMAIL_CREDENTIALS_PATH=/app/config/credentials.json
```

#### 3. Build and Start Services
```bash
# Build images
docker-compose -f docker-compose.production.yml build

# Start services
docker-compose -f docker-compose.production.yml up -d

# Check status
docker-compose -f docker-compose.production.yml ps
```

#### 4. Initialize Database
```bash
# Run migrations
docker-compose exec api python -m fraudlens.db.migrate

# Create admin user
docker-compose exec api python -m fraudlens.db.create_admin \
  --username admin \
  --email admin@example.com \
  --password SecureAdminPassword123!
```

#### 5. Verify Deployment
```bash
# Check health
curl http://localhost:8000/health

# View logs
docker-compose logs -f api
```

### Production Docker Configuration

#### Multi-Stage Build
```dockerfile
# Dockerfile.production
FROM python:3.11-slim as builder
# Build stage - compile dependencies

FROM python:3.11-slim
# Production stage - minimal image
```

#### Docker Compose Production
```yaml
version: '3.8'

services:
  api:
    image: fraudlens:latest
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
    environment:
      - FRAUDLENS_ENV=production
    depends_on:
      - postgres
      - redis
```

## Kubernetes Deployment

### Prerequisites
```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/stable.txt"
kubectl version --client

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### Deploy with Kubernetes

#### 1. Create Namespace
```bash
kubectl create namespace fraudlens
```

#### 2. Create Secrets
```bash
# Create secret from env file
kubectl create secret generic fraudlens-secret \
  --from-env-file=.env.production \
  -n fraudlens

# Create TLS certificate
kubectl create secret tls fraudlens-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key \
  -n fraudlens
```

#### 3. Deploy Application
```bash
# Apply all configurations
kubectl apply -k k8s/base/

# Or use Helm
helm install fraudlens ./helm/fraudlens \
  --namespace fraudlens \
  --values helm/values.production.yaml
```

#### 4. Configure Ingress
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: fraudlens-ingress
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.fraudlens.com
    secretName: fraudlens-tls
  rules:
  - host: api.fraudlens.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: fraudlens-api
            port:
              number: 8000
```

#### 5. Enable Auto-scaling
```bash
kubectl autoscale deployment fraudlens-api \
  --cpu-percent=70 \
  --min=3 \
  --max=10 \
  -n fraudlens
```

## Cloud Deployment

### AWS Deployment

#### Using AWS Services
```yaml
# Infrastructure components
- EC2/ECS/EKS: Application hosting
- RDS PostgreSQL: Database
- ElastiCache: Redis caching
- S3: File storage
- CloudFront: CDN
- Route53: DNS
- ALB: Load balancing
- CloudWatch: Monitoring
```

#### Terraform Configuration
```hcl
# main.tf
resource "aws_ecs_service" "fraudlens" {
  name            = "fraudlens-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = 3

  deployment_configuration {
    maximum_percent         = 200
    minimum_healthy_percent = 100
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }
}
```

### Google Cloud Platform

#### Using GCP Services
```yaml
- GKE: Kubernetes cluster
- Cloud SQL: PostgreSQL
- Memorystore: Redis
- Cloud Storage: Files
- Cloud CDN: Content delivery
- Cloud Load Balancing: Traffic distribution
- Cloud Monitoring: Observability
```

#### Deployment Script
```bash
# Create GKE cluster
gcloud container clusters create fraudlens-cluster \
  --num-nodes=3 \
  --machine-type=n2-standard-2 \
  --region=us-central1

# Deploy application
kubectl apply -f k8s/

# Expose service
kubectl expose deployment fraudlens-api \
  --type=LoadBalancer \
  --port=80 \
  --target-port=8000
```

### Azure Deployment

#### Using Azure Services
```yaml
- AKS: Kubernetes service
- Azure Database for PostgreSQL
- Azure Cache for Redis
- Blob Storage: Files
- Azure CDN: Content delivery
- Application Gateway: Load balancing
- Azure Monitor: Monitoring
```

## Configuration

### Environment Variables

#### Required Variables
```bash
# Application
FRAUDLENS_ENV=production
SECRET_KEY=<32+ character secret>
JWT_SECRET_KEY=<32+ character secret>

# Database
DATABASE_URL=postgresql://user:pass@host:5432/fraudlens

# Redis
REDIS_URL=redis://:password@host:6379/0

# Gmail API
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

#### Optional Variables
```bash
# Performance
WORKERS=4
MAX_CONNECTIONS=100
CACHE_TTL=3600

# Monitoring
SENTRY_DSN=https://key@sentry.io/project
PROMETHEUS_ENABLED=true

# Features
ENABLE_EMAIL_MONITORING=true
ENABLE_REAL_TIME_PROTECTION=true
```

### Configuration Files

#### nginx.conf
```nginx
upstream fraudlens_api {
    least_conn;
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    listen 443 ssl http2;
    server_name api.fraudlens.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    location / {
        proxy_pass http://fraudlens_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Security

### SSL/TLS Configuration

#### Generate Certificates
```bash
# Self-signed (development)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

# Let's Encrypt (production)
certbot certonly --standalone -d api.fraudlens.com
```

#### Configure HTTPS
```yaml
# docker-compose.yml
services:
  nginx:
    volumes:
      - ./ssl:/etc/nginx/ssl:ro
    ports:
      - "443:443"
```

### Security Checklist

- [ ] Change all default passwords
- [ ] Enable HTTPS/TLS
- [ ] Configure firewall rules
- [ ] Enable rate limiting
- [ ] Set up fail2ban
- [ ] Regular security updates
- [ ] Enable audit logging
- [ ] Implement CORS properly
- [ ] Use secure headers
- [ ] Regular backups

### Firewall Rules
```bash
# Allow only necessary ports
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw allow 5432/tcp  # PostgreSQL (only from app servers)
ufw allow 6379/tcp  # Redis (only from app servers)
ufw enable
```

## Monitoring

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'fraudlens'
    static_configs:
      - targets: ['api:9090']
```

### Grafana Dashboards
Import dashboard JSON from `monitoring/grafana/dashboards/`

### Health Checks
```bash
# API health
curl https://api.fraudlens.com/health

# Database health
docker exec postgres pg_isready

# Redis health
docker exec redis redis-cli ping
```

### Log Aggregation
```yaml
# Using Loki
loki:
  image: grafana/loki:latest
  ports:
    - "3100:3100"

promtail:
  image: grafana/promtail:latest
  volumes:
    - ./logs:/var/log
```

## Backup and Recovery

### Database Backup

#### Automated Backups
```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"

# Backup PostgreSQL
pg_dump $DATABASE_URL > $BACKUP_DIR/postgres_$DATE.sql

# Backup Redis
redis-cli --rdb $BACKUP_DIR/redis_$DATE.rdb

# Upload to S3
aws s3 cp $BACKUP_DIR/ s3://fraudlens-backups/ --recursive
```

#### Schedule with Cron
```cron
0 2 * * * /opt/fraudlens/scripts/backup.sh
```

### Disaster Recovery

#### Recovery Steps
1. Restore infrastructure
2. Restore database from backup
3. Restore Redis cache
4. Verify configuration
5. Test functionality

#### Recovery Commands
```bash
# Restore PostgreSQL
psql $DATABASE_URL < backup.sql

# Restore Redis
redis-cli --rdb restore.rdb
```

## Scaling

### Horizontal Scaling

#### Add API Servers
```bash
# Docker Compose
docker-compose up -d --scale api=5

# Kubernetes
kubectl scale deployment fraudlens-api --replicas=5
```

#### Database Scaling
```sql
-- Add read replicas
CREATE PUBLICATION fraudlens_pub FOR ALL TABLES;

-- On replica
CREATE SUBSCRIPTION fraudlens_sub
  CONNECTION 'host=primary dbname=fraudlens'
  PUBLICATION fraudlens_pub;
```

### Vertical Scaling

#### Increase Resources
```yaml
# Kubernetes
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
  limits:
    memory: "8Gi"
    cpu: "4"
```

### Load Testing
```bash
# Using Apache Bench
ab -n 10000 -c 100 https://api.fraudlens.com/health

# Using k6
k6 run loadtest.js
```

## Maintenance

### Update Process

#### Rolling Update
```bash
# Build new image
docker build -t fraudlens:v2.0.0 .

# Update Kubernetes
kubectl set image deployment/fraudlens-api api=fraudlens:v2.0.0

# Monitor rollout
kubectl rollout status deployment/fraudlens-api
```

#### Database Migrations
```bash
# Run migrations
docker exec api python -m alembic upgrade head

# Rollback if needed
docker exec api python -m alembic downgrade -1
```

### Monitoring Checklist

#### Daily
- [ ] Check error logs
- [ ] Monitor API response times
- [ ] Review security alerts
- [ ] Check disk usage

#### Weekly
- [ ] Review performance metrics
- [ ] Check backup integrity
- [ ] Update dependencies
- [ ] Review access logs

#### Monthly
- [ ] Security audit
- [ ] Performance optimization
- [ ] Capacity planning
- [ ] Update documentation

### Common Commands

```bash
# View logs
docker-compose logs -f api
kubectl logs -f deployment/fraudlens-api

# Restart services
docker-compose restart api
kubectl rollout restart deployment/fraudlens-api

# Enter container
docker exec -it fraudlens-api bash
kubectl exec -it pod/fraudlens-api-xxx -- bash

# Database console
docker exec -it postgres psql -U fraudlens
kubectl exec -it postgres-0 -- psql -U fraudlens

# Redis console
docker exec -it redis redis-cli
kubectl exec -it redis-0 -- redis-cli
```

## Troubleshooting

See [Troubleshooting Guide](../troubleshooting/TROUBLESHOOTING_GUIDE.md) for common issues and solutions.

## Support

- Documentation: https://docs.fraudlens.com
- GitHub Issues: https://github.com/fraudlens/fraudlens/issues
- Email: support@fraudlens.com
- Slack: fraudlens.slack.com