# FraudLens Production Deployment Guide

## Overview
This guide covers deploying FraudLens to production using Docker, Docker Compose, or Kubernetes.

## Prerequisites
- Docker 20.10+
- Docker Compose 2.0+ (for Docker Compose deployment)
- kubectl 1.24+ (for Kubernetes deployment)
- Access to container registry (Docker Hub, AWS ECR, etc.)
- SSL certificates (for HTTPS)

## Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/yourusername/FraudLens.git
cd FraudLens
cp .env.example .env.production
# Edit .env.production with your values
```

### 2. Build Docker Image
```bash
# Build production image
docker build -f Dockerfile.production -t fraudlens:latest .

# Or use the deploy script
./scripts/deploy.sh -e production -b
```

### 3. Deploy with Docker Compose
```bash
# Start all services
docker-compose -f docker-compose.production.yml up -d

# Or use the deploy script
./scripts/deploy.sh -e production -t compose
```

## Deployment Options

### Option 1: Docker Compose (Recommended for Small/Medium Scale)

#### Start Services
```bash
# Load environment variables
export $(cat .env.production | grep -v '^#' | xargs)

# Start services
docker-compose -f docker-compose.production.yml up -d

# Check status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f api
```

#### Scale Services
```bash
# Scale API servers
docker-compose -f docker-compose.production.yml up -d --scale api=4

# Scale workers
docker-compose -f docker-compose.production.yml up -d --scale worker=3
```

#### Update Services
```bash
# Pull latest images
docker-compose -f docker-compose.production.yml pull

# Restart with new images
docker-compose -f docker-compose.production.yml up -d --force-recreate
```

### Option 2: Kubernetes (Recommended for Large Scale)

#### Prerequisites
```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Configure kubectl
kubectl config set-cluster production --server=https://your-cluster-api
kubectl config use-context production
```

#### Deploy to Kubernetes
```bash
# Create namespace
kubectl create namespace fraudlens

# Create secrets
kubectl create secret generic fraudlens-secret \
  --from-env-file=.env.production \
  -n fraudlens

# Apply configurations
kubectl apply -k k8s/base/

# Check deployment
kubectl -n fraudlens get pods
kubectl -n fraudlens get services
kubectl -n fraudlens get ingress
```

#### Scale in Kubernetes
```bash
# Scale API deployment
kubectl -n fraudlens scale deployment fraudlens-api --replicas=5

# Enable autoscaling
kubectl -n fraudlens autoscale deployment fraudlens-api \
  --min=3 --max=10 --cpu-percent=70
```

#### Update in Kubernetes
```bash
# Update image
kubectl -n fraudlens set image deployment/fraudlens-api \
  api=fraudlens:v2.0.0

# Rolling update
kubectl -n fraudlens rollout status deployment/fraudlens-api

# Rollback if needed
kubectl -n fraudlens rollout undo deployment/fraudlens-api
```

### Option 3: Single Docker Container (Development/Testing)

```bash
# Run API only
docker run -d \
  --name fraudlens-api \
  --env-file .env.production \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  fraudlens:latest

# Run with external Redis and PostgreSQL
docker run -d \
  --name fraudlens-api \
  --env-file .env.production \
  --env DATABASE_URL=postgresql://user:pass@host:5432/db \
  --env REDIS_URL=redis://:pass@host:6379/0 \
  -p 8000:8000 \
  fraudlens:latest
```

## Environment Configuration

### Required Environment Variables
```bash
# Security (MUST CHANGE IN PRODUCTION)
SECRET_KEY=generate-with-openssl-rand-hex-32
JWT_SECRET_KEY=generate-with-openssl-rand-hex-32

# Database
POSTGRES_PASSWORD=secure-database-password
DATABASE_URL=postgresql://user:pass@host:5432/fraudlens

# Redis
REDIS_PASSWORD=secure-redis-password
REDIS_URL=redis://:pass@host:6379/0

# API
API_RATE_LIMIT=5000
API_CORS_ORIGINS=["https://yourdomain.com"]
```

### Generate Secure Keys
```bash
# Generate SECRET_KEY
openssl rand -hex 32

# Generate JWT_SECRET_KEY
openssl rand -hex 32

# Generate passwords
openssl rand -base64 32
```

## SSL/TLS Configuration

### Using Let's Encrypt (Recommended)
```bash
# Install cert-manager (Kubernetes)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Create ClusterIssuer
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@yourdomain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### Using Custom Certificates
```bash
# Create TLS secret
kubectl create secret tls fraudlens-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key \
  -n fraudlens
```

## Database Management

### Initial Setup
```bash
# Create database
docker exec -it fraudlens-postgres psql -U postgres -c "CREATE DATABASE fraudlens;"

# Run migrations
docker exec -it fraudlens-api python -m fraudlens.db.migrate

# Create admin user
docker exec -it fraudlens-api python -m fraudlens.db.create_admin
```

### Backup and Restore
```bash
# Backup database
docker exec fraudlens-postgres pg_dump -U fraudlens fraudlens > backup.sql

# Restore database
docker exec -i fraudlens-postgres psql -U fraudlens fraudlens < backup.sql

# Backup Redis
docker exec fraudlens-redis redis-cli --rdb /data/dump.rdb
```

## Monitoring

### Access Monitoring Services
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- API Metrics: http://localhost:8000/metrics

### Setup Alerts
```yaml
# prometheus-alerts.yml
groups:
- name: fraudlens
  rules:
  - alert: HighCPUUsage
    expr: rate(process_cpu_seconds_total[5m]) > 0.8
    for: 5m
    annotations:
      summary: High CPU usage detected
      
  - alert: HighMemoryUsage
    expr: process_resident_memory_bytes / 1024 / 1024 / 1024 > 7
    for: 5m
    annotations:
      summary: High memory usage detected
```

## Health Checks

### API Health
```bash
# Check API health
curl http://localhost:8000/health

# Check specific service
curl http://localhost:8000/health/redis
curl http://localhost:8000/health/postgres
```

### System Health
```bash
# Check all services (Docker Compose)
docker-compose -f docker-compose.production.yml ps

# Check pods (Kubernetes)
kubectl -n fraudlens get pods
kubectl -n fraudlens describe pod fraudlens-api-xxx
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Error
```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Test connection
docker exec -it fraudlens-postgres psql -U fraudlens -c "SELECT 1;"

# Check logs
docker logs fraudlens-postgres
```

#### 2. Redis Connection Error
```bash
# Check Redis is running
docker ps | grep redis

# Test connection
docker exec -it fraudlens-redis redis-cli ping

# Check logs
docker logs fraudlens-redis
```

#### 3. API Not Responding
```bash
# Check API logs
docker logs fraudlens-api

# Restart API
docker restart fraudlens-api

# Check resource usage
docker stats fraudlens-api
```

#### 4. High Memory Usage
```bash
# Check memory usage
docker stats

# Increase memory limits
docker update --memory 8g fraudlens-api

# Or in docker-compose.yml:
# deploy:
#   resources:
#     limits:
#       memory: 8G
```

### Debug Mode
```bash
# Enable debug mode (temporary)
docker exec -it fraudlens-api bash
export FRAUDLENS_DEBUG=true
python -m uvicorn fraudlens.api.secured_api:app --reload
```

## Performance Tuning

### API Optimization
```bash
# Increase workers
API_WORKERS=8

# Enable response caching
CACHE_TTL_SECONDS=7200

# Increase rate limits
API_RATE_LIMIT=10000
```

### Database Optimization
```sql
-- PostgreSQL tuning
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET work_mem = '16MB';
SELECT pg_reload_conf();
```

### Redis Optimization
```bash
# Redis configuration
redis-cli CONFIG SET maxmemory 4gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
redis-cli CONFIG REWRITE
```

## Security Best Practices

1. **Change Default Passwords**: Never use default passwords in production
2. **Use HTTPS**: Always use SSL/TLS certificates
3. **Network Isolation**: Use private networks for internal services
4. **Regular Updates**: Keep all components updated
5. **Backup Regularly**: Automate database and data backups
6. **Monitor Access**: Log and monitor all API access
7. **Rate Limiting**: Enforce rate limits to prevent abuse
8. **Input Validation**: Validate all user inputs
9. **Secrets Management**: Use secret management tools (Vault, AWS Secrets Manager)
10. **Audit Logs**: Enable and monitor audit logs

## Maintenance

### Regular Tasks
```bash
# Daily: Check logs and metrics
./scripts/check_health.sh

# Weekly: Backup databases
./scripts/backup.sh

# Monthly: Update dependencies
docker pull fraudlens:latest
kubectl rollout restart deployment -n fraudlens

# Quarterly: Security audit
./scripts/security_audit.sh
```

### Update Process
1. Test in staging environment
2. Backup production database
3. Deploy to one instance
4. Monitor for issues
5. Complete rolling update
6. Verify all services

## Support

For issues or questions:
- Documentation: https://fraudlens.docs
- Issues: https://github.com/yourusername/FraudLens/issues
- Email: support@fraudlens.com