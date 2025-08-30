# FraudLens Troubleshooting Guide

## Quick Diagnostics

Run this diagnostic script to check system status:
```bash
#!/bin/bash
echo "FraudLens System Diagnostic"
echo "=========================="
echo "API Health: $(curl -s http://localhost:8000/health | jq -r .status)"
echo "Database: $(docker exec postgres pg_isready 2>&1)"
echo "Redis: $(docker exec redis redis-cli ping 2>&1)"
echo "Disk Usage: $(df -h / | tail -1 | awk '{print $5}')"
echo "Memory Usage: $(free -h | grep Mem | awk '{print $3"/"$2}')"
echo "Active Containers: $(docker ps --format 'table {{.Names}}\t{{.Status}}' | tail -n +2)"
```

## Common Issues and Solutions

### 1. API Not Responding

#### Symptoms
- API endpoints return timeout errors
- Health check fails
- Cannot connect to service

#### Diagnosis
```bash
# Check if API is running
docker ps | grep fraudlens-api
curl -v http://localhost:8000/health

# Check logs
docker logs fraudlens-api --tail 100

# Check port binding
netstat -tulpn | grep 8000
```

#### Solutions

**Solution 1: Restart API Service**
```bash
docker-compose restart api
# or
kubectl rollout restart deployment/fraudlens-api
```

**Solution 2: Check Configuration**
```bash
# Verify environment variables
docker exec fraudlens-api env | grep FRAUDLENS

# Check config file
docker exec fraudlens-api cat /app/.env
```

**Solution 3: Resource Issues**
```bash
# Check memory
docker stats fraudlens-api

# Increase memory limit
docker update --memory 4g fraudlens-api
```

### 2. Database Connection Errors

#### Symptoms
```
sqlalchemy.exc.OperationalError: could not connect to server
psycopg2.OperationalError: FATAL: password authentication failed
```

#### Diagnosis
```bash
# Test database connection
docker exec -it postgres psql -U fraudlens -c "SELECT 1;"

# Check PostgreSQL logs
docker logs postgres --tail 50

# Verify credentials
echo $DATABASE_URL
```

#### Solutions

**Solution 1: Fix Connection String**
```bash
# Update .env file
DATABASE_URL=postgresql://fraudlens:password@postgres:5432/fraudlens

# Restart API
docker-compose restart api
```

**Solution 2: Reset Database Password**
```bash
# Enter PostgreSQL
docker exec -it postgres psql -U postgres

# Change password
ALTER USER fraudlens WITH PASSWORD 'new_password';

# Update environment
export DATABASE_URL=postgresql://fraudlens:new_password@postgres:5432/fraudlens
```

**Solution 3: Database Not Ready**
```bash
# Wait for database
until docker exec postgres pg_isready; do
  echo "Waiting for database..."
  sleep 2
done
```

### 3. Redis Connection Issues

#### Symptoms
```
redis.exceptions.ConnectionError: Error -2 connecting to redis:6379
```

#### Diagnosis
```bash
# Check Redis status
docker exec redis redis-cli ping

# Check Redis logs
docker logs redis --tail 50

# Test connection
docker exec -it redis redis-cli
```

#### Solutions

**Solution 1: Restart Redis**
```bash
docker-compose restart redis
```

**Solution 2: Fix Redis Password**
```bash
# Set Redis password
docker exec redis redis-cli CONFIG SET requirepass "your_password"

# Update environment
export REDIS_PASSWORD=your_password
export REDIS_URL=redis://:your_password@redis:6379/0
```

**Solution 3: Clear Redis Cache**
```bash
docker exec redis redis-cli FLUSHALL
```

### 4. Gmail API Authentication Errors

#### Symptoms
```
google.auth.exceptions.RefreshError: The credentials do not contain the necessary fields
HttpError 401: Invalid Credentials
```

#### Diagnosis
```bash
# Check credentials file
ls -la ~/.fraudlens/config/credentials.json

# Verify token
cat ~/.fraudlens/config/token.json | jq .

# Test Gmail API
python -m fraudlens.test_gmail_auth
```

#### Solutions

**Solution 1: Re-authenticate**
```bash
# Remove old token
rm ~/.fraudlens/config/token.json

# Re-run authentication
python -m fraudlens.auth.gmail_auth
```

**Solution 2: Update Credentials**
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Navigate to APIs & Services → Credentials
3. Download new credentials.json
4. Replace old credentials file

**Solution 3: Check Scopes**
```python
# Verify scopes in code
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.modify'
]
```

### 5. High Memory Usage

#### Symptoms
- System becomes slow
- Out of memory errors
- Container killed by OOM killer

#### Diagnosis
```bash
# Check memory usage
docker stats

# System memory
free -h

# Top processes
top -o %MEM
```

#### Solutions

**Solution 1: Increase Memory Limits**
```yaml
# docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 8G
```

**Solution 2: Optimize Configuration**
```python
# Reduce worker count
WORKERS=2

# Reduce batch size
BATCH_SIZE=25

# Enable memory profiling
MEMORY_PROFILING=true
```

**Solution 3: Memory Leak Fix**
```bash
# Restart periodically
0 */6 * * * docker-compose restart api
```

### 6. Slow Performance

#### Symptoms
- API response times > 1 second
- Timeout errors
- Queue backlog

#### Diagnosis
```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health

# Monitor CPU
docker stats --no-stream

# Check database queries
docker exec postgres psql -U fraudlens -c "SELECT * FROM pg_stat_activity;"
```

#### Solutions

**Solution 1: Enable Caching**
```python
# Increase cache TTL
CACHE_TTL_SECONDS=7200

# Enable Redis caching
REDIS_ENABLED=true
```

**Solution 2: Database Optimization**
```sql
-- Add indexes
CREATE INDEX idx_emails_date ON emails(created_at);
CREATE INDEX idx_fraud_user ON fraud_detections(user_id);

-- Analyze tables
ANALYZE emails;
VACUUM ANALYZE fraud_detections;
```

**Solution 3: Scale Services**
```bash
# Add more workers
docker-compose up -d --scale worker=4

# Add more API instances
docker-compose up -d --scale api=3
```

### 7. Email Not Scanning

#### Symptoms
- No emails being processed
- Scan results empty
- Monitor not detecting new emails

#### Diagnosis
```bash
# Check monitor status
curl http://localhost:8000/email/monitor/status

# Check worker logs
docker logs fraudlens-worker --tail 100

# Verify Gmail quota
curl "https://www.googleapis.com/gmail/v1/users/me/profile" \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

#### Solutions

**Solution 1: Restart Monitor**
```bash
# Stop monitoring
curl -X POST http://localhost:8000/email/monitor/stop

# Start monitoring
curl -X POST http://localhost:8000/email/monitor/start \
  -H "Content-Type: application/json" \
  -d '{"check_interval": 300}'
```

**Solution 2: Check Rate Limits**
```python
# Reduce scan frequency
CHECK_INTERVAL=600  # 10 minutes

# Reduce batch size
MAX_EMAILS_PER_SCAN=25
```

**Solution 3: Fix Permissions**
```bash
# Re-authorize Gmail
python -m fraudlens.auth.gmail_auth --scopes all
```

### 8. Docker Issues

#### Symptoms
- Containers not starting
- Docker daemon errors
- Volume mount issues

#### Diagnosis
```bash
# Check Docker status
systemctl status docker
docker version

# Check disk space
df -h /var/lib/docker

# Check container logs
docker-compose logs
```

#### Solutions

**Solution 1: Clean Docker**
```bash
# Remove unused containers
docker container prune -f

# Remove unused images
docker image prune -a -f

# Clean build cache
docker builder prune -f

# Clean everything
docker system prune -a -f --volumes
```

**Solution 2: Restart Docker**
```bash
sudo systemctl restart docker
docker-compose down
docker-compose up -d
```

**Solution 3: Fix Permissions**
```bash
# Fix volume permissions
sudo chown -R $USER:$USER ./data
sudo chmod -R 755 ./data
```

### 9. SSL/TLS Certificate Issues

#### Symptoms
- HTTPS not working
- Certificate expired warnings
- Mixed content errors

#### Diagnosis
```bash
# Check certificate
openssl s_client -connect api.fraudlens.com:443 -servername api.fraudlens.com

# Check expiry
echo | openssl s_client -connect api.fraudlens.com:443 2>/dev/null | openssl x509 -noout -dates
```

#### Solutions

**Solution 1: Renew Certificate**
```bash
# Let's Encrypt renewal
certbot renew

# Restart nginx
docker-compose restart nginx
```

**Solution 2: Fix Certificate Path**
```nginx
# nginx.conf
ssl_certificate /etc/nginx/ssl/fullchain.pem;
ssl_certificate_key /etc/nginx/ssl/privkey.pem;
```

### 10. Kubernetes Issues

#### Symptoms
- Pods crashing
- Services unreachable
- Deployment failures

#### Diagnosis
```bash
# Check pod status
kubectl get pods -n fraudlens

# Describe problematic pod
kubectl describe pod fraudlens-api-xxx -n fraudlens

# Check events
kubectl get events -n fraudlens --sort-by='.lastTimestamp'

# Check logs
kubectl logs -f deployment/fraudlens-api -n fraudlens
```

#### Solutions

**Solution 1: Fix Image Pull**
```bash
# Check secret
kubectl get secret regcred -n fraudlens

# Recreate secret
kubectl create secret docker-registry regcred \
  --docker-server=docker.io \
  --docker-username=username \
  --docker-password=password \
  -n fraudlens
```

**Solution 2: Fix Resource Limits**
```yaml
# Increase limits
resources:
  requests:
    memory: "2Gi"
    cpu: "1"
  limits:
    memory: "4Gi"
    cpu: "2"
```

**Solution 3: Fix Persistent Volume**
```bash
# Check PVC
kubectl get pvc -n fraudlens

# Delete and recreate
kubectl delete pvc data-pvc -n fraudlens
kubectl apply -f k8s/base/pvc.yaml
```

## Error Messages Reference

### Application Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'fraudlens'` | Missing installation | `pip install -e .` |
| `RuntimeError: Model not found` | Models not downloaded | `python -m fraudlens.models.download` |
| `ValueError: Invalid configuration` | Bad config file | Check `.env` file |
| `PermissionError: [Errno 13]` | File permissions | `chmod 755 file` |

### API Errors

| Status | Error | Solution |
|--------|-------|----------|
| 401 | Unauthorized | Check API token |
| 403 | Forbidden | Check user permissions |
| 429 | Rate limit exceeded | Wait or increase limits |
| 500 | Internal server error | Check logs |
| 503 | Service unavailable | Check dependencies |

### Database Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `FATAL: database does not exist` | Missing database | `CREATE DATABASE fraudlens;` |
| `FATAL: role does not exist` | Missing user | `CREATE USER fraudlens;` |
| `FATAL: too many connections` | Connection limit | Increase `max_connections` |
| `ERROR: deadlock detected` | Transaction conflict | Retry transaction |

## Performance Tuning

### Database Optimization
```sql
-- PostgreSQL tuning
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
SELECT pg_reload_conf();
```

### Redis Optimization
```bash
# Redis configuration
redis-cli CONFIG SET maxmemory 4gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
redis-cli CONFIG SET tcp-keepalive 60
redis-cli CONFIG SET timeout 300
redis-cli CONFIG REWRITE
```

### API Optimization
```python
# Gunicorn settings
WORKERS = multiprocessing.cpu_count() * 2 + 1
WORKER_CLASS = "uvicorn.workers.UvicornWorker"
WORKER_CONNECTIONS = 1000
MAX_REQUESTS = 1000
MAX_REQUESTS_JITTER = 50
TIMEOUT = 30
KEEPALIVE = 2
```

## Monitoring Commands

### System Health
```bash
# Overall health check
curl http://localhost:8000/health | jq .

# Component health
curl http://localhost:8000/health/database | jq .
curl http://localhost:8000/health/redis | jq .
curl http://localhost:8000/health/gmail | jq .
```

### Metrics
```bash
# Prometheus metrics
curl http://localhost:9090/metrics

# Application metrics
curl http://localhost:8000/metrics
```

### Logs
```bash
# Tail all logs
docker-compose logs -f

# Specific service
docker-compose logs -f api

# Search logs
docker-compose logs | grep ERROR

# JSON logs
docker-compose logs api | jq .
```

## Recovery Procedures

### Complete System Recovery
```bash
#!/bin/bash
# recovery.sh

echo "Starting FraudLens recovery..."

# 1. Stop all services
docker-compose down

# 2. Restore database
psql $DATABASE_URL < backup/latest.sql

# 3. Clear Redis
redis-cli FLUSHALL

# 4. Restore configuration
cp backup/.env.production .env

# 5. Rebuild images
docker-compose build

# 6. Start services
docker-compose up -d

# 7. Run migrations
docker-compose exec api python -m alembic upgrade head

# 8. Verify health
sleep 30
curl http://localhost:8000/health

echo "Recovery complete!"
```

## Getting Help

### Log Collection for Support
```bash
#!/bin/bash
# collect-logs.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="fraudlens-logs-$TIMESTAMP"

mkdir -p $LOGDIR

# Collect logs
docker-compose logs > $LOGDIR/docker-compose.log
docker ps -a > $LOGDIR/containers.log
docker images > $LOGDIR/images.log
df -h > $LOGDIR/disk.log
free -h > $LOGDIR/memory.log
env | grep -E "FRAUDLENS|DATABASE|REDIS" > $LOGDIR/environment.log

# Compress
tar -czf $LOGDIR.tar.gz $LOGDIR

echo "Logs collected in $LOGDIR.tar.gz"
```

### Support Channels

- **Documentation**: https://docs.fraudlens.com
- **GitHub Issues**: https://github.com/fraudlens/fraudlens/issues
- **Email Support**: support@fraudlens.com
- **Community Forum**: https://community.fraudlens.com
- **Slack**: fraudlens.slack.com

### Information to Provide

When reporting issues, include:
1. Error messages and stack traces
2. Steps to reproduce
3. Environment details (OS, Docker version, etc.)
4. Configuration (sanitized .env file)
5. Relevant logs
6. System diagnostics output