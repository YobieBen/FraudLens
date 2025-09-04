"""
FraudLens Production Server
High-performance server with authentication, rate limiting, and monitoring
"""

import os
import jwt
import time
import uuid
import redis
import asyncio
import hashlib
import secrets
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

from fastapi import FastAPI, HTTPException, Depends, Security, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from loguru import logger

# Import FraudLens components
from fraudlens.core.pipeline import FraudDetectionPipeline
from fraudlens.core.config import Config
from fraudlens.monitoring.monitor import FraudLensMonitor


# Metrics
request_count = Counter(
    "fraudlens_requests_total", "Total requests", ["method", "endpoint", "status"]
)
request_duration = Histogram(
    "fraudlens_request_duration_seconds", "Request duration", ["method", "endpoint"]
)
active_connections = Gauge("fraudlens_active_connections", "Active connections")
queue_size = Gauge("fraudlens_queue_size", "Background queue size")
model_accuracy = Gauge("fraudlens_model_accuracy", "Model accuracy score")
cache_hits = Counter("fraudlens_cache_hits_total", "Cache hits")
cache_misses = Counter("fraudlens_cache_misses_total", "Cache misses")


@dataclass
class JobStatus:
    """Background job status"""

    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    created_at: datetime
    updated_at: datetime
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: float = 0.0


@dataclass
class HealthStatus:
    """System health status"""

    status: str  # "healthy", "degraded", "unhealthy"
    version: str
    uptime_seconds: float
    services: Dict[str, bool]
    metrics: Dict[str, float]
    errors: List[str]


class AuthManager:
    """Handle authentication and authorization"""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry = timedelta(hours=24)
        self.refresh_expiry = timedelta(days=7)

        # User store (in production, use database)
        self.users = {}
        self.api_keys = {}
        self.sessions = {}

    def create_user(self, username: str, password: str, role: str = "user") -> str:
        """Create new user"""
        user_id = str(uuid.uuid4())
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        self.users[username] = {
            "user_id": user_id,
            "password_hash": password_hash,
            "role": role,
            "created_at": datetime.utcnow(),
            "api_keys": [],
        }

        return user_id

    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return token"""
        if username not in self.users:
            return None

        user = self.users[username]
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        if password_hash != user["password_hash"]:
            return None

        # Create JWT token
        payload = {
            "user_id": user["user_id"],
            "username": username,
            "role": user["role"],
            "exp": datetime.utcnow() + self.token_expiry,
            "iat": datetime.utcnow(),
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

        # Store session
        self.sessions[token] = {
            "user_id": user["user_id"],
            "username": username,
            "created_at": datetime.utcnow(),
        }

        return token

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Check if session exists
            if token not in self.sessions:
                return None

            return payload
        except jwt.ExpiredSignatureError:
            # Remove expired session
            if token in self.sessions:
                del self.sessions[token]
            return None
        except jwt.InvalidTokenError:
            return None

    def create_api_key(self, user_id: str, name: str = "default") -> str:
        """Create API key for user"""
        api_key = f"sk-{secrets.token_urlsafe(32)}"

        self.api_keys[api_key] = {
            "user_id": user_id,
            "name": name,
            "created_at": datetime.utcnow(),
            "last_used": None,
            "usage_count": 0,
        }

        return api_key

    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key"""
        if api_key not in self.api_keys:
            return None

        # Update usage stats
        self.api_keys[api_key]["last_used"] = datetime.utcnow()
        self.api_keys[api_key]["usage_count"] += 1

        return self.api_keys[api_key]


class RateLimiter:
    """Custom rate limiter with Redis backend"""

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
            except:
                logger.warning("Redis connection failed, using in-memory rate limiting")

        # Fallback to in-memory if Redis not available
        if not self.redis_client:
            self.memory_store = defaultdict(deque)

    def check_rate_limit(self, key: str, limit: int, window_seconds: int) -> Tuple[bool, int]:
        """
        Check if request is within rate limit
        Returns: (allowed, remaining_requests)
        """
        current_time = time.time()

        if self.redis_client:
            # Redis-based rate limiting
            pipe = self.redis_client.pipeline()
            pipe.zadd(key, {str(uuid.uuid4()): current_time})
            pipe.zremrangebyscore(key, 0, current_time - window_seconds)
            pipe.zcard(key)
            pipe.expire(key, window_seconds + 1)
            results = pipe.execute()

            request_count = results[2]
            remaining = max(0, limit - request_count)

            return request_count <= limit, remaining
        else:
            # In-memory rate limiting
            requests = self.memory_store[key]

            # Remove old requests
            while requests and requests[0] < current_time - window_seconds:
                requests.popleft()

            remaining = max(0, limit - len(requests))

            if len(requests) < limit:
                requests.append(current_time)
                return True, remaining

            return False, 0


class BackgroundProcessor:
    """Handle async background processing"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.queue = asyncio.Queue()
        self.jobs = {}
        self.workers = []

    async def start(self):
        """Start background workers"""
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)

    async def stop(self):
        """Stop background workers"""
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)

    async def _worker(self, worker_id: int):
        """Background worker"""
        logger.info(f"Worker {worker_id} started")

        while True:
            try:
                job_id, func, args, kwargs = await self.queue.get()

                # Update job status
                if job_id in self.jobs:
                    self.jobs[job_id].status = "processing"
                    self.jobs[job_id].updated_at = datetime.utcnow()

                # Process job
                try:
                    result = await func(*args, **kwargs)

                    if job_id in self.jobs:
                        self.jobs[job_id].status = "completed"
                        self.jobs[job_id].result = result
                        self.jobs[job_id].updated_at = datetime.utcnow()
                        self.jobs[job_id].progress = 1.0
                except Exception as e:
                    logger.error(f"Job {job_id} failed: {e}")

                    if job_id in self.jobs:
                        self.jobs[job_id].status = "failed"
                        self.jobs[job_id].error = str(e)
                        self.jobs[job_id].updated_at = datetime.utcnow()

                queue_size.set(self.queue.qsize())

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    async def submit_job(self, func, *args, **kwargs) -> str:
        """Submit job for background processing"""
        job_id = str(uuid.uuid4())

        # Create job status
        self.jobs[job_id] = JobStatus(
            job_id=job_id,
            status="pending",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        # Add to queue
        await self.queue.put((job_id, func, args, kwargs))
        queue_size.set(self.queue.qsize())

        return job_id

    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get job status"""
        return self.jobs.get(job_id)


class ProductionServer:
    """
    Production-ready FraudLens server
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.app = FastAPI(
            title="FraudLens API", version="1.0.0", docs_url="/api/docs", redoc_url="/api/redoc"
        )

        # Initialize components
        self.pipeline = FraudDetectionPipeline(self.config)
        self.monitor = FraudLensMonitor()

        # Auth manager
        secret_key = os.getenv("FRAUDLENS_SECRET_KEY", secrets.token_urlsafe(32))
        self.auth = AuthManager(secret_key)

        # Rate limiter
        redis_url = os.getenv("REDIS_URL")
        self.rate_limiter = RateLimiter(redis_url)
        self.limiter = Limiter(key_func=get_remote_address)

        # Background processor
        self.processor = BackgroundProcessor(max_workers=4)

        # Cache
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes

        # Security
        self.security = HTTPBearer()

        # Server state
        self.start_time = datetime.utcnow()

        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()

    def _setup_middleware(self):
        """Configure middleware"""

        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure based on environment
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Trusted hosts
        self.app.add_middleware(
            TrustedHostMiddleware, allowed_hosts=["*.fraudlens.com", "localhost", "127.0.0.1"]
        )

        # Rate limit error handler
        self.app.state.limiter = self.limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

        # Request logging middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()

            # Increment active connections
            active_connections.inc()

            try:
                response = await call_next(request)

                # Log request
                duration = time.time() - start_time
                request_count.labels(
                    method=request.method, endpoint=request.url.path, status=response.status_code
                ).inc()
                request_duration.labels(method=request.method, endpoint=request.url.path).observe(
                    duration
                )

                # Add timing header
                response.headers["X-Process-Time"] = str(duration)

                return response
            finally:
                active_connections.dec()

    def _setup_routes(self):
        """Setup API routes"""

        # Health check
        @self.app.get("/health")
        async def health_check() -> HealthStatus:
            """Health check endpoint"""
            uptime = (datetime.utcnow() - self.start_time).total_seconds()

            # Check services
            services = {
                "pipeline": self.pipeline.is_initialized,
                "monitor": True,
                "cache": True,
                "background": len(self.processor.workers) > 0,
            }

            # Get metrics
            metrics = {
                "uptime_hours": uptime / 3600,
                "active_connections": active_connections._value.get(),
                "queue_size": queue_size._value.get(),
                "cache_size": len(self.cache),
            }

            # Determine overall status
            if all(services.values()):
                status = "healthy"
            elif any(services.values()):
                status = "degraded"
            else:
                status = "unhealthy"

            return HealthStatus(
                status=status,
                version="1.0.0",
                uptime_seconds=uptime,
                services=services,
                metrics=metrics,
                errors=[],
            )

        # Ready check
        @self.app.get("/ready")
        async def ready_check():
            """Readiness probe"""
            if self.pipeline.is_initialized:
                return {"ready": True}
            else:
                raise HTTPException(status_code=503, detail="Service not ready")

        # Metrics endpoint
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics"""
            return generate_latest()

        # Authentication endpoints
        @self.app.post("/api/v1/auth/login")
        async def login(username: str, password: str):
            """User login"""
            token = self.auth.authenticate(username, password)
            if not token:
                raise HTTPException(status_code=401, detail="Invalid credentials")

            return {"access_token": token, "token_type": "bearer"}

        @self.app.post("/api/v1/auth/register")
        async def register(username: str, password: str, email: str):
            """User registration"""
            user_id = self.auth.create_user(username, password)
            api_key = self.auth.create_api_key(user_id)

            return {"user_id": user_id, "api_key": api_key, "message": "User created successfully"}

        # Main detection endpoint
        @self.app.post("/api/v1/detect")
        @self.limiter.limit("100/minute")
        async def detect_fraud(
            request: Request,
            data: Dict[str, Any],
            background_tasks: BackgroundTasks,
            auth: HTTPAuthorizationCredentials = Security(self.security),
        ):
            """Detect fraud in provided data"""

            # Verify authentication
            token_data = self.auth.verify_token(auth.credentials)
            if not token_data:
                # Try API key
                api_key_data = self.auth.verify_api_key(auth.credentials)
                if not api_key_data:
                    raise HTTPException(status_code=401, detail="Invalid authentication")

            # Check cache
            cache_key = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
            if cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                    cache_hits.inc()
                    return cache_entry["result"]

            cache_misses.inc()

            # Run detection
            try:
                result = await self.pipeline.detect(data)

                # Cache result
                self.cache[cache_key] = {"timestamp": time.time(), "result": result}

                # Log detection
                background_tasks.add_task(
                    self.monitor.record_detection,
                    detector_id="api",
                    fraud_score=result.get("fraud_score", 0),
                    latency_ms=0,
                    success=True,
                )

                return result

            except Exception as e:
                logger.error(f"Detection error: {e}")
                raise HTTPException(status_code=500, detail="Detection failed")

        # Async detection endpoint
        @self.app.post("/api/v1/detect/async")
        @self.limiter.limit("50/minute")
        async def detect_fraud_async(
            request: Request,
            data: Dict[str, Any],
            auth: HTTPAuthorizationCredentials = Security(self.security),
        ):
            """Submit fraud detection job"""

            # Verify authentication
            token_data = self.auth.verify_token(auth.credentials)
            if not token_data:
                raise HTTPException(status_code=401, detail="Invalid authentication")

            # Submit job
            job_id = await self.processor.submit_job(self.pipeline.detect, data)

            return {"job_id": job_id, "status": "pending", "message": "Job submitted successfully"}

        # Job status endpoint
        @self.app.get("/api/v1/jobs/{job_id}")
        async def get_job_status(
            job_id: str, auth: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Get job status"""

            # Verify authentication
            token_data = self.auth.verify_token(auth.credentials)
            if not token_data:
                raise HTTPException(status_code=401, detail="Invalid authentication")

            job_status = self.processor.get_job_status(job_id)
            if not job_status:
                raise HTTPException(status_code=404, detail="Job not found")

            return asdict(job_status)

        # Webhook endpoint
        @self.app.post("/api/v1/webhooks/{webhook_id}")
        async def handle_webhook(
            webhook_id: str, request: Request, background_tasks: BackgroundTasks
        ):
            """Handle webhook callbacks"""

            # Verify webhook signature
            signature = request.headers.get("X-Webhook-Signature")
            if not self._verify_webhook_signature(webhook_id, signature, await request.body()):
                raise HTTPException(status_code=401, detail="Invalid webhook signature")

            # Process webhook in background
            data = await request.json()
            background_tasks.add_task(self._process_webhook, webhook_id, data)

            return {"status": "accepted"}

        # Model management
        @self.app.post("/api/v1/models/reload")
        async def reload_models(auth: HTTPAuthorizationCredentials = Security(self.security)):
            """Reload models (admin only)"""

            # Verify authentication
            token_data = self.auth.verify_token(auth.credentials)
            if not token_data or token_data.get("role") != "admin":
                raise HTTPException(status_code=403, detail="Admin access required")

            # Reload models
            await self.pipeline.reload_models()

            return {"status": "success", "message": "Models reloaded"}

        # Feature flags
        @self.app.get("/api/v1/features")
        async def get_features():
            """Get feature flags"""
            return self.config.feature_flags

        @self.app.put("/api/v1/features/{feature}")
        async def update_feature(
            feature: str,
            enabled: bool,
            auth: HTTPAuthorizationCredentials = Security(self.security),
        ):
            """Update feature flag (admin only)"""

            # Verify authentication
            token_data = self.auth.verify_token(auth.credentials)
            if not token_data or token_data.get("role") != "admin":
                raise HTTPException(status_code=403, detail="Admin access required")

            self.config.feature_flags[feature] = enabled

            return {"feature": feature, "enabled": enabled}

    def _verify_webhook_signature(self, webhook_id: str, signature: str, body: bytes) -> bool:
        """Verify webhook signature"""
        # Implement HMAC verification
        expected_signature = hashlib.sha256(f"{webhook_id}:{body.decode()}".encode()).hexdigest()

        return signature == expected_signature

    async def _process_webhook(self, webhook_id: str, data: Dict[str, Any]):
        """Process webhook data"""
        logger.info(f"Processing webhook {webhook_id}: {data}")
        # Implement webhook processing logic

    async def start_server(self, host: str = "0.0.0.0", port: int = 8000, workers: int = 4):
        """Start production server"""

        # Startup event
        @self.app.on_event("startup")
        async def startup():
            logger.info("Starting FraudLens production server...")
            await self.pipeline.initialize()
            await self.processor.start()
            logger.info("Server started successfully")

        # Shutdown event
        @self.app.on_event("shutdown")
        async def shutdown():
            logger.info("Shutting down server...")
            await self.processor.stop()
            await self.pipeline.cleanup()
            logger.info("Server shutdown complete")

        # Run server
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            workers=workers,
            log_level="info",
            access_log=True,
            use_colors=True,
            server_header=False,  # Hide server header for security
            date_header=False,
            forwarded_allow_ips="*",
            proxy_headers=True,
        )

        server = uvicorn.Server(config)
        await server.serve()

    def run(self, host: str = "0.0.0.0", port: int = 8000, workers: int = 4):
        """Run server (blocking)"""
        asyncio.run(self.start_server(host, port, workers))


# Request/Response models
class DetectionRequest(BaseModel):
    """Fraud detection request"""

    data: Dict[str, Any] = Field(..., description="Data to analyze")
    mode: str = Field("sync", description="Processing mode (sync/async)")
    webhook_url: Optional[str] = Field(None, description="Webhook for async results")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class DetectionResponse(BaseModel):
    """Fraud detection response"""

    fraud_score: float = Field(..., ge=0, le=1, description="Fraud probability")
    risk_level: str = Field(..., description="Risk level (low/medium/high/critical)")
    evidence: List[Dict[str, Any]] = Field(..., description="Evidence of fraud")
    recommendations: List[str] = Field(..., description="Recommended actions")
    processing_time_ms: float = Field(..., description="Processing time")


class JobResponse(BaseModel):
    """Background job response"""

    job_id: str = Field(..., description="Unique job ID")
    status: str = Field(..., description="Job status")
    progress: float = Field(0.0, ge=0, le=1, description="Progress percentage")
    created_at: datetime = Field(..., description="Creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    result: Optional[Dict[str, Any]] = Field(None, description="Job result")
    error: Optional[str] = Field(None, description="Error message if failed")


if __name__ == "__main__":
    # Initialize and run server
    server = ProductionServer()
    server.run()
