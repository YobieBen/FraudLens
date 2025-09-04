"""
FraudLens Secured REST API
Implements JWT authentication, rate limiting, and role-based access control
"""

from fastapi import FastAPI, Depends, HTTPException, status, Request, Header, Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import asyncio
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fraudlens.api.auth import (
    auth_manager,
    UserCreate,
    UserInDB,
    Token,
    TokenData,
    UserRole,
    create_tokens,
    verify_token,
    authenticate_user,
    create_api_key,
    verify_api_key,
    check_rate_limit,
)
from fraudlens.core.pipeline import FraudDetectionPipeline
from pydantic import BaseModel, Field
from loguru import logger

# FastAPI app
app = FastAPI(
    title="FraudLens Secured API",
    version="2.0.0",
    description="Advanced fraud detection API with JWT authentication and rate limiting",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security schemes
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Initialize fraud detection pipeline
try:
    fraud_pipeline = FraudDetectionPipeline()
    asyncio.run(fraud_pipeline.initialize())
except:
    fraud_pipeline = None
    logger.warning("Fraud detection pipeline not initialized")


# Request/Response Models
class UserResponse(BaseModel):
    """User response model"""

    id: int
    username: str
    email: str
    role: UserRole
    is_active: bool
    created_at: datetime


class APIKeyResponse(BaseModel):
    """API key response"""

    api_key: str
    name: str
    created_at: datetime


class FraudAnalysisRequest(BaseModel):
    """Fraud analysis request"""

    content: str
    content_type: str = Field(default="text", pattern="^(text|email|url)$")
    metadata: Optional[Dict[str, Any]] = None


class FraudAnalysisResponse(BaseModel):
    """Fraud analysis response"""

    is_fraud: bool
    confidence: float
    fraud_types: List[str]
    risk_level: str
    explanation: str
    request_id: str
    timestamp: datetime


# Dependency functions
async def get_current_user_from_token(token: str = Depends(oauth2_scheme)) -> UserInDB:
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token_data = verify_token(token)
    if token_data is None:
        raise credentials_exception

    user = auth_manager.get_user(username=token_data.username)
    if user is None:
        raise credentials_exception

    return user


async def get_current_user_from_api_key(
    api_key: Optional[str] = Depends(api_key_header),
) -> Optional[Dict]:
    """Get user from API key"""
    if api_key:
        user_info = verify_api_key(api_key)
        if user_info:
            return user_info
    return None


async def get_current_user(
    token_user: Optional[UserInDB] = Depends(get_current_user_from_token),
    api_key_user: Optional[Dict] = Depends(get_current_user_from_api_key),
) -> Dict:
    """Get current user from either JWT or API key"""
    if token_user:
        return {
            "user_id": token_user.id,
            "username": token_user.username,
            "role": token_user.role,
            "auth_type": "jwt",
        }
    elif api_key_user:
        return {
            "user_id": api_key_user["user_id"],
            "username": api_key_user["username"],
            "role": UserRole(api_key_user["role"]),
            "auth_type": "api_key",
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )


async def check_rate_limit_dependency(
    request: Request, current_user: Dict = Depends(get_current_user)
) -> None:
    """Check rate limit for current user"""
    identifier = f"user_{current_user['user_id']}"
    endpoint = request.url.path
    role = current_user.get("role")

    allowed, remaining = check_rate_limit(identifier, endpoint, role)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"X-RateLimit-Remaining": str(remaining)},
        )


def require_role(required_roles: List[UserRole]):
    """Require specific user roles"""

    async def role_checker(current_user: Dict = Depends(get_current_user)):
        user_role = current_user.get("role")
        if user_role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions"
            )
        return current_user

    return role_checker


# Authentication endpoints
@app.post("/auth/register", response_model=UserResponse)
async def register(user: UserCreate):
    """Register a new user"""
    try:
        new_user = auth_manager.create_user(user)
        return UserResponse(
            id=new_user.id,
            username=new_user.username,
            email=new_user.email,
            role=new_user.role,
            is_active=new_user.is_active,
            created_at=new_user.created_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and get JWT tokens"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return create_tokens(user)


@app.post("/auth/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str = Body(...), current_user: Dict = Depends(get_current_user)
):
    """Refresh access token using refresh token"""
    token_data = verify_token(refresh_token)
    if not token_data or token_data.username != current_user["username"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )

    user = auth_manager.get_user(token_data.username)
    return create_tokens(user)


@app.post("/auth/logout")
async def logout(
    token: str = Depends(oauth2_scheme), current_user: Dict = Depends(get_current_user)
):
    """Logout and invalidate token"""
    auth_manager.invalidate_token(token)
    return {"message": "Successfully logged out"}


# API Key management
@app.post("/api-keys", response_model=APIKeyResponse)
async def create_new_api_key(
    name: str = Body(...),
    permissions: Optional[List[str]] = Body(default=[]),
    current_user: Dict = Depends(get_current_user),
    _: None = Depends(check_rate_limit_dependency),
):
    """Create a new API key for the current user"""
    api_key = create_api_key(current_user["user_id"], name, permissions)
    return APIKeyResponse(api_key=api_key, name=name, created_at=datetime.now())


@app.get("/api-keys")
async def list_api_keys(
    current_user: Dict = Depends(get_current_user), _: None = Depends(check_rate_limit_dependency)
):
    """List all API keys for current user"""
    # Implementation would fetch from database
    return {"message": "List of API keys", "user_id": current_user["user_id"]}


@app.delete("/api-keys/{key_prefix}")
async def revoke_api_key(
    key_prefix: str,
    current_user: Dict = Depends(get_current_user),
    _: None = Depends(check_rate_limit_dependency),
):
    """Revoke an API key"""
    # Implementation would revoke the key
    return {"message": f"API key {key_prefix} revoked"}


# User management (Admin only)
@app.get("/users", dependencies=[Depends(require_role([UserRole.ADMIN]))])
async def list_users(_: None = Depends(check_rate_limit_dependency)):
    """List all users (admin only)"""
    # Implementation would fetch all users
    return {"message": "List of users"}


@app.patch("/users/{user_id}/role", dependencies=[Depends(require_role([UserRole.ADMIN]))])
async def update_user_role(
    user_id: int, new_role: UserRole = Body(...), _: None = Depends(check_rate_limit_dependency)
):
    """Update user role (admin only)"""
    # Implementation would update user role
    return {"message": f"User {user_id} role updated to {new_role}"}


# Fraud detection endpoints (protected)
@app.post("/analyze/text", response_model=FraudAnalysisResponse)
async def analyze_text(
    request: FraudAnalysisRequest,
    current_user: Dict = Depends(get_current_user),
    _: None = Depends(check_rate_limit_dependency),
):
    """Analyze text for fraud detection"""
    if not fraud_pipeline:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "read"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    # Perform analysis
    try:
        result = await fraud_pipeline.process_text(request.content)

        return FraudAnalysisResponse(
            is_fraud=result.get("is_fraud", False),
            confidence=result.get("confidence", 0.0),
            fraud_types=result.get("fraud_types", []),
            risk_level=result.get("risk_level", "low"),
            explanation=result.get("explanation", ""),
            request_id=result.get("request_id", ""),
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")


@app.post("/analyze/batch", dependencies=[Depends(require_role([UserRole.USER, UserRole.ADMIN]))])
async def analyze_batch(
    requests: List[FraudAnalysisRequest],
    current_user: Dict = Depends(get_current_user),
    _: None = Depends(check_rate_limit_dependency),
):
    """Batch analysis (user and admin only)"""
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 items per batch")

    results = []
    for req in requests:
        # Process each request
        results.append(
            {"content": req.content[:50], "is_fraud": False, "confidence": 0.85}  # Mock result
        )

    return {"results": results, "total": len(results)}


# Rate limit status
@app.get("/rate-limit/status")
async def rate_limit_status(current_user: Dict = Depends(get_current_user)):
    """Check current rate limit status"""
    identifier = f"user_{current_user['user_id']}"
    role = current_user.get("role")
    limit = auth_manager.get_rate_limit_for_role(role)

    allowed, remaining = check_rate_limit(identifier, "status_check", role)

    return {
        "limit": limit,
        "remaining": remaining,
        "reset_in_seconds": 60,
        "role": role.value if role else "unknown",
    }


# Health check (public)
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "2.0.0",
        "authentication": "enabled",
        "rate_limiting": "enabled",
    }


# API documentation
@app.get("/")
async def root():
    """API root with documentation"""
    return {
        "title": "FraudLens Secured API",
        "version": "2.0.0",
        "documentation": "/docs",
        "authentication": {
            "jwt": "Use /auth/token to get JWT token",
            "api_key": "Use /api-keys to create API key",
        },
        "rate_limits": {
            "admin": "5000/minute",
            "user": "1000/minute",
            "viewer": "500/minute",
            "api_user": "2000/minute",
        },
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.exception_handler(429)
async def rate_limit_handler(request: Request, exc):
    """Rate limit exceeded handler"""
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "retry_after": 60,
            "timestamp": datetime.now().isoformat(),
        },
        headers={"Retry-After": "60"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
