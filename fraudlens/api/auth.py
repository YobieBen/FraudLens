"""
FraudLens API Authentication and Security Module
Implements JWT authentication, API key management, and role-based access control
"""

import os
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from enum import Enum
import jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field, EmailStr
import sqlite3
import json
from pathlib import Path
from functools import wraps
import redis
from loguru import logger

# Security configuration
SECRET_KEY = os.getenv("FRAUDLENS_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
API_KEY_LENGTH = 32

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Database path
DB_PATH = Path(__file__).parent.parent.parent / "auth.db"


class UserRole(str, Enum):
    """User role definitions"""

    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    API_USER = "api_user"


class RateLimitConfig:
    """Rate limiting configuration"""

    DEFAULT_LIMIT = 1000  # requests per minute
    ADMIN_LIMIT = 5000  # requests per minute
    USER_LIMIT = 1000  # requests per minute
    VIEWER_LIMIT = 500  # requests per minute
    API_USER_LIMIT = 2000  # requests per minute

    ROLE_LIMITS = {
        UserRole.ADMIN: ADMIN_LIMIT,
        UserRole.USER: USER_LIMIT,
        UserRole.VIEWER: VIEWER_LIMIT,
        UserRole.API_USER: API_USER_LIMIT,
    }


# Pydantic models
class UserCreate(BaseModel):
    """User creation model"""

    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: UserRole = UserRole.USER


class UserInDB(BaseModel):
    """User database model"""

    id: int
    username: str
    email: str
    hashed_password: str
    role: UserRole
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None
    api_keys: List[str] = []


class Token(BaseModel):
    """Token response model"""

    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token payload data"""

    username: Optional[str] = None
    user_id: Optional[int] = None
    role: Optional[UserRole] = None
    scopes: List[str] = []


class APIKey(BaseModel):
    """API Key model"""

    key: str
    name: str
    user_id: int
    created_at: datetime
    last_used: Optional[datetime] = None
    is_active: bool = True
    permissions: List[str] = []


class AuthManager:
    """Main authentication and security manager"""

    def __init__(self):
        """Initialize auth manager"""
        self.init_database()
        self.init_redis()

    def init_database(self):
        """Initialize SQLite database for user and API key storage"""
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        # Users table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                role TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                metadata TEXT
            )
        """
        )

        # API Keys table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT UNIQUE NOT NULL,
                key_prefix TEXT NOT NULL,
                name TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                permissions TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """
        )

        # Sessions table for tracking
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash TEXT UNIQUE NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """
        )

        # Rate limit tracking
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS rate_limits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                identifier TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                count INTEGER DEFAULT 0,
                window_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(identifier, endpoint, window_start)
            )
        """
        )

        conn.commit()
        conn.close()

        logger.info("Authentication database initialized")

    def init_redis(self):
        """Initialize Redis for rate limiting and session caching"""
        try:
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=0,
                decode_responses=True,
            )
            self.redis_client.ping()
            self.redis_available = True
            logger.info("Redis connected for rate limiting")
        except:
            self.redis_available = False
            logger.warning("Redis not available, using database for rate limiting")

    # Password hashing
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)

    # User management
    def create_user(self, user: UserCreate) -> UserInDB:
        """Create a new user"""
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        hashed_password = self.get_password_hash(user.password)

        try:
            cursor.execute(
                """
                INSERT INTO users (username, email, hashed_password, role)
                VALUES (?, ?, ?, ?)
            """,
                (user.username, user.email, hashed_password, user.role.value),
            )

            user_id = cursor.lastrowid
            conn.commit()

            logger.info(f"User created: {user.username} with role {user.role}")

            return UserInDB(
                id=user_id,
                username=user.username,
                email=user.email,
                hashed_password=hashed_password,
                role=user.role,
                created_at=datetime.now(),
                api_keys=[],
            )

        except sqlite3.IntegrityError as e:
            logger.error(f"User creation failed: {e}")
            raise ValueError("Username or email already exists")
        finally:
            conn.close()

    def get_user(self, username: str) -> Optional[UserInDB]:
        """Get user by username"""
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, username, email, hashed_password, role, is_active, 
                   created_at, last_login
            FROM users WHERE username = ?
        """,
            (username,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return UserInDB(
                id=row[0],
                username=row[1],
                email=row[2],
                hashed_password=row[3],
                role=UserRole(row[4]),
                is_active=bool(row[5]),
                created_at=datetime.fromisoformat(row[6]),
                last_login=datetime.fromisoformat(row[7]) if row[7] else None,
                api_keys=[],
            )
        return None

    def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        """Authenticate user with username and password"""
        user = self.get_user(username)
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        if not user.is_active:
            return None

        # Update last login
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
        """,
            (user.id,),
        )
        conn.commit()
        conn.close()

        return user

    # JWT Token management
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

        return encoded_jwt

    def create_refresh_token(self, data: dict) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        # Add unique identifier to prevent hash collisions
        to_encode.update(
            {
                "exp": expire,
                "type": "refresh",
                "jti": secrets.token_hex(16),  # JWT ID for uniqueness
            }
        )

        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

        # Store in database for tracking
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        token_hash = hashlib.sha256(encoded_jwt.encode()).hexdigest()
        try:
            cursor.execute(
                """
                INSERT INTO sessions (user_id, token_hash, expires_at)
                VALUES (?, ?, ?)
            """,
                (data.get("user_id"), token_hash, expire),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            # Handle duplicate token hash (very rare)
            logger.warning("Token hash collision detected, regenerating")
            conn.close()
            return self.create_refresh_token(data)

        conn.close()

        return encoded_jwt

    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            user_id: int = payload.get("user_id")
            role: str = payload.get("role")

            if username is None:
                return None

            return TokenData(
                username=username,
                user_id=user_id,
                role=UserRole(role) if role else None,
                scopes=payload.get("scopes", []),
            )
        except jwt.PyJWTError:
            return None

    # API Key management
    def create_api_key(self, user_id: int, name: str, permissions: List[str] = None) -> str:
        """Create new API key for user"""
        # Generate secure random key
        api_key = f"flk_{secrets.token_urlsafe(API_KEY_LENGTH)}"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_prefix = api_key[:12]  # Store prefix for identification

        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO api_keys (key_hash, key_prefix, name, user_id, permissions)
            VALUES (?, ?, ?, ?, ?)
        """,
            (key_hash, key_prefix, name, user_id, json.dumps(permissions or [])),
        )

        conn.commit()
        conn.close()

        logger.info(f"API key created for user {user_id}: {name}")

        return api_key

    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key and return associated user info"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT k.user_id, k.permissions, k.is_active, u.username, u.role
            FROM api_keys k
            JOIN users u ON k.user_id = u.id
            WHERE k.key_hash = ? AND k.is_active = 1 AND u.is_active = 1
        """,
            (key_hash,),
        )

        row = cursor.fetchone()

        if row:
            # Update last used
            cursor.execute(
                """
                UPDATE api_keys SET last_used = CURRENT_TIMESTAMP
                WHERE key_hash = ?
            """,
                (key_hash,),
            )
            conn.commit()

            conn.close()

            return {
                "user_id": row[0],
                "permissions": json.loads(row[1]),
                "username": row[3],
                "role": row[4],
            }

        conn.close()
        return None

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE api_keys SET is_active = 0 WHERE key_hash = ?
        """,
            (key_hash,),
        )

        affected = cursor.rowcount
        conn.commit()
        conn.close()

        return affected > 0

    # Rate limiting
    def check_rate_limit(
        self, identifier: str, endpoint: str = "default", limit: Optional[int] = None
    ) -> tuple[bool, int]:
        """Check if rate limit is exceeded"""
        if limit is None:
            limit = RateLimitConfig.DEFAULT_LIMIT

        window_start = datetime.now().replace(second=0, microsecond=0)

        if self.redis_available:
            # Use Redis for rate limiting
            key = f"rate_limit:{identifier}:{endpoint}:{window_start.timestamp()}"

            try:
                current = self.redis_client.incr(key)
                if current == 1:
                    self.redis_client.expire(key, 60)  # 1 minute window

                return current <= limit, limit - current
            except:
                # Fallback to database
                pass

        # Database rate limiting
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO rate_limits (identifier, endpoint, count, window_start)
            VALUES (?, ?, 
                COALESCE((SELECT count + 1 FROM rate_limits 
                         WHERE identifier = ? AND endpoint = ? AND window_start = ?), 1),
                ?)
        """,
            (identifier, endpoint, identifier, endpoint, window_start, window_start),
        )

        cursor.execute(
            """
            SELECT count FROM rate_limits
            WHERE identifier = ? AND endpoint = ? AND window_start = ?
        """,
            (identifier, endpoint, window_start),
        )

        count = cursor.fetchone()[0]
        conn.commit()
        conn.close()

        return count <= limit, limit - count

    def get_rate_limit_for_role(self, role: UserRole) -> int:
        """Get rate limit based on user role"""
        return RateLimitConfig.ROLE_LIMITS.get(role, RateLimitConfig.DEFAULT_LIMIT)

    # Permission checking
    def check_permission(self, user_role: UserRole, required_permission: str) -> bool:
        """Check if user role has required permission"""
        permissions = {
            UserRole.ADMIN: ["read", "write", "delete", "admin"],
            UserRole.USER: ["read", "write"],
            UserRole.VIEWER: ["read"],
            UserRole.API_USER: ["read", "write", "api"],
        }

        return required_permission in permissions.get(user_role, [])

    # Session management
    def invalidate_token(self, token: str) -> bool:
        """Invalidate a token (for logout)"""
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM sessions WHERE token_hash = ?
        """,
            (token_hash,),
        )

        affected = cursor.rowcount
        conn.commit()
        conn.close()

        return affected > 0

    def clean_expired_sessions(self):
        """Clean up expired sessions"""
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM sessions WHERE expires_at < CURRENT_TIMESTAMP
        """
        )

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        logger.info(f"Cleaned {deleted} expired sessions")

        return deleted


# Singleton instance
auth_manager = AuthManager()


# Helper functions for FastAPI
def create_user(user_data: UserCreate) -> UserInDB:
    """Create a new user"""
    return auth_manager.create_user(user_data)


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate user"""
    return auth_manager.authenticate_user(username, password)


def create_tokens(user: UserInDB) -> Token:
    """Create access and refresh tokens for user"""
    access_token_data = {
        "sub": user.username,
        "user_id": user.id,
        "role": user.role.value,
        "scopes": [],
    }

    access_token = auth_manager.create_access_token(access_token_data)
    refresh_token = auth_manager.create_refresh_token(access_token_data)

    return Token(access_token=access_token, refresh_token=refresh_token, token_type="bearer")


def verify_token(token: str) -> Optional[TokenData]:
    """Verify JWT token"""
    return auth_manager.verify_token(token)


def create_api_key(user_id: int, name: str, permissions: List[str] = None) -> str:
    """Create API key"""
    return auth_manager.create_api_key(user_id, name, permissions)


def verify_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """Verify API key"""
    return auth_manager.verify_api_key(api_key)


def check_rate_limit(
    identifier: str, endpoint: str = "default", role: Optional[UserRole] = None
) -> tuple[bool, int]:
    """Check rate limit"""
    limit = auth_manager.get_rate_limit_for_role(role) if role else None
    return auth_manager.check_rate_limit(identifier, endpoint, limit)
