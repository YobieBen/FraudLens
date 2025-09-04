"""
FraudLens Cache Manager
Implements Redis caching for fraud detection results with fallback to in-memory cache
"""

import asyncio
import hashlib
import json
import os
import pickle
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import wraps
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import redis
from loguru import logger


class CacheConfig:
    """Cache configuration"""

    # Redis settings
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 1))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

    # Cache TTL settings (in seconds)
    DEFAULT_TTL = 3600  # 1 hour
    TEXT_CACHE_TTL = 7200  # 2 hours
    IMAGE_CACHE_TTL = 3600  # 1 hour
    VIDEO_CACHE_TTL = 1800  # 30 minutes
    DOCUMENT_CACHE_TTL = 3600  # 1 hour

    # In-memory cache settings
    MAX_MEMORY_ITEMS = 1000
    MEMORY_TTL = 600  # 10 minutes

    # Cache key prefixes
    TEXT_PREFIX = "fraud:text:"
    IMAGE_PREFIX = "fraud:image:"
    VIDEO_PREFIX = "fraud:video:"
    DOCUMENT_PREFIX = "fraud:doc:"
    BATCH_PREFIX = "fraud:batch:"


class InMemoryCache:
    """Thread-safe in-memory LRU cache with TTL"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 600):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.lock = Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                if time.time() < expiry:
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    self.hits += 1
                    return value
                else:
                    # Expired
                    del self.cache[key]

            self.misses += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set item in cache"""
        if ttl is None:
            ttl = self.default_ttl

        with self.lock:
            # Remove oldest items if at capacity
            while len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)

            expiry = time.time() + ttl
            self.cache[key] = (value, expiry)

    def delete(self, key: str) -> bool:
        """Delete item from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def clear(self):
        """Clear all cache"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{hit_rate:.1f}%",
                "total_requests": total,
            }


class CacheManager:
    """Main cache manager with Redis and in-memory fallback"""

    def __init__(self):
        """Initialize cache manager"""
        self.redis_client = None
        self.redis_available = False
        self.memory_cache = InMemoryCache(
            max_size=CacheConfig.MAX_MEMORY_ITEMS, default_ttl=CacheConfig.MEMORY_TTL
        )

        # Try to connect to Redis
        self._init_redis()

    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=CacheConfig.REDIS_HOST,
                port=CacheConfig.REDIS_PORT,
                db=CacheConfig.REDIS_DB,
                password=CacheConfig.REDIS_PASSWORD,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=False,  # We'll handle encoding/decoding
            )

            # Test connection
            self.redis_client.ping()
            self.redis_available = True
            logger.info(
                f"Redis cache connected at {CacheConfig.REDIS_HOST}:{CacheConfig.REDIS_PORT}"
            )

        except (redis.ConnectionError, redis.TimeoutError) as e:
            self.redis_available = False
            logger.warning(f"Redis not available, using in-memory cache: {e}")

    def _generate_key(self, content: Any, prefix: str = "") -> str:
        """Generate cache key from content"""
        if isinstance(content, str):
            content_hash = hashlib.md5(content.encode()).hexdigest()
        elif isinstance(content, bytes):
            content_hash = hashlib.md5(content).hexdigest()
        else:
            content_hash = hashlib.md5(str(content).encode()).hexdigest()

        return f"{prefix}{content_hash}"

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        # Try Redis first
        if self.redis_available:
            try:
                value = self.redis_client.get(key)
                if value:
                    return pickle.loads(value)
            except Exception as e:
                logger.debug(f"Redis get error: {e}")

        # Fallback to memory cache
        return self.memory_cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        if ttl is None:
            ttl = CacheConfig.DEFAULT_TTL

        # Try Redis first
        if self.redis_available:
            try:
                serialized = pickle.dumps(value)
                self.redis_client.setex(key, ttl, serialized)
                return
            except Exception as e:
                logger.debug(f"Redis set error: {e}")

        # Fallback to memory cache
        self.memory_cache.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        deleted = False

        # Try Redis
        if self.redis_available:
            try:
                deleted = bool(self.redis_client.delete(key))
            except Exception as e:
                logger.debug(f"Redis delete error: {e}")

        # Also delete from memory cache
        deleted = self.memory_cache.delete(key) or deleted

        return deleted

    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        # Check Redis
        if self.redis_available:
            try:
                if self.redis_client.exists(key):
                    return True
            except Exception as e:
                logger.debug(f"Redis exists error: {e}")

        # Check memory cache
        return self.memory_cache.get(key) is not None

    def clear_pattern(self, pattern: str):
        """Clear all keys matching pattern"""
        if self.redis_available:
            try:
                # Use SCAN to avoid blocking
                cursor = 0
                while True:
                    cursor, keys = self.redis_client.scan(cursor, match=pattern, count=100)
                    if keys:
                        self.redis_client.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.debug(f"Redis clear pattern error: {e}")

        # Clear from memory cache (simplified - clears all)
        self.memory_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {"redis_available": self.redis_available, "memory_cache": self.memory_cache.stats()}

        if self.redis_available:
            try:
                info = self.redis_client.info("stats")
                stats["redis"] = {
                    "total_connections": info.get("total_connections_received", 0),
                    "total_commands": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                    "used_memory": self.redis_client.info("memory").get("used_memory_human", "N/A"),
                }
            except Exception as e:
                logger.debug(f"Redis stats error: {e}")

        return stats

    # Specialized cache methods for different content types

    def cache_text_result(self, text: str, result: Dict[str, Any]) -> str:
        """Cache text fraud detection result"""
        key = self._generate_key(text, CacheConfig.TEXT_PREFIX)
        self.set(key, result, CacheConfig.TEXT_CACHE_TTL)
        return key

    def get_text_result(self, text: str) -> Optional[Dict[str, Any]]:
        """Get cached text fraud detection result"""
        key = self._generate_key(text, CacheConfig.TEXT_PREFIX)
        return self.get(key)

    def cache_image_result(self, image_data: bytes, result: Dict[str, Any]) -> str:
        """Cache image fraud detection result"""
        key = self._generate_key(image_data, CacheConfig.IMAGE_PREFIX)
        self.set(key, result, CacheConfig.IMAGE_CACHE_TTL)
        return key

    def get_image_result(self, image_data: bytes) -> Optional[Dict[str, Any]]:
        """Get cached image fraud detection result"""
        key = self._generate_key(image_data, CacheConfig.IMAGE_PREFIX)
        return self.get(key)

    def cache_video_result(self, video_hash: str, result: Dict[str, Any]) -> str:
        """Cache video fraud detection result"""
        key = f"{CacheConfig.VIDEO_PREFIX}{video_hash}"
        self.set(key, result, CacheConfig.VIDEO_CACHE_TTL)
        return key

    def get_video_result(self, video_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached video fraud detection result"""
        key = f"{CacheConfig.VIDEO_PREFIX}{video_hash}"
        return self.get(key)

    def cache_document_result(self, doc_data: bytes, result: Dict[str, Any]) -> str:
        """Cache document validation result"""
        key = self._generate_key(doc_data, CacheConfig.DOCUMENT_PREFIX)
        self.set(key, result, CacheConfig.DOCUMENT_CACHE_TTL)
        return key

    def get_document_result(self, doc_data: bytes) -> Optional[Dict[str, Any]]:
        """Get cached document validation result"""
        key = self._generate_key(doc_data, CacheConfig.DOCUMENT_PREFIX)
        return self.get(key)

    def cache_batch_result(self, batch_id: str, results: List[Dict[str, Any]]):
        """Cache batch processing results"""
        key = f"{CacheConfig.BATCH_PREFIX}{batch_id}"
        self.set(key, results, CacheConfig.DEFAULT_TTL * 2)  # Longer TTL for batch
        return key

    def get_batch_result(self, batch_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached batch processing results"""
        key = f"{CacheConfig.BATCH_PREFIX}{batch_id}"
        return self.get(key)


# Singleton instance
cache_manager = CacheManager()


# Decorator for caching function results
def cached(ttl: int = CacheConfig.DEFAULT_TTL, prefix: str = "func:"):
    """Decorator to cache function results"""

    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = f"{prefix}{func.__name__}:{str(args)}:{str(kwargs)}"

            # Try to get from cache
            result = cache_manager.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            logger.debug(f"Cached result for {func.__name__}")

            return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = f"{prefix}{func.__name__}:{str(args)}:{str(kwargs)}"

            # Try to get from cache
            result = cache_manager.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            logger.debug(f"Cached result for {func.__name__}")

            return result

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Helper functions
def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    return cache_manager.get_stats()


def clear_cache(pattern: Optional[str] = None):
    """Clear cache"""
    if pattern:
        cache_manager.clear_pattern(pattern)
    else:
        cache_manager.memory_cache.clear()
        if cache_manager.redis_available:
            cache_manager.redis_client.flushdb()
