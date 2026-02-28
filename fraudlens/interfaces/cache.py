"""
Cache protocol for high-performance data caching.

Author: Yobie Benjamin
Date: 2026-02-28
"""

from typing import Any, Protocol


class CacheProtocol(Protocol):
    """
    Protocol for cache implementations.
    
    Caches provide fast access to frequently used data with automatic
    expiration and eviction policies.
    """
    
    async def get(
        self,
        key: str,
    ) -> Any | None:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found/expired
        """
        ...
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """
        Set value in cache with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = default TTL)
        """
        ...
    
    async def delete(
        self,
        key: str,
    ) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if deleted, False if not found
        """
        ...
    
    async def exists(
        self,
        key: str,
    ) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if key exists
        """
        ...
    
    async def clear(self) -> None:
        """
        Clear all cache entries.
        """
        ...
    
    async def get_many(
        self,
        keys: list[str],
    ) -> dict[str, Any]:
        """
        Get multiple values from cache.
        
        Args:
            keys: List of cache keys
        
        Returns:
            Dictionary mapping keys to values (missing keys excluded)
        """
        ...
    
    async def set_many(
        self,
        items: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        """
        Set multiple values in cache.
        
        Args:
            items: Dictionary of key-value pairs
            ttl: Time-to-live in seconds
        """
        ...
    
    async def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache metrics:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - hit_rate: Cache hit rate (0.0 to 1.0)
            - size: Current cache size
            - max_size: Maximum cache size
        """
        ...
    
    async def invalidate_pattern(
        self,
        pattern: str,
    ) -> int:
        """
        Invalidate cache entries matching a pattern.
        
        Args:
            pattern: Pattern to match (e.g., "user:*")
        
        Returns:
            Number of entries invalidated
        """
        ...
