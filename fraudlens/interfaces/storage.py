"""
Storage protocol for persisting fraud detection data.

Author: Yobie Benjamin
Date: 2026-02-28
"""

from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel


class StorageRecord(BaseModel):
    """Record stored in storage backend."""
    
    id: str
    data: dict[str, Any]
    timestamp: datetime
    metadata: dict[str, Any] = {}


class StorageProtocol(Protocol):
    """
    Protocol for storage backends.
    
    Implementations can use databases, file systems, or cloud storage.
    """
    
    async def store(
        self,
        key: str,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> str:
        """
        Store data with optional TTL.
        
        Args:
            key: Unique identifier for the data
            data: Data to store
            metadata: Optional metadata
            ttl: Time-to-live in seconds (None = no expiration)
        
        Returns:
            Storage record ID
        """
        ...
    
    async def retrieve(
        self,
        key: str,
    ) -> StorageRecord | None:
        """
        Retrieve data by key.
        
        Args:
            key: Key to retrieve
        
        Returns:
            Storage record or None if not found
        """
        ...
    
    async def delete(
        self,
        key: str,
    ) -> bool:
        """
        Delete data by key.
        
        Args:
            key: Key to delete
        
        Returns:
            True if deleted, False if not found
        """
        ...
    
    async def query(
        self,
        filters: dict[str, Any],
        limit: int = 100,
        offset: int = 0,
    ) -> list[StorageRecord]:
        """
        Query storage with filters.
        
        Args:
            filters: Query filters
            limit: Maximum number of results
            offset: Offset for pagination
        
        Returns:
            List of matching records
        """
        ...
    
    async def cleanup_expired(self) -> int:
        """
        Remove expired records based on TTL.
        
        Returns:
            Number of records cleaned up
        """
        ...
    
    async def get_stats(self) -> dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage metrics:
            - total_records: Total number of records
            - size_bytes: Total size in bytes
            - oldest_record: Timestamp of oldest record
            - newest_record: Timestamp of newest record
        """
        ...
