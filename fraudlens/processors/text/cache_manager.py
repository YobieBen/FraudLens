"""
Cache manager for text processing with LRU and vector similarity support.

Author: Yobie Benjamin
Date: 2025-08-26 18:34:00 PDT
"""

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    
    def update_access(self):
        """Update access metadata."""
        self.access_count += 1
        self.timestamp = time.time()


class CacheManager:
    """
    Advanced cache manager with LRU eviction and similarity search.
    
    Features:
    - LRU cache for repeated text analysis
    - TTL support for cache entries
    - Vector similarity search (ChromaDB integration ready)
    - Memory-aware eviction
    - Persistence support
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
        enable_similarity: bool = True,
        persist_path: Optional[Path] = None,
    ):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum number of cache entries
            ttl_seconds: Time-to-live for cache entries
            enable_similarity: Enable similarity-based caching
            persist_path: Path for cache persistence
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_similarity = enable_similarity
        self.persist_path = persist_path
        
        # LRU cache using OrderedDict
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._memory_usage = 0
        self._max_memory = 500 * 1024 * 1024  # 500MB max
        
        # Vector store for similarity search (placeholder for ChromaDB)
        self._vector_store = None
        self._embeddings_cache: Dict[str, List[float]] = {}
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
    async def initialize(self) -> None:
        """Initialize cache manager."""
        logger.info("Initializing cache manager...")
        
        # Load persistent cache if available
        if self.persist_path and self.persist_path.exists():
            await self._load_cache()
        
        # Initialize vector store if enabled
        if self.enable_similarity:
            await self._initialize_vector_store()
        
        logger.info(f"Cache manager initialized with max_size={self.max_size}")
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        # Check if key exists
        if key not in self._cache:
            self._misses += 1
            
            # Try similarity search if enabled
            if self.enable_similarity:
                similar_key = await self._find_similar(key)
                if similar_key and similar_key in self._cache:
                    entry = self._cache[similar_key]
                    # Check TTL
                    if time.time() - entry.timestamp < self.ttl_seconds:
                        entry.update_access()
                        self._move_to_end(similar_key)
                        self._hits += 1
                        return entry.value
            
            return None
        
        entry = self._cache[key]
        
        # Check TTL
        if time.time() - entry.timestamp > self.ttl_seconds:
            # Expired entry
            del self._cache[key]
            self._memory_usage -= entry.size_bytes
            self._misses += 1
            return None
        
        # Update access and move to end (most recently used)
        entry.update_access()
        self._move_to_end(key)
        self._hits += 1
        
        return entry.value
    
    async def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Calculate size
        size_bytes = len(json.dumps(value)) if value else 0
        
        # If replacing existing entry, account for its memory
        if key in self._cache:
            old_entry = self._cache[key]
            self._memory_usage -= old_entry.size_bytes
        
        # Check memory limit
        while self._memory_usage + size_bytes > self._max_memory:
            await self._evict_lru()
        
        # Check size limit (only if adding new key)
        if key not in self._cache:
            while len(self._cache) >= self.max_size:
                await self._evict_lru()
        
        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            size_bytes=size_bytes,
        )
        
        # Add to cache
        self._cache[key] = entry
        self._memory_usage += size_bytes
        
        # Add to vector store if enabled
        if self.enable_similarity:
            await self._add_to_vector_store(key, value)
    
    async def delete(self, key: str) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        if key in self._cache:
            entry = self._cache[key]
            self._memory_usage -= entry.size_bytes
            del self._cache[key]
            
            # Remove from vector store
            if self.enable_similarity and key in self._embeddings_cache:
                del self._embeddings_cache[key]
            
            return True
        
        return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._embeddings_cache.clear()
        self._memory_usage = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0
        
        return {
            "size": len(self._cache),
            "memory_usage_mb": self._memory_usage / (1024 * 1024),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": hit_rate,
        }
    
    def _move_to_end(self, key: str) -> None:
        """Move key to end of OrderedDict (most recently used)."""
        self._cache.move_to_end(key)
    
    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        # Get first item (least recently used)
        key, entry = next(iter(self._cache.items()))
        
        # Remove from cache
        del self._cache[key]
        self._memory_usage -= entry.size_bytes
        self._evictions += 1
        
        # Remove from vector store
        if self.enable_similarity and key in self._embeddings_cache:
            del self._embeddings_cache[key]
    
    async def _initialize_vector_store(self) -> None:
        """Initialize vector store for similarity search."""
        try:
            # Try to use ChromaDB if available
            import chromadb
            from chromadb.config import Settings
            
            # Initialize ChromaDB client
            self._chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(self.persist_path) if self.persist_path else None,
            ))
            
            # Create or get collection
            self._collection = self._chroma_client.get_or_create_collection(
                name="fraudlens_cache",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("ChromaDB vector store initialized")
            
        except ImportError:
            logger.warning("ChromaDB not available. Using simple embedding cache.")
            self._use_simple_embeddings = True
    
    async def _add_to_vector_store(self, key: str, value: Any) -> None:
        """Add entry to vector store."""
        if not self.enable_similarity:
            return
        
        # Generate embedding (simplified - in production, use proper embedding model)
        embedding = self._generate_simple_embedding(str(value))
        self._embeddings_cache[key] = embedding
        
        # Add to ChromaDB if available
        if hasattr(self, '_collection'):
            try:
                self._collection.add(
                    embeddings=[embedding],
                    documents=[json.dumps(value)],
                    ids=[key],
                )
            except:
                pass
    
    async def _find_similar(self, key: str) -> Optional[str]:
        """Find similar cached entry."""
        if not self.enable_similarity or not self._embeddings_cache:
            return None
        
        # Don't use similarity for hash keys (they should be exact matches only)
        # Hash keys are 64 character hex strings
        if len(key) == 64 and all(c in '0123456789abcdef' for c in key):
            return None
        
        # Generate embedding for query
        query_embedding = self._generate_simple_embedding(key)
        
        # Find most similar in cache
        best_similarity = 0.0
        best_key = None
        
        for cached_key, cached_embedding in self._embeddings_cache.items():
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            if similarity > 0.95 and similarity > best_similarity:  # Higher threshold
                best_similarity = similarity
                best_key = cached_key
        
        return best_key
    
    def _generate_simple_embedding(self, text: str) -> List[float]:
        """Generate simple embedding for text (placeholder)."""
        # This is a simplified embedding for demo
        # In production, use proper sentence embeddings
        import hashlib
        
        # Generate deterministic pseudo-embedding
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to float vector
        embedding = []
        for i in range(0, min(len(hash_bytes), 128), 4):
            value = int.from_bytes(hash_bytes[i:i+4], 'big') / (2**32)
            embedding.append(value)
        
        # Pad to fixed size
        while len(embedding) < 32:
            embedding.append(0.0)
        
        return embedding[:32]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def _load_cache(self) -> None:
        """Load cache from persistent storage."""
        cache_file = self.persist_path / "cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    
                for key, entry_data in data.items():
                    entry = CacheEntry(
                        key=key,
                        value=entry_data["value"],
                        timestamp=entry_data["timestamp"],
                        access_count=entry_data.get("access_count", 0),
                        size_bytes=entry_data.get("size_bytes", 0),
                    )
                    
                    # Only load non-expired entries
                    if time.time() - entry.timestamp < self.ttl_seconds:
                        self._cache[key] = entry
                        self._memory_usage += entry.size_bytes
                        
                logger.info(f"Loaded {len(self._cache)} cache entries from disk")
                
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
    
    async def persist(self) -> None:
        """Persist cache to disk."""
        if not self.persist_path:
            return
        
        self.persist_path.mkdir(parents=True, exist_ok=True)
        cache_file = self.persist_path / "cache.json"
        
        # Prepare data for serialization
        data = {}
        for key, entry in self._cache.items():
            data[key] = {
                "value": entry.value,
                "timestamp": entry.timestamp,
                "access_count": entry.access_count,
                "size_bytes": entry.size_bytes,
            }
        
        # Save to file
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Persisted {len(data)} cache entries to disk")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up cache manager...")
        
        # Persist cache if configured
        if self.persist_path:
            await self.persist()
        
        # Clear cache
        await self.clear()
        
        logger.info("Cache manager cleanup complete")
    
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        return self._memory_usage