"""Advanced caching system for ML operations and API responses.

This module provides intelligent caching for:
- LLM API responses (OpenAI, etc.)
- Vector embeddings
- Model predictions
- Data processing results
- Cost optimization through deduplication
"""

import hashlib
import json
import logging
import pickle
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "mlops-template"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    cost_saved: float
    ttl: float | None = None
    tags: list[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class CacheManager:
    """Multi-level cache manager with cost tracking."""

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR):
        """Initialize cache manager.

        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # Add thread safety

        # Initialize SQLite database for metadata
        self.db_path = self.cache_dir / "cache_metadata.db"
        self._init_database()

        # In-memory cache for frequently accessed items
        self.memory_cache: dict[str, CacheEntry] = {}
        self.max_memory_items = 1000

        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "total_cost_saved": 0.0,
            "storage_size": 0,
        }

    def _init_database(self):
        """Initialize SQLite database for cache metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    created_at REAL,
                    accessed_at REAL,
                    access_count INTEGER,
                    cost_saved REAL,
                    ttl REAL,
                    tags TEXT,
                    file_path TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache_entries(accessed_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tags ON cache_entries(tags)
            """)

    def _generate_key(self, data: Any, prefix: str = "") -> str:
        """Generate cache key from data.

        Args:
            data: Data to generate key for
            prefix: Optional prefix for the key

        Returns:
            Cache key string
        """
        if isinstance(data, dict):
            # Sort dict for consistent hashing
            data_str = json.dumps(data, sort_keys=True)
        elif isinstance(data, list | tuple):
            data_str = json.dumps(data)
        else:
            data_str = str(data)

        hash_obj = hashlib.sha256(data_str.encode())
        key = hash_obj.hexdigest()[:16]  # Use first 16 chars

        return f"{prefix}_{key}" if prefix else key

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache.

        Args:
            key: Cache key
            default: Default value to return if key not found

        Returns:
            Cached value or default if not found/expired
        """
        try:
            with self._lock:  # Thread safety
                cache_file = self._get_cache_file(key)
                if not cache_file.exists():
                    return default

                with open(cache_file) as f:
                    cache_data = json.load(f)

                # Check if expired
                if cache_data.get("ttl") is not None:
                    age = time.time() - cache_data["timestamp"]
                    if age > cache_data["ttl"]:
                        cache_file.unlink()
                        return default

                return cache_data["value"]
        except Exception:
            return default

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        cost_saved: float = 0.0,
        tags: list[str] = None,
    ) -> bool:
        """Set a value in the cache."""
        try:
            with self._lock:  # Thread safety
                cache_file = self._get_cache_file(key)
                cache_data = {
                    "value": value,
                    "timestamp": time.time(),
                    "ttl": ttl,
                    "cost_saved": cost_saved,
                    "tags": tags or [],
                }

                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "w") as f:
                    json.dump(cache_data, f, default=str)

                # ALSO update SQLite database for cost tracking consistency
                current_time = time.time()
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO cache_entries
                        (key, created_at, accessed_at, access_count, cost_saved, ttl, tags, file_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            key,
                            current_time,
                            current_time,
                            1,  # Initial access count
                            cost_saved,
                            ttl,
                            json.dumps(tags or []),
                            str(cache_file),
                        ),
                    )

                return True
        except Exception as e:
            logger.error(f"Error setting cache value for key {key}: {e}")
            return False

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        if entry.ttl is None:
            return False
        return time.time() - entry.created_at > entry.ttl

    def _get_from_disk(self, key: str) -> CacheEntry | None:
        """Get cache entry from disk."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM cache_entries WHERE key = ?", (key,))
            row = cursor.fetchone()

            if not row:
                return None

            # Load value from file
            file_path = Path(row[7])  # file_path column
            if not file_path.exists():
                # Clean up orphaned metadata
                conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                return None

            try:
                with open(file_path, "rb") as f:
                    value = pickle.load(f)  # nosec B301 - Only loading trusted cache data

                return CacheEntry(
                    key=row[0],
                    value=value,
                    created_at=row[1],
                    accessed_at=row[2],
                    access_count=row[3],
                    cost_saved=row[4],
                    ttl=row[5] if row[5] else None,
                    tags=json.loads(row[6]) if row[6] else [],
                )
            except Exception as e:
                logger.error(f"Error loading cached value for key {key}: {e}")
                return None

    def _save_to_disk(self, entry: CacheEntry) -> None:
        """Save cache entry to disk."""
        # Save value to file
        file_path = self.cache_dir / f"{entry.key}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(entry.value, f)

        # Update metadata in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache_entries
                (key, created_at, accessed_at, access_count, cost_saved, ttl, tags, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entry.key,
                    entry.created_at,
                    entry.accessed_at,
                    entry.access_count,
                    entry.cost_saved,
                    entry.ttl,
                    json.dumps(entry.tags),
                    str(file_path),
                ),
            )

    def _update_metadata(self, entry: CacheEntry) -> None:
        """Update cache entry metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE cache_entries
                SET accessed_at = ?, access_count = ?
                WHERE key = ?
            """,
                (entry.accessed_at, entry.access_count, entry.key),
            )

    def _add_to_memory_cache(self, key: str, entry: CacheEntry) -> None:
        """Add entry to memory cache with LRU eviction."""
        if len(self.memory_cache) >= self.max_memory_items:
            # Remove least recently accessed item
            lru_key = min(
                self.memory_cache.keys(), key=lambda k: self.memory_cache[k].accessed_at
            )
            del self.memory_cache[lru_key]

        self.memory_cache[key] = entry

    def cache_llm_response(
        self,
        prompt: str,
        model: str,
        response: str,
        cost: float = 0.0,
        ttl: float = 86400,
    ) -> str:
        """Cache LLM response with automatic key generation.

        Args:
            prompt: Input prompt
            model: Model name
            response: LLM response
            cost: Cost of the API call
            ttl: Cache TTL in seconds (default 24 hours)

        Returns:
            Cache key for the response
        """
        cache_data = {"prompt": prompt, "model": model}
        key = self._generate_key(cache_data, "llm")

        self.set(key=key, value=response, ttl=ttl, cost_saved=cost, tags=["llm", model])

        return key

    def get_llm_response(self, prompt: str, model: str) -> str | None:
        """Get cached LLM response.

        Args:
            prompt: Input prompt
            model: Model name

        Returns:
            Cached response or None
        """
        cache_data = {"prompt": prompt, "model": model}
        key = self._generate_key(cache_data, "llm")
        return self.get(key)

    def cache_embedding(
        self, text: str, model: str, embedding: list[float], cost: float = 0.0
    ) -> str:
        """Cache text embedding.

        Args:
            text: Input text
            model: Embedding model name
            embedding: Vector embedding
            cost: Cost of generating embedding

        Returns:
            Cache key
        """
        cache_data = {"text": text, "model": model}
        key = self._generate_key(cache_data, "emb")

        self.set(key=key, value=embedding, cost_saved=cost, tags=["embedding", model])

        return key

    def get_embedding(self, text: str, model: str) -> list[float] | None:
        """Get cached embedding.

        Args:
            text: Input text
            model: Embedding model name

        Returns:
            Cached embedding or None
        """
        cache_data = {"text": text, "model": model}
        key = self._generate_key(cache_data, "emb")
        return self.get(key)

    def clean_expired(self) -> int:
        """Clean expired cache entries.

        Returns:
            Number of entries cleaned
        """
        cleaned = 0

        with sqlite3.connect(self.db_path) as conn:
            # Get all entries with TTL
            cursor = conn.execute(
                "SELECT key, created_at, ttl, file_path FROM cache_entries WHERE ttl IS NOT NULL"
            )

            now = time.time()
            for row in cursor:
                key, created_at, ttl, file_path = row
                if now - created_at > ttl:
                    # Remove file
                    file_path = Path(file_path)
                    if file_path.exists():
                        file_path.unlink()

                    # Remove from database
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))

                    # Remove from memory cache
                    self.memory_cache.pop(key, None)

                    cleaned += 1

        logger.info(f"Cleaned {cleaned} expired cache entries")
        return cleaned

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*), SUM(cost_saved) FROM cache_entries")
            row = cursor.fetchone()
            total_entries = row[0] if row and row[0] is not None else 0
            total_saved = row[1] if row and row[1] is not None else 0.0

            # Calculate cache size - check both .pkl and .json files for compatibility
            cache_size = 0
            cache_size += sum(
                f.stat().st_size for f in self.cache_dir.glob("*.pkl") if f.is_file()
            )
            cache_size += sum(
                f.stat().st_size for f in self.cache_dir.glob("*.json") if f.is_file()
            )

        hit_rate = (
            self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])
            if (self.stats["hits"] + self.stats["misses"]) > 0
            else 0
        )

        return {
            "hit_rate": f"{hit_rate:.2%}",
            "total_entries": total_entries,
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_size_mb": cache_size / (1024 * 1024),
            "total_cost_saved": f"${total_saved:.2f}",
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
        }

    def clear_by_tags(self, tags: list[str]) -> int:
        """Clear cache entries by tags.

        Args:
            tags: Tags to match

        Returns:
            Number of entries cleared
        """
        cleared = 0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT key, tags, file_path FROM cache_entries")

            for row in cursor:
                key, entry_tags_json, file_path = row
                entry_tags = json.loads(entry_tags_json) if entry_tags_json else []

                if any(tag in entry_tags for tag in tags):
                    # Remove file
                    file_path = Path(file_path)
                    if file_path.exists():
                        file_path.unlink()

                    # Remove from database
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))

                    # Remove from memory cache
                    self.memory_cache.pop(key, None)

                    cleared += 1

        logger.info(f"Cleared {cleared} cache entries with tags: {tags}")
        return cleared

    def delete(self, key: str) -> bool:
        """Delete a specific cache entry.

        Args:
            key: Cache key to delete

        Returns:
            True if key was found and deleted, False otherwise
        """
        try:
            with self._lock:  # Thread safety
                cache_file = self._get_cache_file(key)
                if cache_file.exists():
                    cache_file.unlink()
                    return True
                return False
        except Exception:
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            with self._lock:  # Thread safety
                # Clear SQLite database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM cache_entries")

                # Remove all cache files (both .json and .pkl for compatibility)
                for cache_file in self.cache_dir.glob("*.json"):
                    try:
                        cache_file.unlink()
                    except Exception:
                        pass  # Continue even if one file fails

                for cache_file in self.cache_dir.glob("*.pkl"):
                    try:
                        cache_file.unlink()
                    except Exception:
                        pass  # Continue even if one file fails

                # Clear memory cache
                self.memory_cache.clear()

                # Reset stats
                self.stats = {"hits": 0, "misses": 0}

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            pass  # Continue even if clearing fails

    def _get_cache_file(self, key: str) -> Path:
        """Get the file path for a cache key."""
        # Create a hash of the key to avoid filesystem issues
        key_hash = hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()
        return self.cache_dir / f"{key_hash}.json"


# Global cache manager instance
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def cached_llm_call(model: str, cost_per_call: float = 0.01):
    """Decorator for caching LLM API calls.

    Args:
        model: Model name
        cost_per_call: Estimated cost per API call
    """

    def decorator(func):
        def wrapper(prompt: str, **kwargs):
            cache = get_cache_manager()

            # Try to get from cache first
            cached_response = cache.get_llm_response(prompt, model)
            if cached_response:
                logger.debug(f"Cache hit for LLM call: {model}")
                return cached_response

            # Make actual API call
            response = func(prompt, **kwargs)

            # Cache the response
            cache.cache_llm_response(prompt, model, response, cost_per_call)
            logger.debug(
                f"Cached LLM response: {model}, cost saved: ${cost_per_call:.4f}"
            )

            return response

        return wrapper

    return decorator
