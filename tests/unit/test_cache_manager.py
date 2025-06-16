"""Unit tests for the cache management system."""

import tempfile
import time
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.utils.cache_manager import (
    CacheEntry,
    CacheManager,
    get_cache_manager,
    cached_llm_call
)


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestCacheEntry:
    """Test CacheEntry dataclass."""
    
    def test_cache_entry_creation(self):
        """Test CacheEntry creation with required fields."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time(),
            accessed_at=time.time(),
            access_count=1,
            cost_saved=0.5
        )
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.cost_saved == 0.5
        assert entry.tags == []  # Default empty list
    
    def test_cache_entry_with_tags(self):
        """Test CacheEntry with tags."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time(),
            accessed_at=time.time(),
            access_count=1,
            cost_saved=0.5,
            tags=["llm", "gpt-4"]
        )
        assert entry.tags == ["llm", "gpt-4"]


class TestCacheManager:
    """Test CacheManager class."""
    
    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Cache manager instance with temp directory."""
        return CacheManager(cache_dir=temp_cache_dir)
    
    def test_cache_manager_initialization(self, temp_cache_dir):
        """Test cache manager initialization."""
        manager = CacheManager(cache_dir=temp_cache_dir)
        
        assert manager.cache_dir == temp_cache_dir
        assert manager.cache_dir.exists()
        assert manager.db_path.exists()
        assert manager.max_memory_items == 1000
        assert manager.memory_cache == {}
    
    def test_generate_key_dict(self, cache_manager):
        """Test key generation from dictionary."""
        data = {"prompt": "test prompt", "model": "gpt-4"}
        key1 = cache_manager._generate_key(data, "llm")
        key2 = cache_manager._generate_key(data, "llm")
        
        # Same data should generate same key
        assert key1 == key2
        assert key1.startswith("llm_")
        assert len(key1.split("_")[1]) == 16  # 16 char hash
    
    def test_generate_key_different_data(self, cache_manager):
        """Test key generation for different data."""
        data1 = {"prompt": "test prompt", "model": "gpt-4"}
        data2 = {"prompt": "different prompt", "model": "gpt-4"}
        
        key1 = cache_manager._generate_key(data1, "llm")
        key2 = cache_manager._generate_key(data2, "llm")
        
        assert key1 != key2
    
    def test_set_and_get_basic(self, cache_manager):
        """Test basic set and get operations."""
        cache_manager.set("test_key", "test_value", cost_saved=1.0)
        
        result = cache_manager.get("test_key")
        assert result == "test_value"
        assert cache_manager.stats["hits"] == 1
        assert cache_manager.stats["total_cost_saved"] == 1.0
    
    def test_get_nonexistent_key(self, cache_manager):
        """Test getting non-existent key."""
        result = cache_manager.get("nonexistent")
        assert result is None
        assert cache_manager.stats["misses"] == 1
    
    def test_cache_with_ttl(self, cache_manager):
        """Test cache with TTL expiration."""
        # Set with very short TTL
        cache_manager.set("test_key", "test_value", ttl=0.1)
        
        # Should be available immediately
        result = cache_manager.get("test_key")
        assert result == "test_value"
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired
        result = cache_manager.get("test_key")
        assert result is None
    
    def test_memory_cache_lru_eviction(self, cache_manager):
        """Test LRU eviction in memory cache."""
        # Set max memory items to 2 for testing
        cache_manager.max_memory_items = 2
        
        # Add 3 items (should evict first one)
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        cache_manager.set("key3", "value3")
        
        # key1 should be evicted from memory but available from disk
        assert "key1" not in cache_manager.memory_cache
        assert "key2" in cache_manager.memory_cache
        assert "key3" in cache_manager.memory_cache
        
        # But should still be retrievable from disk
        result = cache_manager.get("key1")
        assert result == "value1"
    
    def test_llm_response_caching(self, cache_manager):
        """Test LLM response caching functionality."""
        prompt = "What is machine learning?"
        model = "gpt-4"
        response = "Machine learning is a subset of AI..."
        cost = 0.02
        
        # Cache the response
        key = cache_manager.cache_llm_response(prompt, model, response, cost)
        
        # Retrieve the response
        cached_response = cache_manager.get_llm_response(prompt, model)
        
        assert cached_response == response
        assert key.startswith("llm_")
    
    def test_embedding_caching(self, cache_manager):
        """Test embedding caching functionality."""
        text = "This is a test sentence"
        model = "text-embedding-ada-002"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        cost = 0.001
        
        # Cache the embedding
        key = cache_manager.cache_embedding(text, model, embedding, cost)
        
        # Retrieve the embedding
        cached_embedding = cache_manager.get_embedding(text, model)
        
        assert cached_embedding == embedding
        assert key.startswith("emb_")
    
    def test_clean_expired_entries(self, cache_manager):
        """Test cleaning expired cache entries."""
        # Add entries with different TTLs
        cache_manager.set("key1", "value1", ttl=0.1)
        cache_manager.set("key2", "value2", ttl=10.0)
        cache_manager.set("key3", "value3")  # No TTL
        
        # Wait for some to expire
        time.sleep(0.2)
        
        # Clean expired entries
        cleaned = cache_manager.clean_expired()
        
        assert cleaned == 1  # Only key1 should be cleaned
        assert cache_manager.get("key1") is None
        assert cache_manager.get("key2") == "value2"
        assert cache_manager.get("key3") == "value3"
    
    def test_cache_stats(self, cache_manager):
        """Test cache statistics."""
        # Add some entries
        cache_manager.set("key1", "value1", cost_saved=1.0)
        cache_manager.set("key2", "value2", cost_saved=2.0)
        
        # Access them
        cache_manager.get("key1")
        cache_manager.get("key2")
        cache_manager.get("nonexistent")
        
        stats = cache_manager.get_stats()
        
        assert "hit_rate" in stats
        assert "total_entries" in stats
        assert "total_cost_saved" in stats
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert "$3.00" in stats["total_cost_saved"]
    
    def test_clear_by_tags(self, cache_manager):
        """Test clearing cache entries by tags."""
        # Add entries with different tags
        cache_manager.set("key1", "value1", tags=["llm", "gpt-4"])
        cache_manager.set("key2", "value2", tags=["embedding"])
        cache_manager.set("key3", "value3", tags=["llm", "gpt-3.5"])
        
        # Clear entries with "llm" tag
        cleared = cache_manager.clear_by_tags(["llm"])
        
        assert cleared == 2
        assert cache_manager.get("key1") is None
        assert cache_manager.get("key2") == "value2"
        assert cache_manager.get("key3") is None
    
    def test_database_operations(self, cache_manager):
        """Test SQLite database operations."""
        # Add entry
        cache_manager.set("test_key", "test_value", cost_saved=1.5, tags=["test"])
        
        # Check database directly
        with sqlite3.connect(cache_manager.db_path) as conn:
            cursor = conn.execute("SELECT * FROM cache_entries WHERE key = ?", ("test_key",))
            row = cursor.fetchone()
            
            assert row is not None
            assert row[0] == "test_key"  # key
            assert row[4] == 1.5  # cost_saved
            assert '"test"' in row[6]  # tags (JSON)
    
    def test_disk_persistence(self, cache_manager):
        """Test that cache persists to disk."""
        # Add entry
        cache_manager.set("persistent_key", {"data": "complex_value"})
        
        # Create new cache manager with same directory
        new_manager = CacheManager(cache_dir=cache_manager.cache_dir)
        
        # Should be able to retrieve from disk
        result = new_manager.get("persistent_key")
        assert result == {"data": "complex_value"}


class TestCachedLLMCallDecorator:
    """Test cached_llm_call decorator."""
    
    def test_decorator_cache_hit(self):
        """Test decorator with cache hit."""
        @cached_llm_call(model="gpt-4", cost_per_call=0.02)
        def mock_llm_call(prompt: str):
            return f"Response to: {prompt}"
        
        with patch('src.utils.cache_manager.get_cache_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_llm_response.return_value = "Cached response"
            mock_get_manager.return_value = mock_manager
            
            result = mock_llm_call("Test prompt")
            
            assert result == "Cached response"
            mock_manager.get_llm_response.assert_called_once_with("Test prompt", "gpt-4")
            mock_manager.cache_llm_response.assert_not_called()
    
    def test_decorator_cache_miss(self):
        """Test decorator with cache miss."""
        @cached_llm_call(model="gpt-4", cost_per_call=0.02)
        def mock_llm_call(prompt: str):
            return f"Response to: {prompt}"
        
        with patch('src.utils.cache_manager.get_cache_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_llm_response.return_value = None  # Cache miss
            mock_get_manager.return_value = mock_manager
            
            result = mock_llm_call("Test prompt")
            
            assert result == "Response to: Test prompt"
            mock_manager.get_llm_response.assert_called_once_with("Test prompt", "gpt-4")
            mock_manager.cache_llm_response.assert_called_once_with(
                "Test prompt", "gpt-4", "Response to: Test prompt", 0.02
            )


class TestGlobalCacheManager:
    """Test global cache manager functionality."""
    
    def test_get_cache_manager_singleton(self):
        """Test that get_cache_manager returns singleton."""
        manager1 = get_cache_manager()
        manager2 = get_cache_manager()
        assert manager1 is manager2
    
    def test_default_cache_directory(self):
        """Test default cache directory creation."""
        manager = get_cache_manager()
        expected_path = Path("data/cache")
        
        # Should use default path or create it
        assert manager.cache_dir.name == "cache"


@pytest.mark.integration
class TestCacheManagerIntegration:
    """Integration tests for cache manager."""
    
    def test_full_workflow(self, temp_cache_dir):
        """Test complete caching workflow."""
        manager = CacheManager(cache_dir=temp_cache_dir)
        
        # Simulate LLM API calls
        prompts_and_responses = [
            ("What is AI?", "AI is artificial intelligence..."),
            ("Explain ML", "Machine learning is..."),
            ("What is AI?", "AI is artificial intelligence..."),  # Duplicate
        ]
        
        total_cost = 0
        
        for prompt, response in prompts_and_responses:
            # Check cache first
            cached = manager.get_llm_response(prompt, "gpt-4")
            
            if cached:
                # Cache hit - no cost
                actual_response = cached
                cost = 0
            else:
                # Cache miss - simulate API call
                actual_response = response
                cost = 0.02
                manager.cache_llm_response(prompt, "gpt-4", response, cost)
            
            total_cost += cost
        
        # Should have saved money on duplicate prompt
        assert total_cost == 0.04  # Only 2 API calls, not 3
        
        # Check stats
        stats = manager.get_stats()
        assert stats["hits"] == 1  # One cache hit
        assert "$0.04" in stats["total_cost_saved"]  # Cost saved from cache hit
    
    def test_concurrent_cache_access(self, temp_cache_dir):
        """Test concurrent access to cache."""
        import threading
        
        manager = CacheManager(cache_dir=temp_cache_dir)
        results = []
        errors = []
        
        def cache_worker(worker_id):
            try:
                for i in range(10):
                    key = f"worker_{worker_id}_key_{i}"
                    value = f"worker_{worker_id}_value_{i}"
                    
                    manager.set(key, value)
                    retrieved = manager.get(key)
                    
                    if retrieved == value:
                        results.append(True)
                    else:
                        results.append(False)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=cache_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert all(results), "Some cache operations failed"
        assert len(results) == 50  # 5 workers × 10 operations
    
    def test_large_cache_performance(self, temp_cache_dir):
        """Test cache performance with many entries."""
        manager = CacheManager(cache_dir=temp_cache_dir)
        
        # Add many entries
        num_entries = 1000
        start_time = time.time()
        
        for i in range(num_entries):
            manager.set(f"key_{i}", f"value_{i}", cost_saved=0.01)
        
        set_time = time.time() - start_time
        
        # Retrieve entries
        start_time = time.time()
        retrieved = 0
        
        for i in range(num_entries):
            if manager.get(f"key_{i}") is not None:
                retrieved += 1
        
        get_time = time.time() - start_time
        
        # Performance assertions
        assert retrieved == num_entries
        assert set_time < 10.0  # Should set 1000 entries in < 10 seconds
        assert get_time < 5.0   # Should retrieve 1000 entries in < 5 seconds
        
        # Check stats
        stats = manager.get_stats()
        assert stats["total_entries"] == num_entries
        assert "$10.00" in stats["total_cost_saved"] 