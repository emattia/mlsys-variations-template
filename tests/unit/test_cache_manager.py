"""Test cases for cache manager functionality."""

import tempfile
import time
from pathlib import Path


from src.utils.cache_manager import CacheManager


class TestCacheManager:
    """Test cases for CacheManager class."""

    def test_cache_manager_initialization(self):
        """Test CacheManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            manager = CacheManager(cache_dir=cache_dir)
            assert manager.cache_dir == cache_dir
            assert cache_dir.exists()

    def test_cache_manager_default_directory(self):
        """Test CacheManager with default directory."""
        manager = CacheManager()
        assert manager.cache_dir is not None
        assert manager.cache_dir.exists()

    def test_set_and_get_cache(self):
        """Test setting and getting cache values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            manager = CacheManager(cache_dir=cache_dir)

            # Set cache value
            test_data = {"key": "value", "number": 42}
            manager.set("test_key", test_data)

            # Get cache value
            retrieved_data = manager.get("test_key")
            assert retrieved_data == test_data

    def test_get_nonexistent_cache(self):
        """Test getting non-existent cache key."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            manager = CacheManager(cache_dir=cache_dir)

            # Should return None for non-existent key
            result = manager.get("nonexistent_key")
            assert result is None

    def test_get_with_default_value(self):
        """Test getting cache with default value."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            manager = CacheManager(cache_dir=cache_dir)

            default_value = {"default": True}
            result = manager.get("nonexistent_key", default=default_value)
            assert result == default_value

    def test_cache_expiration(self):
        """Test cache expiration functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            manager = CacheManager(cache_dir=cache_dir)

            # Set cache with short expiration
            test_data = "expires_soon"
            manager.set("expiring_key", test_data, ttl=1)  # 1 second TTL

            # Should be available immediately
            result = manager.get("expiring_key")
            assert result == test_data

            # Wait for expiration
            time.sleep(1.1)

            # Should be expired now
            result = manager.get("expiring_key")
            assert result is None

    def test_delete_cache(self):
        """Test deleting cache entries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            manager = CacheManager(cache_dir=cache_dir)

            # Set and verify cache
            manager.set("delete_me", "test_data")
            assert manager.get("delete_me") == "test_data"

            # Delete cache
            success = manager.delete("delete_me")
            assert success is True

            # Should be gone now
            assert manager.get("delete_me") is None

    def test_delete_nonexistent_cache(self):
        """Test deleting non-existent cache entry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            manager = CacheManager(cache_dir=cache_dir)

            # Try to delete non-existent key
            success = manager.delete("nonexistent")
            assert success is False

    def test_clear_cache(self):
        """Test clearing all cache entries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            manager = CacheManager(cache_dir=cache_dir)

            # Set multiple cache entries
            manager.set("key1", "value1")
            manager.set("key2", "value2")
            manager.set("key3", "value3")

            # Verify they exist
            assert manager.get("key1") == "value1"
            assert manager.get("key2") == "value2"
            assert manager.get("key3") == "value3"

            # Clear cache
            manager.clear()

            # All should be gone
            assert manager.get("key1") is None
            assert manager.get("key2") is None
            assert manager.get("key3") is None

    def test_cache_with_complex_data(self):
        """Test caching complex data structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            manager = CacheManager(cache_dir=cache_dir)

            complex_data = {
                "list": [1, 2, 3, 4, 5],
                "nested_dict": {
                    "inner_list": ["a", "b", "c"],
                    "inner_dict": {"x": 10, "y": 20},
                },
                "tuple": (1, 2, 3),
                "set": {1, 2, 3, 4, 5},
            }

            manager.set("complex_key", complex_data)
            retrieved_data = manager.get("complex_key")

            # Note: sets become lists when serialized/deserialized
            assert retrieved_data["list"] == [1, 2, 3, 4, 5]
            assert retrieved_data["nested_dict"]["inner_list"] == ["a", "b", "c"]
            assert retrieved_data["nested_dict"]["inner_dict"]["x"] == 10


class TestCacheIntegration:
    """Integration test cases for cache functionality."""

    def test_cache_performance_optimization(self):
        """Test cache performance optimization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            manager = CacheManager(cache_dir=cache_dir)

            def expensive_operation(value):
                # Simulate processing time
                time.sleep(0.1)
                return f"processed_{value}"

            # First call should be slow
            start_time = time.time()
            result1 = expensive_operation("test")
            manager.set("test_key", result1)
            first_call_time = time.time() - start_time

            # Second call should be fast (from cache)
            start_time = time.time()
            result2 = manager.get("test_key")
            second_call_time = time.time() - start_time

            assert result1 == result2
            assert second_call_time < first_call_time

    def test_cache_persistence_across_sessions(self):
        """Test cache persistence across different manager instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"

            # First manager instance
            manager1 = CacheManager(cache_dir=cache_dir)
            manager1.set("persistent_key", "persistent_value")

            # Second manager instance (simulating restart)
            manager2 = CacheManager(cache_dir=cache_dir)
            result = manager2.get("persistent_key")

            assert result == "persistent_value"

    def test_concurrent_cache_access(self):
        """Test concurrent cache access."""
        import threading
        import time

        cache = CacheManager()

        results = {}

        def set_and_get(i):
            key = f"concurrent_key_{i}"
            value = f"value_{i}"
            # Set value
            cache.set(key, value)
            # Small delay to ensure value is written
            time.sleep(0.001)
            # Get value
            result = cache.get(key)
            results[i] = result

        # Create and start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=set_and_get, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check that each thread got its own value correctly
        for i in range(3):
            assert i in results, f"Thread {i} did not complete"
            assert (
                results[i] == f"value_{i}"
            ), f"Thread {i} got wrong value: {results[i]}"
