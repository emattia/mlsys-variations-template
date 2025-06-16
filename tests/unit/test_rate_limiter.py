"""Unit tests for the rate limiting system."""

import asyncio
import pytest
import time
from unittest.mock import patch, MagicMock

from src.utils.rate_limiter import (
    RateLimit,
    RateLimiter,
    get_rate_limiter,
    rate_limited,
    estimate_openai_cost,
    wait_for_capacity
)


class TestRateLimit:
    """Test RateLimit dataclass."""
    
    def test_rate_limit_defaults(self):
        """Test RateLimit default values."""
        limit = RateLimit()
        assert limit.requests_per_minute == 60
        assert limit.requests_per_hour == 1000
        assert limit.requests_per_day == 10000
        assert limit.cost_limit_per_day == 100.0
        assert limit.burst_limit == 10
    
    def test_rate_limit_custom_values(self):
        """Test RateLimit with custom values."""
        limit = RateLimit(
            requests_per_minute=30,
            requests_per_hour=500,
            requests_per_day=5000,
            cost_limit_per_day=25.0,
            burst_limit=5
        )
        assert limit.requests_per_minute == 30
        assert limit.requests_per_hour == 500
        assert limit.requests_per_day == 5000
        assert limit.cost_limit_per_day == 25.0
        assert limit.burst_limit == 5


class TestRateLimiter:
    """Test RateLimiter class."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            "test_service": RateLimit(
                requests_per_minute=5,
                requests_per_hour=20,
                requests_per_day=100,
                cost_limit_per_day=10.0
            )
        }
    
    @pytest.fixture
    def limiter(self, config):
        """Rate limiter instance."""
        return RateLimiter(config)
    
    @pytest.mark.asyncio
    async def test_acquire_success(self, limiter):
        """Test successful request acquisition."""
        result = await limiter.acquire("test_service", cost=1.0)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_acquire_unknown_service(self, limiter):
        """Test acquisition for unknown service."""
        result = await limiter.acquire("unknown_service", cost=1.0)
        assert result is True  # Should allow unknown services with warning
    
    @pytest.mark.asyncio
    async def test_minute_rate_limit(self, limiter):
        """Test minute-based rate limiting."""
        # Make requests up to the limit
        for _ in range(5):
            result = await limiter.acquire("test_service", cost=0.5)
            assert result is True
        
        # Next request should be denied
        result = await limiter.acquire("test_service", cost=0.5)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_cost_limit(self, limiter):
        """Test cost-based limiting."""
        # Make requests up to cost limit
        for _ in range(5):
            result = await limiter.acquire("test_service", cost=2.0)
            assert result is True
        
        # Next request should be denied due to cost
        result = await limiter.acquire("test_service", cost=1.0)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_daily_cost_reset(self, limiter):
        """Test daily cost reset functionality."""
        # Mock time to simulate day passage
        with patch('time.time') as mock_time:
            # Start at time 0
            mock_time.return_value = 0
            
            # Use up daily cost limit
            for _ in range(5):
                await limiter.acquire("test_service", cost=2.0)
            
            # Should be at cost limit
            result = await limiter.acquire("test_service", cost=1.0)
            assert result is False
            
            # Advance time by more than 24 hours
            mock_time.return_value = 86401  # 24 hours + 1 second
            
            # Should be able to make requests again
            result = await limiter.acquire("test_service", cost=1.0)
            assert result is True
    
    def test_get_status(self, limiter):
        """Test status reporting."""
        status = limiter.get_status("test_service")
        
        assert "service" in status
        assert "requests" in status
        assert "costs" in status
        assert "available" in status
        assert status["service"] == "test_service"
    
    def test_get_status_unknown_service(self, limiter):
        """Test status for unknown service."""
        status = limiter.get_status("unknown_service")
        assert "error" in status


class TestRateLimitedDecorator:
    """Test rate_limited decorator."""
    
    @pytest.mark.asyncio
    async def test_async_function_decoration(self):
        """Test decorator on async function."""
        @rate_limited(service="test", cost=1.0)
        async def test_func():
            return "success"
        
        # Mock the rate limiter to always allow
        with patch('src.utils.rate_limiter.get_rate_limiter') as mock_get_limiter:
            mock_limiter = MagicMock()
            mock_limiter.acquire = MagicMock(return_value=asyncio.Future())
            mock_limiter.acquire.return_value.set_result(True)
            mock_get_limiter.return_value = mock_limiter
            
            result = await test_func()
            assert result == "success"
            mock_limiter.acquire.assert_called_once_with("test", 1.0)
    
    @pytest.mark.asyncio
    async def test_async_function_rate_limited(self):
        """Test decorator when rate limited."""
        @rate_limited(service="test", cost=1.0)
        async def test_func():
            return "success"
        
        # Mock the rate limiter to deny
        with patch('src.utils.rate_limiter.get_rate_limiter') as mock_get_limiter:
            mock_limiter = MagicMock()
            mock_limiter.acquire = MagicMock(return_value=asyncio.Future())
            mock_limiter.acquire.return_value.set_result(False)
            mock_get_limiter.return_value = mock_limiter
            
            with pytest.raises(Exception, match="Rate limit exceeded for test"):
                await test_func()
    
    def test_sync_function_decoration(self):
        """Test decorator on sync function."""
        @rate_limited(service="test", cost=1.0)
        def test_func():
            return "success"
        
        # Mock the rate limiter and asyncio.run
        with patch('src.utils.rate_limiter.get_rate_limiter') as mock_get_limiter, \
             patch('asyncio.run') as mock_run:
            
            mock_limiter = MagicMock()
            mock_get_limiter.return_value = mock_limiter
            mock_run.return_value = True
            
            result = test_func()
            assert result == "success"
            mock_run.assert_called_once()


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_estimate_openai_cost_gpt4(self):
        """Test cost estimation for GPT-4."""
        cost = estimate_openai_cost("gpt-4", input_tokens=1000, output_tokens=500)
        expected = (1000 * 0.03/1000) + (500 * 0.06/1000)
        assert cost == expected
    
    def test_estimate_openai_cost_gpt35_turbo(self):
        """Test cost estimation for GPT-3.5 Turbo."""
        cost = estimate_openai_cost("gpt-3.5-turbo", input_tokens=1000, output_tokens=500)
        expected = (1000 * 0.001/1000) + (500 * 0.002/1000)
        assert cost == expected
    
    def test_estimate_openai_cost_unknown_model(self):
        """Test cost estimation for unknown model."""
        cost = estimate_openai_cost("unknown-model", input_tokens=1000)
        assert cost == 0.01  # Default cost
    
    def test_estimate_openai_cost_embedding(self):
        """Test cost estimation for embedding model."""
        cost = estimate_openai_cost("text-embedding-ada-002", input_tokens=1000)
        expected = 1000 * 0.0001/1000
        assert cost == expected
    
    def test_wait_for_capacity_success(self):
        """Test waiting for capacity when it becomes available."""
        with patch('src.utils.rate_limiter.get_rate_limiter') as mock_get_limiter, \
             patch('asyncio.run') as mock_run, \
             patch('time.sleep'):
            
            mock_limiter = MagicMock()
            mock_get_limiter.return_value = mock_limiter
            mock_run.return_value = True
            
            result = wait_for_capacity("test_service", max_wait=5)
            assert result is True
    
    def test_wait_for_capacity_timeout(self):
        """Test waiting for capacity with timeout."""
        with patch('src.utils.rate_limiter.get_rate_limiter') as mock_get_limiter, \
             patch('asyncio.run') as mock_run, \
             patch('time.sleep'), \
             patch('time.time') as mock_time:
            
            mock_limiter = MagicMock()
            mock_get_limiter.return_value = mock_limiter
            mock_run.return_value = False  # Always rate limited
            
            # Mock time progression
            mock_time.side_effect = [0, 1, 2, 3, 4, 5, 6]  # Exceed max_wait
            
            result = wait_for_capacity("test_service", max_wait=5)
            assert result is False


class TestGlobalRateLimiter:
    """Test global rate limiter functionality."""
    
    def test_get_rate_limiter_singleton(self):
        """Test that get_rate_limiter returns singleton."""
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        assert limiter1 is limiter2
    
    def test_default_configuration(self):
        """Test default rate limiter configuration."""
        limiter = get_rate_limiter()
        
        assert "openai" in limiter.config
        assert "huggingface" in limiter.config
        assert "inference" in limiter.config
        
        # Check OpenAI config
        openai_config = limiter.config["openai"]
        assert openai_config.requests_per_minute == 60
        assert openai_config.cost_limit_per_day == 50.0


@pytest.mark.integration
class TestRateLimiterIntegration:
    """Integration tests for rate limiter."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test rate limiter with concurrent requests."""
        config = {
            "test_service": RateLimit(
                requests_per_minute=10,
                requests_per_hour=100,
                requests_per_day=1000,
                cost_limit_per_day=50.0
            )
        }
        limiter = RateLimiter(config)
        
        # Make multiple concurrent requests
        tasks = [
            limiter.acquire("test_service", cost=1.0)
            for _ in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed within limits
        assert all(results)
        
        # Check status
        status = limiter.get_status("test_service")
        assert "5/10" in status["requests"]["minute"]
    
    @pytest.mark.asyncio
    async def test_realistic_usage_pattern(self):
        """Test realistic usage pattern with bursts and delays."""
        config = {
            "api": RateLimit(
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=10000,
                cost_limit_per_day=100.0
            )
        }
        limiter = RateLimiter(config)
        
        # Simulate burst of requests
        burst_results = []
        for _ in range(30):
            result = await limiter.acquire("api", cost=0.01)
            burst_results.append(result)
        
        # All should succeed
        assert all(burst_results)
        
        # Continue until rate limit
        continue_results = []
        for _ in range(35):  # This should exceed minute limit
            result = await limiter.acquire("api", cost=0.01)
            continue_results.append(result)
        
        # Some should fail due to rate limiting
        assert not all(continue_results)
        assert any(continue_results)  # But some should still succeed 