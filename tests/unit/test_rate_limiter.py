"""
Comprehensive tests for the RateLimiter functionality.

This module tests the rate limiting capabilities including:
- Basic rate limiting functionality
- Async function decorators
- Service-specific rate limits
- Cost estimation
- Concurrent access patterns
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.platform.utils.rate_limiter import (
    RateLimit,
    RateLimiter,
    estimate_openai_cost,
    get_rate_limiter,
    rate_limited,
    wait_for_capacity,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minute_config() -> dict[str, RateLimit]:
    """Configuration with a *very* small minute limit for fast tests."""

    return {
        "test_service": RateLimit(
            requests_per_minute=5,
            requests_per_hour=20,
            requests_per_day=100,
            cost_limit_per_day=50.0,
        )
    }


@pytest.fixture()
def cost_config() -> dict[str, RateLimit]:
    """Configuration where rate counts are high so we can test cost limits."""

    return {
        "test_service": RateLimit(
            requests_per_minute=100,
            requests_per_hour=200,
            requests_per_day=300,
            cost_limit_per_day=10.0,
        )
    }


@pytest.fixture()
def limiter(minute_config: dict[str, RateLimit]) -> RateLimiter:
    """Rate limiter using the *minute* test configuration."""

    return RateLimiter(minute_config)


@pytest.fixture()
def cost_limiter(cost_config: dict[str, RateLimit]) -> RateLimiter:
    """Rate limiter using the *cost* test configuration."""

    return RateLimiter(cost_config)


# ---------------------------------------------------------------------------
# Basic dataclass & construction tests
# ---------------------------------------------------------------------------


def test_rate_limit_defaults() -> None:
    """Ensure default field values are sane."""

    default = RateLimit()

    assert default.requests_per_minute == 60
    assert default.requests_per_hour == 1000
    assert default.requests_per_day == 10000
    assert default.cost_limit_per_day == 100.0
    assert default.burst_limit == 10


def test_rate_limit_custom_values() -> None:
    """Custom values should be retained verbatim."""

    custom = RateLimit(
        requests_per_minute=30,
        requests_per_hour=500,
        requests_per_day=5000,
        cost_limit_per_day=25.0,
        burst_limit=5,
    )

    assert custom.requests_per_minute == 30
    assert custom.requests_per_hour == 500
    assert custom.requests_per_day == 5000
    assert custom.cost_limit_per_day == 25.0
    assert custom.burst_limit == 5


# ---------------------------------------------------------------------------
# RateLimiter behavioural tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acquire_success(limiter: RateLimiter) -> None:
    """A simple request should succeed when under all limits."""

    assert await limiter.acquire("test_service", cost=1.0)


@pytest.mark.asyncio
async def test_acquire_unknown_service(limiter: RateLimiter) -> None:
    """Unknown services are permitted but should log a warning."""

    # Unknown services default to *allowed* behaviour.
    assert await limiter.acquire("unknown_service", cost=1.0)


@pytest.mark.asyncio
async def test_minute_rate_limit(limiter: RateLimiter) -> None:
    """The limiter should deny requests over the per-minute quota."""

    # 5 requests are allowed (limit == 5)
    for _ in range(5):
        allowed = await limiter.acquire("test_service", cost=0.1)
        assert allowed is True

    # 6th request exceeds the quota and should be denied.
    allowed = await limiter.acquire("test_service", cost=0.1)
    assert allowed is False


@pytest.mark.asyncio
async def test_daily_cost_reset(cost_limiter: RateLimiter) -> None:
    """Cost tracking should reset after 24h."""

    # Patch *time.time* so we can simulate the passage of a day.
    with patch("time.time") as mock_time:
        # Start of day.
        mock_time.return_value = 0.0

        # Use up the daily cost limit.
        for _ in range(5):
            allowed = cost_limiter._check_limits("test_service", cost=2.0)  # type: ignore[attr-defined]
            assert allowed is True

        # Next request is over the daily USD limit (10.0) and must be denied.
        assert cost_limiter._check_limits("test_service", cost=1.0) is False  # type: ignore[attr-defined]

        # Advance mock time > 24h to trigger cost reset.
        mock_time.return_value = 24 * 60 * 60 + 1

        # Now it should succeed again.
        assert cost_limiter._check_limits("test_service", cost=1.0) is True  # type: ignore[attr-defined]


def test_get_status(limiter: RateLimiter) -> None:
    """Status dictionary should contain expected keys & values."""

    status = limiter.get_status("test_service")

    assert status["service"] == "test_service"
    assert set(status).issuperset({"requests", "costs", "available"})


def test_get_status_unknown_service(limiter: RateLimiter) -> None:
    """Unknown services should yield an error status."""

    status = limiter.get_status("unknown_service")
    assert "error" in status


# ---------------------------------------------------------------------------
# Decorator helpers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_function_decoration() -> None:
    """The decorator should await *acquire* and bubble the result."""

    async def _make_future(value: bool) -> asyncio.Future:  # helper
        fut: asyncio.Future = asyncio.Future()
        fut.set_result(value)
        return fut

    with patch("src.platform.utils.rate_limiter.get_rate_limiter") as mock_get_limiter:
        mock_limiter = AsyncMock()
        mock_limiter.acquire.side_effect = lambda *_a, **_kw: _make_future(True)
        mock_get_limiter.return_value = mock_limiter

        @rate_limited(service="test", cost=1.0)
        async def func():  # pragma: no cover – behaviour tested via result
            return "success"

        assert await func() == "success"
        mock_limiter.acquire.assert_called_once_with("test", 1.0)


@pytest.mark.asyncio
async def test_async_function_rate_limited():
    """Test rate limited decorator with async function."""
    # Create a custom limiter with very low limits
    test_config = {
        "test": RateLimit(
            requests_per_minute=2,  # Allow 2 requests
            requests_per_hour=2,
            requests_per_day=2,
            cost_limit_per_day=0.2,  # Allow some cost initially
        )
    }

    test_limiter = RateLimiter(test_config)

    # Patch the global rate limiter to use our test instance
    with patch(
        "src.platform.utils.rate_limiter.get_rate_limiter", return_value=test_limiter
    ):

        @rate_limited("test", cost=0.1)  # Moderate cost
        async def test_function():
            return "success"

        # First call should succeed
        result = await test_function()
        assert result == "success"

        # Second call should still succeed
        result2 = await test_function()
        assert result2 == "success"

        # Third call should fail due to request limit
        with pytest.raises(Exception, match="Rate limit exceeded for test"):
            await test_function()


def test_sync_function_decoration() -> None:
    """Decorator should work for synchronous functions as well."""

    with (
        patch("src.platform.utils.rate_limiter.get_rate_limiter") as mock_get_limiter,
        patch("asyncio.run", return_value=True) as mock_run,
    ):
        mock_get_limiter.return_value = AsyncMock()

        @rate_limited(service="test", cost=1.0)
        def func():  # pragma: no cover
            return "sync-success"

        assert func() == "sync-success"
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


def test_estimate_openai_cost() -> None:
    """Ensure cost estimation matches the reference table."""

    cost_gpt4 = estimate_openai_cost("gpt-4", 1_000, 500)
    expected_gpt4 = (1_000 * 0.03 / 1_000) + (500 * 0.06 / 1_000)
    assert cost_gpt4 == expected_gpt4

    cost_turbo = estimate_openai_cost("gpt-3.5-turbo", 1_000, 500)
    expected_turbo = (1_000 * 0.001 / 1_000) + (500 * 0.002 / 1_000)
    assert cost_turbo == expected_turbo

    cost_embed = estimate_openai_cost("text-embedding-ada-002", 1_000)
    expected_embed = 1_000 * 0.0001 / 1_000
    assert cost_embed == expected_embed

    # Unknown model falls back to default 0.01
    assert estimate_openai_cost("unknown", 1_000) == 0.01


def test_wait_for_capacity_success() -> None:
    """wait_for_capacity() should return *True* immediately when capacity exists."""

    with (
        patch("src.platform.utils.rate_limiter.get_rate_limiter") as mock_get_limiter,
        patch("asyncio.run", return_value=True),
        patch("time.sleep"),
    ):
        mock_get_limiter.return_value = AsyncMock()
        assert wait_for_capacity("svc", max_wait=2) is True


def test_wait_for_capacity_timeout() -> None:
    """Should eventually give up when capacity never becomes available."""

    with (
        patch("src.platform.utils.rate_limiter.get_rate_limiter") as mock_get_limiter,
        patch("asyncio.run", return_value=False),
        patch("time.sleep"),
        patch("time.time") as mock_time,
    ):
        mock_get_limiter.return_value = AsyncMock()
        mock_time.side_effect = [0, 1, 2, 3, 4, 5, 6]
        assert wait_for_capacity("svc", max_wait=5) is False


# ---------------------------------------------------------------------------
# Global singleton tests
# ---------------------------------------------------------------------------


def test_get_rate_limiter_singleton() -> None:
    """The helper should return the same instance every time."""

    assert get_rate_limiter() is get_rate_limiter()


def test_default_configuration() -> None:
    """The default limiter should contain expected service presets."""

    limiter = get_rate_limiter()
    for svc in ("openai", "huggingface", "inference"):
        assert svc in limiter.config


# ---------------------------------------------------------------------------
# Integration tests (limited in scope to keep runtime low)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_requests() -> None:
    """Multiple concurrent requests should succeed within limits."""

    config = {
        "svc": RateLimit(
            requests_per_minute=10,
            requests_per_hour=100,
            requests_per_day=1_000,
            cost_limit_per_day=50.0,
        )
    }
    limiter = RateLimiter(config)

    async def make_req() -> bool:
        return await limiter.acquire("svc", cost=1.0)

    # Launch 5 concurrent tasks – all should be allowed.
    results = await asyncio.gather(*(make_req() for _ in range(5)))
    assert all(results)

    # Status should reflect 5 requests.
    status = limiter.get_status("svc")
    assert status["requests"]["minute"].startswith("5/")
