"""Rate limiting utilities for API calls and resource management.

This module provides comprehensive rate limiting for:
- API calls (OpenAI, external services)
- Model inference requests
- Data processing operations
- Cost management and monitoring
"""

import asyncio
import functools
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RateLimit:
    """Rate limit configuration."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    cost_limit_per_day: float = 100.0  # USD
    burst_limit: int = 10  # Allow short bursts


class RateLimiter:
    """Thread-safe rate limiter with cost tracking."""

    def __init__(self, config: dict[str, RateLimit]):
        """Initialize rate limiter with configuration.

        Args:
            config: Dictionary mapping service names to rate limits
        """
        self.config = config
        self.request_history: dict[str, deque] = defaultdict(deque)
        self.cost_tracking: dict[str, dict[str, float]] = defaultdict(
            lambda: {"daily": 0.0, "last_reset": time.time()}
        )
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def acquire(self, service: str, cost: float = 0.0) -> bool:
        """Acquire permission to make a request.

        Args:
            service: Service name (e.g., 'openai', 'huggingface')
            cost: Estimated cost of the request in USD

        Returns:
            True if request is allowed, False if rate limited
        """
        if service not in self.config:
            logger.warning(f"No rate limit config for service: {service}")
            return True

        async with self._locks[service]:
            return self._check_limits(service, cost)

    def _check_limits(self, service: str, cost: float) -> bool:
        """Check if request is within rate limits."""
        limits = self.config[service]
        now = time.time()
        history = self.request_history[service]

        # Clean old requests and get counts
        request_counts = self._get_request_counts(history, now)

        # Check rate limits
        if not self._check_rate_limits(service, limits, request_counts):
            return False

        # Check cost limits
        if not self._check_cost_limits(service, limits, cost, now):
            return False

        # Record the request
        self._record_request(service, now, cost)
        return True

    def _get_request_counts(self, history: Any, now: float) -> dict[str, int]:
        """Get request counts for different time windows."""
        # Clean old requests
        cutoff_day = now - 86400
        while history and history[0]["timestamp"] < cutoff_day:
            history.popleft()

        # Count requests in time windows
        cutoff_minute = now - 60
        cutoff_hour = now - 3600

        minute_count = sum(1 for req in history if req["timestamp"] > cutoff_minute)
        hour_count = sum(1 for req in history if req["timestamp"] > cutoff_hour)
        day_count = len(history)

        return {"minute": minute_count, "hour": hour_count, "day": day_count}

    def _check_rate_limits(
        self, service: str, limits: Any, counts: dict[str, int]
    ) -> bool:
        """Check if request counts are within limits."""
        if counts["minute"] >= limits.requests_per_minute:
            logger.warning(f"Rate limit exceeded for {service}: {counts['minute']}/min")
            return False

        if counts["hour"] >= limits.requests_per_hour:
            logger.warning(f"Rate limit exceeded for {service}: {counts['hour']}/hour")
            return False

        if counts["day"] >= limits.requests_per_day:
            logger.warning(f"Rate limit exceeded for {service}: {counts['day']}/day")
            return False

        return True

    def _check_cost_limits(
        self, service: str, limits: Any, cost: float, now: float
    ) -> bool:
        """Check if cost is within limits."""
        cost_info = self.cost_tracking[service]

        # Reset daily costs if needed
        if now - cost_info["last_reset"] > 86400:
            cost_info["daily"] = 0.0
            cost_info["last_reset"] = now

        if cost_info["daily"] + cost > limits.cost_limit_per_day:
            logger.warning(
                f"Cost limit exceeded for {service}: ${cost_info['daily']:.2f}/day"
            )
            return False

        return True

    def _record_request(self, service: str, now: float, cost: float) -> None:
        """Record the request in history and cost tracking."""
        history = self.request_history[service]
        cost_info = self.cost_tracking[service]

        history.append({"timestamp": now, "cost": cost})
        cost_info["daily"] += cost

    def get_status(self, service: str) -> dict[str, Any]:
        """Get current rate limit status for a service."""
        if service not in self.config:
            return {"error": f"No config for service: {service}"}

        history = self.request_history[service]
        now = time.time()

        minute_count = sum(1 for req in history if req["timestamp"] > now - 60)
        hour_count = sum(1 for req in history if req["timestamp"] > now - 3600)
        day_count = len(history)

        limits = self.config[service]
        cost_info = self.cost_tracking[service]

        return {
            "service": service,
            "requests": {
                "minute": f"{minute_count}/{limits.requests_per_minute}",
                "hour": f"{hour_count}/{limits.requests_per_hour}",
                "day": f"{day_count}/{limits.requests_per_day}",
            },
            "costs": {
                "daily": f"${cost_info['daily']:.2f}/${limits.cost_limit_per_day:.2f}",
            },
            "available": {
                "minute": limits.requests_per_minute - minute_count,
                "hour": limits.requests_per_hour - hour_count,
                "day": limits.requests_per_day - day_count,
            },
        }


# Global rate limiter instance
_default_limiter = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _default_limiter
    if _default_limiter is None:
        # Default configuration
        default_config = {
            "openai": RateLimit(
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=10000,
                cost_limit_per_day=50.0,
            ),
            "huggingface": RateLimit(
                requests_per_minute=100,
                requests_per_hour=2000,
                requests_per_day=20000,
                cost_limit_per_day=20.0,
            ),
            "inference": RateLimit(
                requests_per_minute=1000,
                requests_per_hour=10000,
                requests_per_day=100000,
                cost_limit_per_day=0.0,  # No cost for local inference
            ),
        }
        _default_limiter = RateLimiter(default_config)
    return _default_limiter


def rate_limited(service: str, cost: float = 0.0):
    """Decorator for rate limiting function calls.

    Args:
        service: Service name for rate limiting
        cost: Estimated cost per call
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            limiter = get_rate_limiter()
            if await limiter.acquire(service, cost):
                return await func(*args, **kwargs)
            else:
                raise Exception(f"Rate limit exceeded for {service}")

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            limiter = get_rate_limiter()
            # For sync functions, we need to run acquire in event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, create a task
                    async def check_limit():
                        return await limiter.acquire(service, cost)

                    task = asyncio.create_task(check_limit())
                    # Wait for the task to complete
                    while not task.done():
                        pass
                    allowed = task.result()
                else:
                    allowed = asyncio.run(limiter.acquire(service, cost))
            except Exception:
                # No event loop, create one
                allowed = asyncio.run(limiter.acquire(service, cost))

            if allowed:
                return func(*args, **kwargs)
            else:
                raise Exception(f"Rate limit exceeded for {service}")

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def wait_for_capacity(service: str, max_wait: int = 300) -> bool:
    """Wait until rate limit capacity is available.

    Args:
        service: Service name
        max_wait: Maximum wait time in seconds

    Returns:
        True if capacity became available, False if timeout
    """
    limiter = get_rate_limiter()
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            if asyncio.run(limiter.acquire(service, 0.0)):
                return True
        except Exception:
            pass
        time.sleep(1)

    return False


# Cost estimation utilities
COST_ESTIMATES = {
    "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},  # Per token
    "gpt-3.5-turbo": {"input": 0.001 / 1000, "output": 0.002 / 1000},
    "text-embedding-ada-002": {"input": 0.0001 / 1000, "output": 0.0},
}


def estimate_openai_cost(
    model: str, input_tokens: int, output_tokens: int = 0
) -> float:
    """Estimate cost for OpenAI API call.

    Args:
        model: Model name
        input_tokens: Input token count
        output_tokens: Output token count

    Returns:
        Estimated cost in USD
    """
    if model not in COST_ESTIMATES:
        logger.warning(f"No cost estimate for model: {model}")
        return 0.01  # Default small cost

    costs = COST_ESTIMATES[model]
    return (input_tokens * costs["input"]) + (output_tokens * costs["output"])
