"""Base classes for LLM providers and token estimation."""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ModelType(Enum):
    """Types of LLM models."""

    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""

    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 30
    max_retries: int = 3
    model_type: ModelType = ModelType.CHAT


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    usage: dict[str, int] | None = None
    finish_reason: str | None = None
    metadata: dict[str, Any] | None = None


class TokenEstimator:
    """Utility for estimating token usage and costs."""

    # Token costs per 1K tokens (approximate, update with current pricing)
    TOKEN_COSTS = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }

    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """
        Estimate token count for text.

        This is a rough approximation. For production use, consider
        using tiktoken or the specific tokenizer for your model.
        """
        # Rough approximation: 1 token ≈ 4 characters for English text
        return len(text) // 4

    @classmethod
    def estimate_cost(cls, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request."""
        if model not in cls.TOKEN_COSTS:
            # Default to GPT-4 pricing for unknown models
            model = "gpt-4"

        costs = cls.TOKEN_COSTS[model]
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]

        return input_cost + output_cost

    @classmethod
    def get_model_info(cls, model: str) -> dict[str, Any]:
        """Get information about a model including costs."""
        return {
            "model": model,
            "costs_per_1k_tokens": cls.TOKEN_COSTS.get(model, cls.TOKEN_COSTS["gpt-4"]),
            "estimated_tokens_per_char": 0.25,  # 1 token ≈ 4 characters
        }


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass

    @abstractmethod
    async def generate_with_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text and return detailed response including usage."""
        pass

    @abstractmethod
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate text as a stream."""
        pass

    async def batch_generate(self, prompts: list[str], **kwargs) -> list[str]:
        """Generate text for multiple prompts."""
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def estimate_cost(
        self, prompt: str, max_output_tokens: int = None
    ) -> dict[str, Any]:
        """Estimate the cost for a request."""
        input_tokens = TokenEstimator.estimate_tokens(prompt)
        output_tokens = max_output_tokens or self.config.max_tokens

        cost = TokenEstimator.estimate_cost(
            self.config.model, input_tokens, output_tokens
        )

        return {
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
            "estimated_total_tokens": input_tokens + output_tokens,
            "estimated_cost_usd": cost,
            "model": self.config.model,
        }

    def _build_request_params(self, **kwargs) -> dict[str, Any]:
        """Build request parameters from config and kwargs."""
        params = {
            "model": self.config.model,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }

        # Add provider-specific parameters
        if hasattr(self.config, "frequency_penalty"):
            params["frequency_penalty"] = kwargs.get(
                "frequency_penalty", self.config.frequency_penalty
            )
        if hasattr(self.config, "presence_penalty"):
            params["presence_penalty"] = kwargs.get(
                "presence_penalty", self.config.presence_penalty
            )

        return params

    async def _retry_request(self, request_func, *args, **kwargs):
        """Retry a request with exponential backoff."""
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                return await request_func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                self.logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )

                if attempt < self.config.max_retries - 1:
                    # Exponential backoff
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)

        # All retries failed
        raise last_exception
