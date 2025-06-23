"""Anthropic Claude LLM provider implementation."""

from collections.abc import AsyncGenerator
from typing import Any

from .base import LLMConfig, LLMProvider, LLMResponse

# Try to import anthropic at module level for testing
try:
    import anthropic
except ImportError:
    # Create a mock module for testing when anthropic is not installed
    import types

    anthropic = types.ModuleType("anthropic")
    anthropic.AsyncAnthropic = None


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Anthropic client."""
        # Check if we have a real anthropic module, or if AsyncAnthropic is mocked for testing
        if not hasattr(anthropic, "__version__") and anthropic.AsyncAnthropic is None:
            self.logger.error(
                "Anthropic library not installed. Install with: pip install anthropic"
            )
            raise ImportError("Anthropic library not available")

        try:
            self.client = anthropic.AsyncAnthropic(
                api_key=self.config.api_key, timeout=self.config.timeout
            )

            self.logger.info(
                f"Anthropic client initialized for model: {self.config.model}"
            )

        except Exception as e:
            self.logger.error(f"Error initializing Anthropic client: {e}")
            raise

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic API."""
        response = await self.generate_with_response(prompt, **kwargs)
        return response.content

    async def generate_with_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text and return detailed response."""
        if not self.client:
            raise ValueError("Anthropic client not initialized")

        # Build messages for Claude
        messages = [{"role": "user", "content": prompt}]

        # Build request parameters
        params = self._build_anthropic_params(**kwargs)
        params["messages"] = messages

        try:
            response = await self._retry_request(self.client.messages.create, **params)

            content = response.content[0].text if response.content else ""
            usage = (
                {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens
                    + response.usage.output_tokens,
                }
                if response.usage
                else None
            )

            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                finish_reason=response.stop_reason,
                metadata={"response_id": response.id},
            )

        except Exception as e:
            self.logger.error(f"Error generating with Anthropic: {e}")
            raise

    async def stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate text as a stream."""
        if not self.client:
            raise ValueError("Anthropic client not initialized")

        # Build messages for Claude
        messages = [{"role": "user", "content": prompt}]

        # Build request parameters
        params = self._build_anthropic_params(**kwargs)
        params["messages"] = messages
        params["stream"] = True

        try:
            async with self.client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            self.logger.error(f"Error streaming with Anthropic: {e}")
            raise

    def _build_anthropic_params(self, **kwargs) -> dict[str, Any]:
        """Build Anthropic-specific request parameters."""
        params = {
            "model": self.config.model,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }

        # Handle system message
        if "system_message" in kwargs:
            params["system"] = kwargs["system_message"]

        # Handle stop sequences
        if "stop" in kwargs:
            params["stop_sequences"] = kwargs["stop"]

        return params
