"""OpenAI LLM provider implementation."""

from collections.abc import AsyncGenerator
from typing import Any

from .base import LLMConfig, LLMProvider, LLMResponse

# Try to import openai at module level for testing
try:
    import openai
except ImportError:
    # Create a mock module for testing when openai is not installed
    import types

    openai = types.ModuleType("openai")
    openai.AsyncOpenAI = None


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider supporting GPT models."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenAI client."""
        # Check if we have a real openai module, or if AsyncOpenAI is mocked for testing
        if not hasattr(openai, "__version__") and openai.AsyncOpenAI is None:
            self.logger.error(
                "OpenAI library not installed. Install with: pip install openai"
            )
            raise ImportError("OpenAI library not available")

        try:
            self.client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )

            self.logger.info(
                f"OpenAI client initialized for model: {self.config.model}"
            )

        except Exception as e:
            self.logger.error(f"Error initializing OpenAI client: {e}")
            raise

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API."""
        response = await self.generate_with_response(prompt, **kwargs)
        return response.content

    async def generate_with_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text and return detailed response."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")

        # Build messages for chat completion
        messages = [{"role": "user", "content": prompt}]

        # Handle system message if provided
        if "system_message" in kwargs:
            messages.insert(0, {"role": "system", "content": kwargs["system_message"]})

        # Build request parameters
        params = self._build_request_params(**kwargs)
        params["messages"] = messages

        try:
            response = await self._retry_request(
                self.client.chat.completions.create, **params
            )

            content = response.choices[0].message.content
            usage = (
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                if response.usage
                else None
            )

            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                finish_reason=response.choices[0].finish_reason,
                metadata={"response_id": response.id},
            )

        except Exception as e:
            self.logger.error(f"Error generating with OpenAI: {e}")
            raise

    async def stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate text as a stream."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")

        # Build messages for chat completion
        messages = [{"role": "user", "content": prompt}]

        # Handle system message if provided
        if "system_message" in kwargs:
            messages.insert(0, {"role": "system", "content": kwargs["system_message"]})

        # Build request parameters
        params = self._build_request_params(**kwargs)
        params["messages"] = messages
        params["stream"] = True

        try:
            stream = await self.client.chat.completions.create(**params)

            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            self.logger.error(f"Error streaming with OpenAI: {e}")
            raise

    async def generate_embeddings(
        self, texts: list[str], model: str = "text-embedding-ada-002"
    ) -> list[list[float]]:
        """Generate embeddings for texts."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")

        try:
            response = await self.client.embeddings.create(model=model, input=texts)

            return [embedding.embedding for embedding in response.data]

        except Exception as e:
            self.logger.error(f"Error generating embeddings with OpenAI: {e}")
            raise

    def _build_request_params(self, **kwargs) -> dict[str, Any]:
        """Build OpenAI-specific request parameters."""
        params = super()._build_request_params(**kwargs)

        # Add OpenAI-specific parameters
        params["frequency_penalty"] = kwargs.get(
            "frequency_penalty", self.config.frequency_penalty
        )
        params["presence_penalty"] = kwargs.get(
            "presence_penalty", self.config.presence_penalty
        )

        # Handle stop sequences
        if "stop" in kwargs:
            params["stop"] = kwargs["stop"]

        # Handle function calling if provided
        if "functions" in kwargs:
            params["functions"] = kwargs["functions"]
        if "function_call" in kwargs:
            params["function_call"] = kwargs["function_call"]

        return params
