"""Concrete implementations of LLMProvider plugins."""

import openai

from .base import ExecutionContext, LLMProvider
from .registry import register_plugin


@register_plugin(
    name="openai",
    category="llm_provider",
    description="An LLM provider that uses the OpenAI API.",
)
class OpenAIProvider(LLMProvider):
    """An LLM provider that uses the OpenAI API."""

    def initialize(self, context: ExecutionContext) -> None:
        """Initializes the OpenAI client."""
        self.api_key = self.config.get("api_key")
        self.model = self.config.get("model", "gpt-3.5-turbo")
        if not self.api_key:
            raise ValueError(
                "'api_key' not specified in the configuration for OpenAIProvider."
            )
        self.client = openai.OpenAI(api_key=self.api_key)
        self.logger.info(f"OpenAIProvider initialized for model: {self.model}")

    def generate(self, prompt: str, context: ExecutionContext) -> str:
        """Generates a response from the OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {e}")
            raise
