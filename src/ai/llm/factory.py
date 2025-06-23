"""Factory for creating LLM providers."""

from typing import Any

from .anthropic_provider import AnthropicProvider
from .base import LLMConfig, LLMProvider
from .local_provider import LocalLLMProvider
from .openai_provider import OpenAIProvider


class LLMFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create(provider_name: str, config: dict[str, Any]) -> LLMProvider:
        """Create an LLM provider based on configuration."""

        # Create LLMConfig from dict, adding provider parameter
        config_with_provider = config.copy()
        config_with_provider["provider"] = provider_name

        llm_config = LLMConfig(**config_with_provider)

        # Create appropriate provider
        if provider_name == "openai":
            return OpenAIProvider(llm_config)
        elif provider_name == "anthropic":
            return AnthropicProvider(llm_config)
        elif provider_name == "local":
            return LocalLLMProvider(llm_config)
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

    @staticmethod
    def create_from_config(config: LLMConfig) -> LLMProvider:
        """Create an LLM provider from LLMConfig object."""
        provider_name = config.provider

        if provider_name == "openai":
            return OpenAIProvider(config)
        elif provider_name == "anthropic":
            return AnthropicProvider(config)
        elif provider_name == "local":
            return LocalLLMProvider(config)
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

    @staticmethod
    def get_supported_providers() -> dict[str, dict[str, Any]]:
        """Get information about supported providers."""
        return {
            "openai": {
                "description": "OpenAI GPT models",
                "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                "supports_streaming": True,
                "supports_functions": True,
            },
            "anthropic": {
                "description": "Anthropic Claude models",
                "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
                "supports_streaming": True,
                "supports_functions": False,
            },
            "local": {
                "description": "Local models (Ollama, Transformers, etc.)",
                "models": ["llama2", "codellama", "mistral", "custom"],
                "supports_streaming": True,
                "supports_functions": False,
            },
        }

    @staticmethod
    def validate_config(provider_name: str, config: dict[str, Any]) -> bool:
        """Validate configuration for a provider."""
        required_fields = ["model"]

        # Provider-specific validations
        if provider_name in ["openai", "anthropic"]:
            required_fields.append("api_key")

        for field in required_fields:
            if field not in config:
                raise ValueError(
                    f"Missing required field '{field}' for provider '{provider_name}'"
                )

        return True


# Convenience function
def create_llm_provider(provider_name: str, **config) -> LLMProvider:
    """Convenience function to create an LLM provider."""
    return LLMFactory.create(provider_name, config)
