"""Tests for LLM providers and token estimation."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.ai.llm import (
    AnthropicProvider,
    LLMConfig,
    LLMFactory,
    LLMProvider,
    LLMResponse,
    LocalLLMProvider,
    ModelType,
    OpenAIProvider,
    TokenEstimator,
)


class TestTokenEstimator:
    """Test TokenEstimator utility class."""

    def test_estimate_tokens(self):
        """Test token estimation from text."""
        text = "Hello, world! This is a test."
        tokens = TokenEstimator.estimate_tokens(text)

        # Should be roughly 1/4 of character count
        expected = len(text) // 4
        assert tokens == expected

    def test_estimate_cost_gpt4(self):
        """Test cost estimation for GPT-4."""
        cost = TokenEstimator.estimate_cost("gpt-4", 1000, 500)

        # GPT-4: $0.03 input, $0.06 output per 1K tokens
        expected = (1000 / 1000 * 0.03) + (500 / 1000 * 0.06)
        assert cost == expected

    def test_estimate_cost_unknown_model(self):
        """Test cost estimation for unknown model defaults to GPT-4."""
        cost = TokenEstimator.estimate_cost("unknown-model", 1000, 500)

        # Should default to GPT-4 pricing
        expected = (1000 / 1000 * 0.03) + (500 / 1000 * 0.06)
        assert cost == expected

    def test_get_model_info(self):
        """Test getting model information."""
        info = TokenEstimator.get_model_info("gpt-3.5-turbo")

        assert info["model"] == "gpt-3.5-turbo"
        assert "costs_per_1k_tokens" in info
        assert "estimated_tokens_per_char" in info


class TestLLMConfig:
    """Test LLMConfig dataclass."""

    def test_llm_config_creation(self):
        """Test basic LLM configuration creation."""
        config = LLMConfig(provider="openai", model="gpt-4", api_key="test-key")

        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.api_key == "test-key"
        assert config.temperature == 0.7  # default
        assert config.max_tokens == 2000  # default

    def test_llm_config_with_custom_values(self):
        """Test LLM config with custom values."""
        config = LLMConfig(
            provider="anthropic",
            model="claude-3-opus",
            temperature=0.3,
            max_tokens=4000,
            model_type=ModelType.CHAT,
        )

        assert config.temperature == 0.3
        assert config.max_tokens == 4000
        assert config.model_type == ModelType.CHAT


class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_llm_response_creation(self):
        """Test LLM response creation."""
        response = LLMResponse(
            content="Test response",
            model="gpt-4",
            usage={"total_tokens": 100},
            finish_reason="stop",
        )

        assert response.content == "Test response"
        assert response.model == "gpt-4"
        assert response.usage["total_tokens"] == 100
        assert response.finish_reason == "stop"


class TestOpenAIProvider:
    """Test OpenAI LLM provider."""

    @pytest.fixture
    def openai_config(self):
        """Create OpenAI configuration."""
        return LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="test-key",
            temperature=0.5,
        )

    def test_openai_provider_initialization(self, openai_config):
        """Test OpenAI provider initialization."""
        with patch("src.ai.llm.openai_provider.openai.AsyncOpenAI") as mock_openai:
            provider = OpenAIProvider(openai_config)

            assert provider.config == openai_config
            mock_openai.assert_called_once()

    @pytest.mark.asyncio
    async def test_openai_generate(self, openai_config):
        """Test OpenAI text generation."""
        with patch("src.ai.llm.openai_provider.openai.AsyncOpenAI") as mock_openai:
            # Setup mock client
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Generated response"
            mock_response.model = "gpt-3.5-turbo"
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 20
            mock_response.usage.total_tokens = 30
            mock_response.choices[0].finish_reason = "stop"
            mock_response.id = "test-id"

            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(openai_config)
            provider.client = mock_client

            result = await provider.generate("Test prompt")

            assert result == "Generated response"
            mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_openai_generate_with_response(self, openai_config):
        """Test OpenAI generation with detailed response."""
        with patch("src.ai.llm.openai_provider.openai.AsyncOpenAI") as mock_openai:
            # Setup mock client
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Generated response"
            mock_response.model = "gpt-3.5-turbo"
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 20
            mock_response.usage.total_tokens = 30
            mock_response.choices[0].finish_reason = "stop"
            mock_response.id = "test-id"

            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(openai_config)
            provider.client = mock_client

            result = await provider.generate_with_response("Test prompt")

            assert isinstance(result, LLMResponse)
            assert result.content == "Generated response"
            assert result.model == "gpt-3.5-turbo"
            assert result.usage["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_openai_stream_generate(self, openai_config):
        """Test OpenAI streaming generation."""
        with patch("src.ai.llm.openai_provider.openai.AsyncOpenAI") as mock_openai:
            # Setup mock streaming response
            mock_client = AsyncMock()

            async def mock_stream():
                chunks = ["Hello", " ", "world", "!"]
                for chunk_text in chunks:
                    chunk = Mock()
                    chunk.choices = [Mock()]
                    chunk.choices[0].delta.content = chunk_text
                    yield chunk

            mock_client.chat.completions.create.return_value = mock_stream()
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(openai_config)
            provider.client = mock_client

            result_chunks = []
            async for chunk in provider.stream_generate("Test prompt"):
                result_chunks.append(chunk)

            assert "".join(result_chunks) == "Hello world!"

    @pytest.mark.asyncio
    async def test_openai_generate_embeddings(self, openai_config):
        """Test OpenAI embeddings generation."""
        with patch("src.ai.llm.openai_provider.openai.AsyncOpenAI") as mock_openai:
            # Setup mock client
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.data = [Mock(), Mock()]
            mock_response.data[0].embedding = [0.1, 0.2, 0.3]
            mock_response.data[1].embedding = [0.4, 0.5, 0.6]

            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(openai_config)
            provider.client = mock_client

            embeddings = await provider.generate_embeddings(["text1", "text2"])

            assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


class TestAnthropicProvider:
    """Test Anthropic LLM provider."""

    @pytest.fixture
    def anthropic_config(self):
        """Create Anthropic configuration."""
        return LLMConfig(
            provider="anthropic", model="claude-3-sonnet", api_key="test-key"
        )

    def test_anthropic_provider_initialization(self, anthropic_config):
        """Test Anthropic provider initialization."""
        with patch(
            "src.ai.llm.anthropic_provider.anthropic.AsyncAnthropic"
        ) as mock_anthropic:
            provider = AnthropicProvider(anthropic_config)

            assert provider.config == anthropic_config
            mock_anthropic.assert_called_once()

    @pytest.mark.asyncio
    async def test_anthropic_generate(self, anthropic_config):
        """Test Anthropic text generation."""
        with patch(
            "src.ai.llm.anthropic_provider.anthropic.AsyncAnthropic"
        ) as mock_anthropic:
            # Setup mock client
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = "Generated response"
            mock_response.model = "claude-3-sonnet"
            mock_response.usage = Mock()
            mock_response.usage.input_tokens = 10
            mock_response.usage.output_tokens = 20
            mock_response.stop_reason = "end_turn"
            mock_response.id = "test-id"

            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            provider = AnthropicProvider(anthropic_config)
            provider.client = mock_client

            result = await provider.generate("Test prompt")

            assert result == "Generated response"


class TestLocalLLMProvider:
    """Test Local LLM provider."""

    @pytest.fixture
    def local_config(self):
        """Create local LLM configuration."""
        return LLMConfig(
            provider="local", model="llama2", base_url="http://localhost:8080"
        )

    def test_local_provider_initialization(self, local_config):
        """Test local provider initialization."""
        provider = LocalLLMProvider(local_config)

        assert provider.config == local_config

    @pytest.mark.asyncio
    async def test_local_generate(self, local_config):
        """Test local provider generation (placeholder)."""
        provider = LocalLLMProvider(local_config)

        result = await provider.generate("Test prompt")

        # Should return placeholder response
        assert "Local model response" in result

    @pytest.mark.asyncio
    async def test_local_generate_with_response(self, local_config):
        """Test local provider with detailed response."""
        provider = LocalLLMProvider(local_config)

        result = await provider.generate_with_response("Test prompt")

        assert isinstance(result, LLMResponse)
        assert result.model == local_config.model
        assert result.usage is not None


class TestLLMFactory:
    """Test LLM factory for creating providers."""

    def test_create_openai_provider(self):
        """Test creating OpenAI provider via factory."""
        with patch("src.ai.llm.openai_provider.openai.AsyncOpenAI"):
            provider = LLMFactory.create(
                "openai", {"model": "gpt-4", "api_key": "test-key"}
            )

            assert isinstance(provider, OpenAIProvider)
            assert provider.config.provider == "openai"
            assert provider.config.model == "gpt-4"

    def test_create_anthropic_provider(self):
        """Test creating Anthropic provider via factory."""
        with patch("src.ai.llm.anthropic_provider.anthropic.AsyncAnthropic"):
            provider = LLMFactory.create(
                "anthropic", {"model": "claude-3-opus", "api_key": "test-key"}
            )

            assert isinstance(provider, AnthropicProvider)
            assert provider.config.provider == "anthropic"

    def test_create_local_provider(self):
        """Test creating local provider via factory."""
        provider = LLMFactory.create("local", {"model": "llama2"})

        assert isinstance(provider, LocalLLMProvider)
        assert provider.config.provider == "local"

    def test_create_unknown_provider(self):
        """Test creating unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMFactory.create("unknown", {"model": "test"})

    def test_get_supported_providers(self):
        """Test getting supported providers information."""
        providers = LLMFactory.get_supported_providers()

        assert "openai" in providers
        assert "anthropic" in providers
        assert "local" in providers

        # Check structure
        assert "description" in providers["openai"]
        assert "models" in providers["openai"]
        assert "supports_streaming" in providers["openai"]

    def test_validate_config_success(self):
        """Test successful config validation."""
        config = {"model": "gpt-4", "api_key": "test-key"}

        result = LLMFactory.validate_config("openai", config)
        assert result is True

    def test_validate_config_missing_field(self):
        """Test config validation with missing required field."""
        config = {
            "model": "gpt-4"
            # Missing api_key
        }

        with pytest.raises(ValueError, match="Missing required field 'api_key'"):
            LLMFactory.validate_config("openai", config)


class TestLLMProviderBase:
    """Test base LLM provider functionality."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock provider for testing base functionality."""
        config = LLMConfig(
            provider="test", model="test-model", temperature=0.5, max_tokens=1000
        )

        # Create a concrete implementation of the abstract base class
        class TestProvider(LLMProvider):
            async def generate(self, prompt: str, **kwargs) -> str:
                return "test response"

            async def generate_with_response(
                self, prompt: str, **kwargs
            ) -> LLMResponse:
                return LLMResponse("test response", "test-model")

            async def stream_generate(self, prompt: str, **kwargs):
                yield "test"
                yield " response"

        return TestProvider(config)

    @pytest.mark.asyncio
    async def test_batch_generate(self, mock_provider):
        """Test batch generation functionality."""
        prompts = ["prompt 1", "prompt 2", "prompt 3"]

        results = await mock_provider.batch_generate(prompts)

        assert len(results) == 3
        assert all(result == "test response" for result in results)

    @pytest.mark.asyncio
    async def test_estimate_cost(self, mock_provider):
        """Test cost estimation functionality."""
        prompt = "This is a test prompt for cost estimation"

        cost_info = await mock_provider.estimate_cost(prompt, max_output_tokens=500)

        assert "estimated_input_tokens" in cost_info
        assert "estimated_output_tokens" in cost_info
        assert "estimated_total_tokens" in cost_info
        assert "estimated_cost_usd" in cost_info
        assert cost_info["model"] == "test-model"

    def test_build_request_params(self, mock_provider):
        """Test building request parameters."""
        params = mock_provider._build_request_params(temperature=0.8, max_tokens=1500)

        assert params["model"] == "test-model"
        assert params["temperature"] == 0.8  # Override
        assert params["max_tokens"] == 1500  # Override


if __name__ == "__main__":
    pytest.main([__file__])
