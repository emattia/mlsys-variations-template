"""Integration tests for AI CLI commands."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from click.testing import CliRunner

from src.cli.ai import cli


class TestAICliIntegration:
    """Test AI CLI command integration."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "ai_config.yaml"
            config_content = """
ai:
  agents:
    default_agent:
      type: react
      llm_provider: openai
      model: gpt-3.5-turbo
      temperature: 0.7

  rag:
    chunk_size: 1000
    chunk_overlap: 200
    vector_store_type: memory
    top_k: 4

  llm_providers:
    openai:
      api_key: test-key
      model: gpt-3.5-turbo
"""
            config_path.write_text(config_content)
            yield temp_dir

    def test_cli_agent_run_command(self, runner):
        """Test agent run CLI command."""
        with patch("src.cli.ai.Agent") as mock_agent_class:
            # Setup mock agent
            mock_agent = AsyncMock()
            mock_agent.run.return_value = "Test agent response"
            mock_agent_class.create.return_value = mock_agent

            with patch("src.cli.ai.AgentMonitor") as mock_monitor:
                mock_monitor.return_value = Mock()

                result = runner.invoke(
                    cli,
                    [
                        "agent",
                        "run",
                        "--task",
                        "Test task",
                        "--agent-type",
                        "react",
                        "--model",
                        "gpt-3.5-turbo",
                    ],
                )

                assert result.exit_code == 0
                assert "Test agent response" in result.output

    def test_cli_agent_run_with_tools(self, runner):
        """Test agent run with tools."""
        with patch("src.cli.ai.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = "Agent used calculator tool"
            mock_agent_class.create.return_value = mock_agent

            with patch("src.cli.ai.AgentMonitor") as mock_monitor:
                mock_monitor.return_value = Mock()

                result = runner.invoke(
                    cli,
                    [
                        "agent",
                        "run",
                        "--task",
                        "Calculate 2+2",
                        "--tools",
                        "calculator,web_search",
                    ],
                )

                assert result.exit_code == 0
                # Should have registered tools
                mock_agent.add_tool.assert_called()

    def test_cli_rag_ingest_command(self, runner):
        """Test RAG ingest CLI command."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test document content for RAG ingestion.")
            temp_file = f.name

        try:
            with patch("src.cli.ai.RAGPipeline") as mock_rag_class:
                mock_rag = AsyncMock()
                mock_rag_class.return_value = mock_rag

                with patch("src.cli.ai.LLMProvider") as mock_llm:
                    mock_llm.create.return_value = Mock()

                    result = runner.invoke(
                        cli,
                        [
                            "rag",
                            "ingest",
                            "--documents",
                            temp_file,
                            "--chunk-size",
                            "500",
                        ],
                    )

                    assert result.exit_code == 0
                    assert "ingested successfully" in result.output.lower()
                    mock_rag.ingest_documents.assert_called_once()

        finally:
            os.unlink(temp_file)

    def test_cli_rag_query_command(self, runner):
        """Test RAG query CLI command."""
        with patch("src.cli.ai.RAGPipeline") as mock_rag_class:
            mock_rag = AsyncMock()
            mock_rag.query.return_value = "RAG response to query"
            mock_rag_class.return_value = mock_rag

            with patch("src.cli.ai.LLMProvider") as mock_llm:
                mock_llm.create.return_value = Mock()

                result = runner.invoke(
                    cli,
                    [
                        "rag",
                        "query",
                        "--question",
                        "What is machine learning?",
                        "--top-k",
                        "3",
                    ],
                )

                assert result.exit_code == 0
                assert "RAG response to query" in result.output
                mock_rag.query.assert_called_once_with("What is machine learning?")

    def test_cli_prompt_validate_command(self, runner):
        """Test prompt validation CLI command."""
        with patch("src.cli.ai.PromptValidator") as mock_validator_class:
            mock_validator = Mock()
            mock_result = Mock()
            mock_result.is_valid = True
            mock_result.quality_score = 0.85
            mock_result.issues = []
            mock_result.recommendations = ["Use more specific examples"]
            mock_validator.validate_prompt.return_value = mock_result
            mock_validator_class.return_value = mock_validator

            result = runner.invoke(
                cli,
                [
                    "prompt",
                    "validate",
                    "--text",
                    "Please analyze the data and provide insights",
                ],
            )

            assert result.exit_code == 0
            assert "Valid: True" in result.output
            assert "Quality Score: 0.85" in result.output

    def test_cli_prompt_optimize_command(self, runner):
        """Test prompt optimization CLI command."""
        with patch("src.cli.ai.PromptOptimizer") as mock_optimizer_class:
            mock_optimizer = AsyncMock()
            mock_optimizer.optimize_prompt.return_value = {
                "optimized_prompt": "Optimized version of the prompt",
                "improvement_score": 0.2,
                "changes_made": ["Added clarity instructions", "Simplified language"],
            }
            mock_optimizer_class.return_value = mock_optimizer

            with patch("src.cli.ai.LLMProvider") as mock_llm:
                mock_llm.create.return_value = Mock()

                result = runner.invoke(
                    cli,
                    [
                        "prompt",
                        "optimize",
                        "--text",
                        "Original prompt text",
                        "--target-score",
                        "0.8",
                    ],
                )

                assert result.exit_code == 0
                assert "Optimized version of the prompt" in result.output

    def test_cli_monitor_stats_command(self, runner):
        """Test monitoring stats CLI command."""
        with patch("src.cli.ai.AgentMonitor") as mock_monitor_class:
            mock_monitor = Mock()
            mock_stats = {
                "events": {"total_events": 10},
                "cost_tracking": {"total_cost": 0.05, "total_calls": 5},
                "performance_tracking": {"total_executions": 8, "success_rate": 0.9},
                "summary": {"active_agents": 2},
            }
            mock_monitor.get_stats.return_value = mock_stats
            mock_monitor_class.return_value = mock_monitor

            result = runner.invoke(cli, ["monitor", "stats"])

            assert result.exit_code == 0
            assert "Total Events: 10" in result.output
            assert "Total Cost: $0.05" in result.output
            assert "Success Rate: 90.0%" in result.output

    def test_cli_monitor_export_command(self, runner):
        """Test monitoring export CLI command."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            export_file = f.name

        try:
            with patch("src.cli.ai.AgentMonitor") as mock_monitor_class:
                mock_monitor = Mock()
                mock_events = [
                    {"event_type": "agent_start", "agent_name": "test_agent"},
                    {"event_type": "agent_complete", "agent_name": "test_agent"},
                ]
                mock_monitor.export_events.return_value = mock_events
                mock_monitor_class.return_value = mock_monitor

                result = runner.invoke(
                    cli,
                    ["monitor", "export", "--output", export_file, "--format", "json"],
                )

                assert result.exit_code == 0
                assert f"exported to {export_file}" in result.output.lower()

        finally:
            if os.path.exists(export_file):
                os.unlink(export_file)

    def test_cli_config_validation(self, runner, temp_config_dir):
        """Test CLI configuration validation."""
        # Test with invalid configuration
        with patch("src.cli.ai.load_hydra_config") as mock_load_config:
            mock_load_config.side_effect = Exception("Invalid config")

            result = runner.invoke(cli, ["agent", "run", "--task", "Test task"])

            # Should handle config error gracefully
            assert result.exit_code != 0
            assert "configuration" in result.output.lower()

    def test_cli_error_handling(self, runner):
        """Test CLI error handling."""
        with patch("src.cli.ai.Agent") as mock_agent_class:
            # Setup agent that raises an error
            mock_agent_class.create.side_effect = Exception("Agent creation failed")

            result = runner.invoke(cli, ["agent", "run", "--task", "Test task"])

            assert result.exit_code != 0
            assert "error" in result.output.lower()

    def test_cli_verbose_output(self, runner):
        """Test CLI verbose output."""
        with patch("src.cli.ai.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = "Test response"
            mock_agent_class.create.return_value = mock_agent

            with patch("src.cli.ai.AgentMonitor") as mock_monitor:
                mock_monitor.return_value = Mock()

                result = runner.invoke(
                    cli, ["agent", "run", "--task", "Test task", "--verbose"]
                )

                assert result.exit_code == 0
                # Verbose mode should show additional information
                assert (
                    "agent configuration" in result.output.lower()
                    or "creating agent" in result.output.lower()
                )

    def test_cli_list_templates_command(self, runner):
        """Test listing prompt templates CLI command."""
        with patch("src.cli.ai.PromptLibrary") as mock_library:
            mock_library.list_templates.return_value = [
                "summarization",
                "question_answering",
                "code_generation",
            ]

            result = runner.invoke(cli, ["prompt", "list-templates"])

            assert result.exit_code == 0
            assert "summarization" in result.output
            assert "question_answering" in result.output
            assert "code_generation" in result.output

    def test_cli_get_template_command(self, runner):
        """Test getting specific prompt template CLI command."""
        with patch("src.cli.ai.PromptLibrary") as mock_library:
            mock_template = Mock()
            mock_template.name = "summarization"
            mock_template.template = "Summarize the following text: {text}"
            mock_template.variables = ["text"]
            mock_library.get_template.return_value = mock_template

            result = runner.invoke(
                cli, ["prompt", "get-template", "--name", "summarization"]
            )

            assert result.exit_code == 0
            assert "summarization" in result.output
            assert "Summarize the following text" in result.output


class TestCliAsyncIntegration:
    """Test CLI async operations integration."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.mark.asyncio
    async def test_async_agent_operations(self, runner):
        """Test that async agent operations work correctly in CLI."""
        with patch("src.cli.ai.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = "Async agent response"
            mock_agent_class.create.return_value = mock_agent

            with patch("src.cli.ai.AgentMonitor") as mock_monitor:
                mock_monitor.return_value = Mock()

                # The CLI should handle async operations properly
                result = runner.invoke(
                    cli, ["agent", "run", "--task", "Async test task"]
                )

                assert result.exit_code == 0
                assert "Async agent response" in result.output

    @pytest.mark.asyncio
    async def test_async_rag_operations(self, runner):
        """Test that async RAG operations work correctly in CLI."""
        with patch("src.cli.ai.RAGPipeline") as mock_rag_class:
            mock_rag = AsyncMock()
            mock_rag.query.return_value = "Async RAG response"
            mock_rag_class.return_value = mock_rag

            with patch("src.cli.ai.LLMProvider") as mock_llm:
                mock_llm.create.return_value = Mock()

                result = runner.invoke(
                    cli, ["rag", "query", "--question", "Async RAG test question"]
                )

                assert result.exit_code == 0
                assert "Async RAG response" in result.output


class TestCliConfigIntegration:
    """Test CLI configuration integration."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_config_loading_from_file(self, runner):
        """Test loading configuration from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
ai:
  agents:
    default:
      type: react
      llm_provider: openai
      model: gpt-4
      temperature: 0.3
""")
            config_file = f.name

        try:
            with patch("src.cli.ai.Agent") as mock_agent_class:
                mock_agent = AsyncMock()
                mock_agent.run.return_value = "Config test response"
                mock_agent_class.create.return_value = mock_agent

                with patch("src.cli.ai.AgentMonitor") as mock_monitor:
                    mock_monitor.return_value = Mock()

                    result = runner.invoke(
                        cli,
                        [
                            "agent",
                            "run",
                            "--task",
                            "Test with config",
                            "--config",
                            config_file,
                        ],
                    )

                    assert result.exit_code == 0
                    # Should use config from file
                    # Verify agent was created with correct config
                    call_args = mock_agent_class.create.call_args[0][0]
                    assert call_args.temperature == 0.3
                    assert call_args.model == "gpt-4"

        finally:
            os.unlink(config_file)

    def test_cli_environment_variable_config(self, runner):
        """Test CLI respects environment variables."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-test-key"}):
            with patch("src.cli.ai.Agent") as mock_agent_class:
                mock_agent = AsyncMock()
                mock_agent.run.return_value = "Env config test"
                mock_agent_class.create.return_value = mock_agent

                with patch("src.cli.ai.AgentMonitor") as mock_monitor:
                    mock_monitor.return_value = Mock()

                    result = runner.invoke(
                        cli, ["agent", "run", "--task", "Test with env vars"]
                    )

                    # Should not fail due to missing API key
                    assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__])
