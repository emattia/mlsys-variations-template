"""AI agent CLI commands."""

import asyncio
from pathlib import Path

import typer

from src.ai import (
    Agent,
    AgentConfig,
    AgentMonitor,
    AgentType,
    LLMFactory,
    PerformanceAnalytics,
    PromptLibrary,
    PromptValidator,
    RAGConfig,
    RAGPipeline,
    ToolRegistry,
)

app = typer.Typer(help="AI agent management and execution")
cli = app  # Alias for tests


@app.command()
def run_agent(
    task: str = typer.Argument(..., help="Task for the agent to perform"),
    agent_type: str = typer.Option(
        "react", help="Type of agent: react, langraph, crewai"
    ),
    model: str = typer.Option("gpt-4", help="LLM model to use"),
    provider: str = typer.Option(
        "openai", help="LLM provider: openai, anthropic, local"
    ),
    temperature: float = typer.Option(0.7, help="Temperature for LLM"),
    max_tokens: int = typer.Option(2000, help="Maximum tokens for response"),
    verbose: bool = typer.Option(False, help="Verbose output"),
):
    """Run an AI agent to perform a task."""

    async def _run():
        # Create agent config
        agent_config = AgentConfig(
            name=f"{agent_type}_agent",
            agent_type=AgentType(agent_type),
            llm_provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
        )

        # Create monitor
        monitor = AgentMonitor()

        # Create agent
        agent = Agent.create(agent_config, monitor)

        # Add built-in tools
        tool_registry = ToolRegistry()
        tool_registry.register_builtin_tools()

        for tool_name, tool in tool_registry.get_all_tools().items():
            agent.add_tool(tool_name, tool)

        typer.echo(f"🤖 Running {agent_type} agent with {provider}/{model}")
        typer.echo(f"📝 Task: {task}")
        typer.echo("")

        try:
            result = await agent.run(task)

            typer.echo("✅ Agent completed successfully!")
            typer.echo("=" * 50)
            typer.echo(result)
            typer.echo("=" * 50)

        except Exception as e:
            typer.echo(f"❌ Agent failed: {str(e)}")
            raise typer.Exit(1) from e

    asyncio.run(_run())


@app.command()
def rag_query(
    question: str = typer.Argument(..., help="Question to ask the RAG system"),
    docs_path: Path = typer.Option("./docs", help="Path to documents directory"),
    chunk_size: int = typer.Option(1000, help="Chunk size for documents"),
    top_k: int = typer.Option(4, help="Number of documents to retrieve"),
    provider: str = typer.Option("openai", help="LLM provider"),
    model: str = typer.Option("gpt-4", help="LLM model"),
    vector_store: str = typer.Option(
        "chroma", help="Vector store type: chroma, faiss, memory"
    ),
):
    """Query a RAG system with documents."""

    async def _query():
        # Create RAG config
        rag_config = RAGConfig(
            chunk_size=chunk_size, top_k=top_k, vector_store_type=vector_store
        )

        # Create LLM provider
        llm_provider = LLMFactory.create(
            provider,
            {
                "model": model,
                "api_key": None,  # Should be set via environment
            },
        )

        # Create RAG pipeline
        rag_pipeline = RAGPipeline(rag_config, llm_provider)

        typer.echo(f"🔍 Querying RAG system: {question}")
        typer.echo(f"📚 Documents from: {docs_path}")
        typer.echo("")

        # Check if we need to ingest documents first
        if docs_path.exists() and docs_path.is_dir():
            from src.ai.rag.base import Document

            typer.echo("📥 Ingesting documents...")
            documents = []

            for file_path in docs_path.rglob("*.md"):
                try:
                    content = file_path.read_text(encoding="utf-8")
                    doc = Document(
                        content=content,
                        metadata={"source": str(file_path), "type": "markdown"},
                    )
                    documents.append(doc)
                except Exception as e:
                    typer.echo(f"⚠️  Error reading {file_path}: {e}")

            if documents:
                await rag_pipeline.ingest_documents(documents)
                typer.echo(f"✅ Ingested {len(documents)} documents")
            else:
                typer.echo("⚠️  No documents found to ingest")

        try:
            answer = await rag_pipeline.query(question)

            typer.echo("✅ RAG query completed!")
            typer.echo("=" * 50)
            typer.echo(answer)
            typer.echo("=" * 50)

        except Exception as e:
            typer.echo(f"❌ RAG query failed: {str(e)}")
            raise typer.Exit(1) from e

    asyncio.run(_query())


@app.command()
def validate_prompt(
    prompt: str = typer.Argument(..., help="Prompt to validate"),
):
    """Validate a prompt for quality and best practices."""
    validator = PromptValidator()
    result = validator.validate_prompt(prompt)

    typer.echo("🔍 Prompt Validation Results")
    typer.echo("=" * 40)
    typer.echo(f"✅ Valid: {result.is_valid}")
    typer.echo(f"📊 Quality Score: {result.quality_score:.2f}/1.0")
    typer.echo("")

    if result.issues:
        typer.echo("⚠️  Issues Found:")
        for issue in result.issues:
            severity_emoji = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}
            emoji = severity_emoji.get(issue.severity.value, "•")
            typer.echo(f"  {emoji} {issue.message}")
            if issue.suggestion:
                typer.echo(f"    💡 {issue.suggestion}")
        typer.echo("")

    if result.recommendations:
        typer.echo("🎯 Recommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            typer.echo(f"  {i}. {rec}")


@app.command()
def list_templates():
    """List available prompt templates."""
    templates = PromptLibrary.get_all_templates()

    typer.echo("📝 Available Prompt Templates")
    typer.echo("=" * 40)

    for name, template in templates.items():
        typer.echo(f"• {name}")
        typer.echo(f"  Type: {template.prompt_type.value}")
        if template.metadata.get("description"):
            typer.echo(f"  Description: {template.metadata['description']}")
        typer.echo("")


@app.command()
def monitor_stats(
    agent_name: str | None = typer.Option(None, help="Specific agent to analyze"),
    days: int = typer.Option(7, help="Number of days to analyze"),
    export: bool = typer.Option(False, help="Export detailed data"),
):
    """Show agent monitoring statistics."""
    monitor = AgentMonitor()
    analytics = PerformanceAnalytics(monitor)

    if export:
        # Export data
        data = monitor.export_data()
        export_path = Path("agent_monitoring_export.json")
        export_path.write_text(data)
        typer.echo(f"📊 Data exported to: {export_path}")
    else:
        # Show report
        report = analytics.generate_report(agent_name, days)
        typer.echo(report)


@app.command()
def list_tools():
    """List available agent tools."""
    registry = ToolRegistry()
    registry.register_builtin_tools()

    tools = registry.get_all_tools()

    typer.echo("🛠️  Available Agent Tools")
    typer.echo("=" * 40)

    for name, tool in tools.items():
        typer.echo(f"• {name}")
        typer.echo(f"  Description: {tool.description}")
        typer.echo(f"  Parameters: {len(tool.parameters)}")
        typer.echo("")


@app.command()
def test_tool(
    tool_name: str = typer.Argument(..., help="Name of tool to test"),
):
    """Test a specific agent tool."""

    async def _test():
        registry = ToolRegistry()
        registry.register_builtin_tools()

        tool = registry.get_tool(tool_name)
        if not tool:
            typer.echo(f"❌ Tool '{tool_name}' not found")
            available = registry.list_tools()
            typer.echo(f"Available tools: {', '.join(available)}")
            raise typer.Exit(1)

        typer.echo(f"🧪 Testing tool: {tool_name}")
        typer.echo(f"Description: {tool.description}")
        typer.echo("")

        # Show parameters
        typer.echo("Parameters:")
        for param in tool.parameters:
            required = "required" if param.required else "optional"
            typer.echo(
                f"  • {param.name} ({param.type}, {required}): {param.description}"
            )
        typer.echo("")

        # Test with sample parameters based on tool
        if tool_name == "calculator":
            result = await registry.execute_tool("calculator", expression="2 + 3 * 4")
        elif tool_name == "web_search":
            result = await registry.execute_tool(
                "web_search", query="AI agents", num_results=3
            )
        elif tool_name == "filesystem":
            result = await registry.execute_tool(
                "filesystem", operation="exists", path="./"
            )
        elif tool_name == "code_executor":
            result = await registry.execute_tool(
                "code_executor", code="print('Hello from AI agent!')"
            )
        else:
            typer.echo("⚠️  No test case defined for this tool")
            return

        typer.echo("🔍 Test Result:")
        typer.echo(f"Success: {result.success}")
        typer.echo(f"Result: {result.result}")
        if result.error:
            typer.echo(f"Error: {result.error}")
        if result.metadata:
            typer.echo(f"Metadata: {result.metadata}")

    asyncio.run(_test())


if __name__ == "__main__":
    app()
