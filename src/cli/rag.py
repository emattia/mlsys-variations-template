"""
RAG (Retrieval-Augmented Generation) functionality for MLOps CLI.

This module provides commands for managing RAG pipelines including:
- Vector store setup and management
- Document ingestion and indexing
- Query processing and retrieval
"""

import textwrap
from pathlib import Path

import typer
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.platform.config import ConfigManager
from src.platform.plugins import ExecutionContext, get_plugin

app = typer.Typer()


def get_context() -> ExecutionContext:
    """Helper to create an execution context."""
    config_manager = ConfigManager()
    config = config_manager.get_config()
    return ExecutionContext(run_id="rag_cli", config=config)


@app.command()
def build_index(
    data_path: Path = typer.Option(
        ...,
        "--path",
        "-p",
        help="Path to a file or directory containing documents to index.",
    ),
    glob_pattern: str = typer.Option(
        "**/*.md", "--glob", "-g", help="Glob pattern to find files in the directory."
    ),
):
    """Builds a vector store index from local documents."""
    typer.echo(f"Building index from documents in: {data_path}")
    context = get_context()

    # Load documents
    if data_path.is_dir():
        loader = DirectoryLoader(
            str(data_path), glob=glob_pattern, loader_cls=TextLoader
        )
    else:
        loader = TextLoader(str(data_path))

    documents = loader.load()
    if not documents:
        typer.echo("No documents found to index.")
        return

    typer.echo(f"Loaded {len(documents)} document(s).")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=context.config.rag.chunk_size,
        chunk_overlap=context.config.rag.chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)
    typer.echo(f"Split documents into {len(chunks)} chunks.")

    # Initialize vector store plugin
    vector_store_plugin = get_plugin(
        name=context.config.vector_store.plugin,
        category="vector_store",
        config=context.config.vector_store.config,
    )
    vector_store_plugin.initialize(context)

    # Add documents to vector store
    vector_store_plugin.add_documents(chunks, context)

    typer.echo("✅ Index built successfully.")


@app.command()
def query(
    query_text: str = typer.Argument(..., help="The query to ask the RAG pipeline."),
    k: int = typer.Option(4, "--top-k", "-k", help="Number of documents to retrieve."),
):
    """Queries the RAG pipeline to get an answer."""
    typer.echo(f"Querying with: '{query_text}'")
    context = get_context()

    # Initialize plugins
    vector_store_plugin = get_plugin(
        name=context.config.vector_store.plugin,
        category="vector_store",
        config=context.config.vector_store.config,
    )
    vector_store_plugin.initialize(context)

    llm_provider_plugin = get_plugin(
        name=context.config.llm_provider.plugin,
        category="llm_provider",
        config=context.config.llm_provider.config,
    )
    llm_provider_plugin.initialize(context)

    # Retrieve documents
    typer.echo(f"Retrieving {k} relevant documents...")
    retrieved_docs = vector_store_plugin.similarity_search(query_text, k, context)

    if not retrieved_docs:
        typer.echo("Could not retrieve any relevant documents. Cannot answer query.")
        return

    # Build prompt
    context_str = "\\n\\n---\\n\\n".join(
        [
            f"Source {i + 1}:\\n{doc.page_content}"
            for i, doc in enumerate(retrieved_docs)
        ]
    )

    prompt = textwrap.dedent(f"""
        You are an expert Q&A assistant. Your goal is to provide helpful and accurate answers based on the context provided.

        **Context:**
        {context_str}

        ---

        **Question:**
        {query_text}

        **Answer:**
    """)

    # Generate answer
    typer.echo("Generating answer...")
    answer = llm_provider_plugin.generate(prompt, context)

    typer.echo("\\n" + "=" * 20)
    typer.echo("📝 Answer:")
    typer.echo(f"   {answer}")
    typer.echo("=" * 20 + "\\n")
    typer.echo("📚 Sources:")
    for i, doc in enumerate(retrieved_docs):
        typer.echo(f"  [{i + 1}] Source: {doc.metadata.get('source', 'N/A')}")
    typer.echo("\\n")


if __name__ == "__main__":
    app()
