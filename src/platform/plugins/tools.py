"""Tools and utilities for plugin development."""

from src.platform.plugins import ExecutionContext, get_plugin
from src.platform.plugins.base import MLOpsComponent
from src.platform.plugins.registry import register_plugin


class Tool(MLOpsComponent):
    """Base class for tools that can be used by an agent."""

    def run(self, args: dict) -> str:
        raise NotImplementedError


@register_plugin(name="rag_tool", category="tool")
class RAGTool(Tool):
    """A tool that allows an agent to query the RAG pipeline."""

    def initialize(self, context: ExecutionContext) -> None:
        self.context = context
        self.vector_store = get_plugin(
            name=context.config.vector_store.plugin,
            category="vector_store",
            config=context.config.vector_store.config,
        )
        self.vector_store.initialize(context)
        self.llm_provider = get_plugin(
            name=context.config.llm_provider.plugin,
            category="llm_provider",
            config=context.config.llm_provider.config,
        )
        self.llm_provider.initialize(context)

    def run(self, tool_input: str) -> str:
        """Queries the RAG pipeline to get an answer."""
        retrieved_docs = self.vector_store.similarity_search(
            tool_input, k=4, context=self.context
        )
        if not retrieved_docs:
            return "No relevant information found."

        context_str = "\\n\\n---\\n\\n".join(
            [doc.page_content for doc in retrieved_docs]
        )
        prompt = f"Based on the following context, answer the question.\\n\\nContext:\\n{context_str}\\n\\nQuestion:\\n{tool_input}\\n\\nAnswer:"
        return self.llm_provider.generate(prompt, self.context)
