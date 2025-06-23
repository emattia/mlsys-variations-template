"""Complete RAG pipeline implementation."""

import logging

from ..llm import LLMProvider
from .base import Document, RAGConfig, RAGSystem, RetrievalMethod
from .chunking import DocumentChunker
from .retrievers import HybridRetriever, VectorRetriever


class RAGPipeline(RAGSystem):
    """
    Complete RAG pipeline that integrates chunking, retrieval, and generation.

    This is a minimal but functional implementation that can be extended
    with more sophisticated features like reranking, query expansion, etc.
    """

    def __init__(self, config: RAGConfig, llm_provider: LLMProvider = None):
        self.config = config
        self.chunker = DocumentChunker(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )

        # Initialize retriever based on config
        if config.retrieval_method == RetrievalMethod.HYBRID:
            self.retriever = HybridRetriever(config)
        else:
            self.retriever = VectorRetriever(config)

        self.llm = llm_provider
        self.logger = logging.getLogger(__name__)

    async def ingest_documents(self, documents: list[Document]) -> None:
        """Ingest documents by chunking and storing them."""
        self.logger.info(f"Ingesting {len(documents)} documents")

        # Chunk documents
        chunked_docs = []
        for doc in documents:
            chunks = await self.chunker.chunk_document(doc)
            chunked_docs.extend(chunks)

        self.logger.info(f"Created {len(chunked_docs)} chunks")

        # Store in retriever
        await self.retriever.add_documents(chunked_docs)

        self.logger.info("Document ingestion completed")

    async def query(self, question: str, context: dict | None = None) -> str:
        """Query the RAG system for an answer."""
        if not self.llm:
            raise ValueError("LLM provider required for query generation")

        # Retrieve relevant documents
        retrieved_docs = await self.retrieve(question)

        if not retrieved_docs:
            return "I couldn't find relevant information to answer your question."

        # Build context from retrieved documents
        context_text = "\n\n".join(
            [f"Document {i + 1}: {doc.content}" for i, doc in enumerate(retrieved_docs)]
        )

        # Generate prompt
        prompt = self._build_prompt(question, context_text, context)

        # Generate answer
        answer = await self.llm.generate(prompt)

        # Log query for monitoring
        self._log_query(question, retrieved_docs, answer)

        return answer

    async def retrieve(self, query: str, k: int = None) -> list[Document]:
        """Retrieve relevant documents for a query."""
        k = k or self.config.top_k
        return await self.retriever.similarity_search(query, k)

    def _build_prompt(
        self, question: str, context: str, metadata: dict | None = None
    ) -> str:
        """Build the prompt for the LLM."""
        prompt = f"""
You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer the question.
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

        return prompt.strip()

    def _log_query(self, question: str, retrieved_docs: list[Document], answer: str):
        """Log query details for monitoring and analytics."""
        self.logger.info(f"RAG Query: {question[:100]}...")
        self.logger.info(f"Retrieved {len(retrieved_docs)} documents")
        self.logger.info(f"Answer length: {len(answer)} characters")

        # TODO: Send to monitoring system
        # This could integrate with the monitoring module

    async def evaluate_retrieval(
        self, queries: list[str], ground_truth: list[list[str]]
    ) -> dict[str, float]:
        """Evaluate retrieval performance with ground truth."""
        if len(queries) != len(ground_truth):
            raise ValueError("Queries and ground truth must have same length")

        total_precision = 0
        total_recall = 0

        for query, relevant_docs in zip(queries, ground_truth, strict=False):
            retrieved = await self.retrieve(query)
            retrieved_ids = [doc.id for doc in retrieved if doc.id]

            # Calculate precision and recall
            relevant_set = set(relevant_docs)
            retrieved_set = set(retrieved_ids)

            if retrieved_set:
                precision = len(relevant_set & retrieved_set) / len(retrieved_set)
            else:
                precision = 0

            if relevant_set:
                recall = len(relevant_set & retrieved_set) / len(relevant_set)
            else:
                recall = 0

            total_precision += precision
            total_recall += recall

        avg_precision = total_precision / len(queries)
        avg_recall = total_recall / len(queries)

        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
            if (avg_precision + avg_recall) > 0
            else 0,
        }
