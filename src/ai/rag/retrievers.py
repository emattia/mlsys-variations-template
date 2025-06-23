"""Document retrieval implementations for RAG systems."""

import logging
from typing import Any

from .base import Document, RAGConfig, VectorStore


class VectorRetriever:
    """Vector-based document retriever using embeddings."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store: VectorStore | None = None
        self.logger = logging.getLogger(__name__)
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize the vector store based on configuration."""
        if self.config.vector_store_type == "chroma":
            from .stores import ChromaVectorStore

            self.vector_store = ChromaVectorStore(self.config)
        elif self.config.vector_store_type == "faiss":
            from .stores import FAISSVectorStore

            self.vector_store = FAISSVectorStore(self.config)
        elif self.config.vector_store_type == "memory":
            from .stores import InMemoryVectorStore

            self.vector_store = InMemoryVectorStore(self.config)
        else:
            raise ValueError(
                f"Unknown vector store type: {self.config.vector_store_type}"
            )

    async def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")

        await self.vector_store.add_documents(documents)
        self.logger.info(f"Added {len(documents)} documents to vector store")

    async def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Perform similarity search using vector embeddings."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")

        results = await self.vector_store.similarity_search(query, k)

        # Filter by similarity threshold
        filtered_results = [
            doc
            for doc in results
            if doc.score is None or doc.score >= self.config.similarity_threshold
        ]

        return filtered_results[:k]


class KeywordRetriever:
    """Keyword-based document retriever using BM25 or similar."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.documents: list[Document] = []
        self.index: Any | None = None
        self.logger = logging.getLogger(__name__)

    async def add_documents(self, documents: list[Document]) -> None:
        """Add documents and build keyword index."""
        self.documents.extend(documents)
        await self._build_index()
        self.logger.info(f"Added {len(documents)} documents to keyword index")

    async def _build_index(self):
        """Build keyword search index (placeholder)."""
        # TODO: Implement BM25 or similar keyword search
        # This could use libraries like rank_bm25 or build a simple TF-IDF index
        pass

    async def keyword_search(self, query: str, k: int = 4) -> list[Document]:
        """Perform keyword-based search."""
        # Simple placeholder implementation
        # In practice, this would use BM25 or similar
        query_terms = query.lower().split()

        scored_docs = []
        for doc in self.documents:
            content_lower = doc.content.lower()
            score = sum(1 for term in query_terms if term in content_lower)

            if score > 0:
                doc_copy = Document(
                    content=doc.content,
                    metadata=doc.metadata.copy(),
                    id=doc.id,
                    score=score / len(query_terms),  # Normalized score
                )
                scored_docs.append(doc_copy)

        # Sort by score and return top k
        scored_docs.sort(key=lambda x: x.score, reverse=True)
        return scored_docs[:k]


class HybridRetriever:
    """Hybrid retriever combining vector and keyword search."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_retriever = VectorRetriever(config)
        self.keyword_retriever = KeywordRetriever(config)
        self.logger = logging.getLogger(__name__)

    async def add_documents(self, documents: list[Document]) -> None:
        """Add documents to both retrievers."""
        await self.vector_retriever.add_documents(documents)
        await self.keyword_retriever.add_documents(documents)
        self.logger.info(f"Added {len(documents)} documents to hybrid retriever")

    async def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Perform hybrid search combining vector and keyword results."""
        # Get results from both retrievers
        vector_results = await self.vector_retriever.similarity_search(query, k)
        keyword_results = await self.keyword_retriever.keyword_search(query, k)

        # Combine and rerank results
        combined_results = self._combine_results(vector_results, keyword_results, k)

        return combined_results

    def _combine_results(
        self, vector_results: list[Document], keyword_results: list[Document], k: int
    ) -> list[Document]:
        """Combine and rerank results from vector and keyword search."""
        # Simple combination strategy: weighted average of scores
        vector_weight = 0.7
        keyword_weight = 0.3

        # Create a dictionary to track documents and their combined scores
        doc_scores: dict[str, dict] = {}

        # Process vector results
        for i, doc in enumerate(vector_results):
            doc_id = doc.id or doc.content[:50]  # Use content prefix if no ID
            # Vector score based on ranking (higher rank = higher score)
            vector_score = (len(vector_results) - i) / len(vector_results)

            doc_scores[doc_id] = {
                "document": doc,
                "vector_score": vector_score,
                "keyword_score": 0,
                "combined_score": vector_weight * vector_score,
            }

        # Process keyword results
        for i, doc in enumerate(keyword_results):
            doc_id = doc.id or doc.content[:50]
            keyword_score = doc.score or (
                (len(keyword_results) - i) / len(keyword_results)
            )

            if doc_id in doc_scores:
                # Update existing document
                doc_scores[doc_id]["keyword_score"] = keyword_score
                doc_scores[doc_id]["combined_score"] = (
                    vector_weight * doc_scores[doc_id]["vector_score"]
                    + keyword_weight * keyword_score
                )
            else:
                # Add new document
                doc_scores[doc_id] = {
                    "document": doc,
                    "vector_score": 0,
                    "keyword_score": keyword_score,
                    "combined_score": keyword_weight * keyword_score,
                }

        # Sort by combined score and return top k
        ranked_docs = sorted(
            doc_scores.values(), key=lambda x: x["combined_score"], reverse=True
        )

        # Update document scores and return
        result_docs = []
        for item in ranked_docs[:k]:
            doc = item["document"]
            doc.score = item["combined_score"]
            result_docs.append(doc)

        return result_docs
