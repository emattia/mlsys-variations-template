"""Base classes for RAG system components."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class RetrievalMethod(Enum):
    """Types of retrieval methods."""

    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass
class Document:
    """Represents a document with content and metadata."""

    content: str
    metadata: dict[str, Any]
    id: str | None = None
    score: float | None = None


@dataclass
class RAGConfig:
    """Configuration for RAG system."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_method: RetrievalMethod = RetrievalMethod.VECTOR
    top_k: int = 4
    similarity_threshold: float = 0.0
    vector_store_type: str = "chroma"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str | None = None


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store."""
        pass

    @abstractmethod
    async def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Perform similarity search."""
        pass

    @abstractmethod
    async def delete_documents(self, document_ids: list[str]) -> None:
        """Delete documents from the vector store."""
        pass


class RAGSystem(ABC):
    """Abstract base class for RAG systems."""

    @abstractmethod
    async def ingest_documents(self, documents: list[Document]) -> None:
        """Ingest documents into the RAG system."""
        pass

    @abstractmethod
    async def query(self, question: str, context: dict | None = None) -> str:
        """Query the RAG system for an answer."""
        pass

    @abstractmethod
    async def retrieve(self, query: str, k: int = 4) -> list[Document]:
        """Retrieve relevant documents for a query."""
        pass
