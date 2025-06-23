"""Vector store implementations for RAG systems."""

import logging

from .base import Document, RAGConfig, VectorStore


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of vector store."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = None
        self.collection = None
        self.logger = logging.getLogger(__name__)
        self._initialize()

    def _initialize(self):
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            # Create persistent client
            self.client = chromadb.PersistentClient(path="./data/chroma_db")

            # Set up embedding function
            self.embedding_function = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.config.embedding_model
                )
            )

            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name="rag_documents", embedding_function=self.embedding_function
            )

            self.logger.info("ChromaDB initialized successfully")

        except ImportError:
            self.logger.error(
                "ChromaDB not installed. Install with: pip install chromadb"
            )
            raise
        except Exception as e:
            self.logger.error(f"Error initializing ChromaDB: {e}")
            raise

    async def add_documents(self, documents: list[Document]) -> None:
        """Add documents to ChromaDB collection."""
        if not self.collection:
            raise ValueError("ChromaDB collection not initialized")

        # Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []

        for i, doc in enumerate(documents):
            doc_id = doc.id or f"doc_{i}"
            ids.append(doc_id)
            texts.append(doc.content)
            metadatas.append(doc.metadata)

        try:
            # Add to collection (ChromaDB handles embedding generation)
            self.collection.add(documents=texts, metadatas=metadatas, ids=ids)

            self.logger.info(f"Added {len(documents)} documents to ChromaDB")

        except Exception as e:
            self.logger.error(f"Error adding documents to ChromaDB: {e}")
            raise

    async def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Perform similarity search using ChromaDB."""
        if not self.collection:
            raise ValueError("ChromaDB collection not initialized")

        try:
            # Query the collection
            results = self.collection.query(query_texts=[query], n_results=k)

            # Convert results to Document objects
            documents = []
            if results["documents"] and results["documents"][0]:
                for i, content in enumerate(results["documents"][0]):
                    metadata = (
                        results["metadatas"][0][i] if results["metadatas"][0] else {}
                    )
                    doc_id = results["ids"][0][i] if results["ids"][0] else None
                    distance = (
                        results["distances"][0][i] if results["distances"] else None
                    )

                    # Convert distance to similarity score (lower distance = higher similarity)
                    score = 1 - distance if distance is not None else None

                    doc = Document(
                        content=content, metadata=metadata, id=doc_id, score=score
                    )
                    documents.append(doc)

            return documents

        except Exception as e:
            self.logger.error(f"Error querying ChromaDB: {e}")
            raise

    async def delete_documents(self, document_ids: list[str]) -> None:
        """Delete documents from ChromaDB collection."""
        if not self.collection:
            raise ValueError("ChromaDB collection not initialized")

        try:
            self.collection.delete(ids=document_ids)
            self.logger.info(f"Deleted {len(document_ids)} documents from ChromaDB")
        except Exception as e:
            self.logger.error(f"Error deleting documents from ChromaDB: {e}")
            raise


class FAISSVectorStore(VectorStore):
    """FAISS implementation of vector store (placeholder)."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.index = None
        self.documents: list[Document] = []
        self.logger = logging.getLogger(__name__)
        self._initialize()

    def _initialize(self):
        """Initialize FAISS index."""
        try:
            import faiss

            # TODO: Initialize FAISS index based on embedding dimension
            # This is a placeholder implementation
            self.dimension = 384  # Default for all-MiniLM-L6-v2
            self.index = faiss.IndexFlatIP(
                self.dimension
            )  # Inner product for cosine similarity

            self.logger.info("FAISS index initialized")

        except ImportError:
            self.logger.error(
                "FAISS not installed. Install with: pip install faiss-cpu"
            )
            raise
        except Exception as e:
            self.logger.error(f"Error initializing FAISS: {e}")
            raise

    async def add_documents(self, documents: list[Document]) -> None:
        """Add documents to FAISS index."""
        # TODO: Implement FAISS document addition
        # This would involve:
        # 1. Generate embeddings for documents
        # 2. Add embeddings to FAISS index
        # 3. Store documents separately for retrieval

        self.documents.extend(documents)
        self.logger.info(
            f"Added {len(documents)} documents to FAISS store (placeholder)"
        )

    async def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Perform similarity search using FAISS."""
        # TODO: Implement FAISS similarity search
        # This would involve:
        # 1. Generate embedding for query
        # 2. Search FAISS index
        # 3. Return corresponding documents

        # Placeholder: return first k documents
        return self.documents[:k]

    async def delete_documents(self, document_ids: list[str]) -> None:
        """Delete documents from FAISS index."""
        # TODO: Implement FAISS document deletion
        # Note: FAISS doesn't support deletion directly, would need to rebuild index

        self.logger.info(
            f"Delete operation for {len(document_ids)} documents (placeholder)"
        )


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store for testing and development."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.documents: list[Document] = []
        self.embeddings: list[list[float]] = []
        self.logger = logging.getLogger(__name__)

    async def add_documents(self, documents: list[Document]) -> None:
        """Add documents to in-memory store."""
        # For simplicity, just store documents without actual embeddings
        self.documents.extend(documents)

        # Placeholder embeddings (in practice, would generate real embeddings)
        import random

        for _ in documents:
            embedding = [random.random() for _ in range(384)]  # Random 384-dim vector
            self.embeddings.append(embedding)

        self.logger.info(f"Added {len(documents)} documents to in-memory store")

    async def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Perform similarity search using simple text matching."""
        # Simple keyword matching for development/testing
        query_lower = query.lower()

        scored_docs = []
        for doc in self.documents:
            # Simple scoring based on keyword overlap
            content_lower = doc.content.lower()
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            overlap = len(query_words & content_words)

            if overlap > 0:
                doc_copy = Document(
                    content=doc.content,
                    metadata=doc.metadata.copy(),
                    id=doc.id,
                    score=overlap / len(query_words),
                )
                scored_docs.append(doc_copy)

        # Sort by score and return top k
        scored_docs.sort(key=lambda x: x.score, reverse=True)
        return scored_docs[:k]

    async def delete_documents(self, document_ids: list[str]) -> None:
        """Delete documents from in-memory store."""
        id_set = set(document_ids)

        # Remove documents and corresponding embeddings
        new_documents = []
        new_embeddings = []

        for i, doc in enumerate(self.documents):
            if doc.id not in id_set:
                new_documents.append(doc)
                if i < len(self.embeddings):
                    new_embeddings.append(self.embeddings[i])

        self.documents = new_documents
        self.embeddings = new_embeddings

        self.logger.info(f"Deleted {len(document_ids)} documents from in-memory store")
