"""Concrete implementations of VectorStore plugins."""

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain.schema.document import Document

from .base import ExecutionContext, VectorStore
from .registry import register_plugin


@register_plugin(
    name="chroma",
    category="vector_store",
    description="A vector store that uses ChromaDB.",
)
class ChromaVectorStore(VectorStore):
    """A vector store that uses ChromaDB for storage and retrieval."""

    def initialize(self, context: ExecutionContext) -> None:
        """Initializes the ChromaDB client and collection."""
        self.path = self.config.get("path", "data/chroma")
        self.collection_name = self.config.get("collection_name", "default_collection")
        self.embedding_model_name = self.config.get(
            "embedding_model", "all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(path=str(Path(self.path).resolve()))
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )
        self.logger.info(
            f"ChromaVectorStore initialized at path '{self.path}' "
            f"with collection '{self.collection_name}'."
        )

    def add_documents(
        self, documents: list[Document], context: ExecutionContext
    ) -> None:
        """Adds documents to the ChromaDB collection."""
        if not documents:
            return

        ids = [f"doc_{i}" for i in range(len(documents))]
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        try:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )
            self.logger.info(f"Added {len(documents)} documents to the collection.")
        except Exception as e:
            self.logger.error(f"Error adding documents to ChromaDB: {e}")
            raise

    def similarity_search(
        self, query: str, k: int, context: ExecutionContext
    ) -> list[Document]:
        """Performs a similarity search in the ChromaDB collection."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
            )

            if not results or not results.get("documents"):
                return []

            documents = []
            for i, text in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] or {}
                documents.append(Document(page_content=text, metadata=metadata))

            return documents
        except Exception as e:
            self.logger.error(f"Error querying ChromaDB: {e}")
            raise
