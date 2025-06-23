"""Tests for RAG system components."""

from unittest.mock import AsyncMock, patch

import pytest

from src.ai.llm import LLMProvider
from src.ai.rag import (
    ChunkingStrategy,
    Document,
    DocumentChunker,
    HybridRetriever,
    RAGConfig,
    RAGPipeline,
    RetrievalMethod,
    VectorRetriever,
)


class TestDocument:
    """Test Document dataclass."""

    def test_document_creation(self):
        """Test basic document creation."""
        doc = Document(
            content="Test document content",
            metadata={"source": "test.txt", "topic": "testing"},
        )

        assert doc.content == "Test document content"
        assert doc.metadata["source"] == "test.txt"
        assert doc.metadata["topic"] == "testing"
        assert doc.id is None
        assert doc.score is None

    def test_document_with_id_and_score(self):
        """Test document with ID and score."""
        doc = Document(content="Test content", metadata={}, id="doc_123", score=0.85)

        assert doc.id == "doc_123"
        assert doc.score == 0.85


class TestRAGConfig:
    """Test RAG configuration."""

    def test_rag_config_defaults(self):
        """Test RAG config with default values."""
        config = RAGConfig()

        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.retrieval_method == RetrievalMethod.VECTOR
        assert config.top_k == 4
        assert config.similarity_threshold == 0.0
        assert config.vector_store_type == "chroma"

    def test_rag_config_custom_values(self):
        """Test RAG config with custom values."""
        config = RAGConfig(
            chunk_size=500,
            chunk_overlap=50,
            retrieval_method=RetrievalMethod.HYBRID,
            top_k=6,
            vector_store_type="faiss",
        )

        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.retrieval_method == RetrievalMethod.HYBRID
        assert config.top_k == 6
        assert config.vector_store_type == "faiss"


class TestDocumentChunker:
    """Test document chunking functionality."""

    @pytest.fixture
    def sample_document(self):
        """Create sample document for testing."""
        content = """
        This is the first paragraph of the document. It contains some information
        about machine learning and artificial intelligence.

        This is the second paragraph. It talks about neural networks and deep learning.
        The content is structured to test different chunking strategies.

        This is the third paragraph. It discusses applications of AI in various industries
        such as healthcare, finance, and autonomous vehicles.
        """

        return Document(
            content=content,
            metadata={"source": "test_doc.txt", "topic": "AI"},
            id="test_doc",
        )

    @pytest.mark.asyncio
    async def test_fixed_size_chunking(self, sample_document):
        """Test fixed-size chunking strategy."""
        chunker = DocumentChunker(
            chunk_size=100, chunk_overlap=20, strategy=ChunkingStrategy.FIXED_SIZE
        )

        chunks = await chunker.chunk_document(sample_document)

        assert len(chunks) > 1  # Should create multiple chunks

        # Check chunk properties
        for chunk in chunks:
            assert isinstance(chunk, Document)
            assert len(chunk.content) <= 100 + 20  # Size + overlap buffer
            assert chunk.metadata["parent_id"] == "test_doc"
            assert "chunk_id" in chunk.metadata

    @pytest.mark.asyncio
    async def test_sentence_chunking(self, sample_document):
        """Test sentence-based chunking strategy."""
        chunker = DocumentChunker(chunk_size=200, strategy=ChunkingStrategy.SENTENCE)

        chunks = await chunker.chunk_document(sample_document)

        assert len(chunks) > 0

        # Check that chunks respect sentence boundaries
        for chunk in chunks:
            assert chunk.metadata["chunking_method"] == "sentence"

    @pytest.mark.asyncio
    async def test_paragraph_chunking(self, sample_document):
        """Test paragraph-based chunking strategy."""
        chunker = DocumentChunker(chunk_size=300, strategy=ChunkingStrategy.PARAGRAPH)

        chunks = await chunker.chunk_document(sample_document)

        assert len(chunks) > 0

        for chunk in chunks:
            assert chunk.metadata["chunking_method"] == "paragraph"

    @pytest.mark.asyncio
    async def test_semantic_chunking_fallback(self, sample_document):
        """Test semantic chunking falls back to sentence chunking."""
        chunker = DocumentChunker(chunk_size=200, strategy=ChunkingStrategy.SEMANTIC)

        chunks = await chunker.chunk_document(sample_document)

        # Should fall back to sentence chunking
        assert len(chunks) > 0


class TestVectorRetriever:
    """Test vector-based retrieval."""

    @pytest.fixture
    def rag_config(self):
        """Create RAG config for testing."""
        return RAGConfig(vector_store_type="memory", top_k=3)

    def test_vector_retriever_initialization(self, rag_config):
        """Test vector retriever initialization."""
        # Override to use in-memory store for testing
        rag_config.vector_store_type = "memory"

        retriever = VectorRetriever(rag_config)

        assert retriever.config == rag_config
        assert retriever.vector_store is not None

    @pytest.mark.asyncio
    async def test_add_documents(self, rag_config):
        """Test adding documents to vector retriever."""
        with patch("src.ai.rag.stores.InMemoryVectorStore") as mock_store_class:
            mock_store = AsyncMock()
            mock_store_class.return_value = mock_store

            rag_config.vector_store_type = "memory"
            retriever = VectorRetriever(rag_config)
            retriever.vector_store = mock_store

            documents = [
                Document("Test content 1", {"id": "1"}),
                Document("Test content 2", {"id": "2"}),
            ]

            await retriever.add_documents(documents)

            mock_store.add_documents.assert_called_once_with(documents)

    @pytest.mark.asyncio
    async def test_similarity_search(self, rag_config):
        """Test similarity search."""
        with patch("src.ai.rag.stores.InMemoryVectorStore") as mock_store_class:
            mock_store = AsyncMock()
            mock_results = [
                Document("Result 1", {"id": "1"}, score=0.9),
                Document("Result 2", {"id": "2"}, score=0.8),
            ]
            mock_store.similarity_search.return_value = mock_results
            mock_store_class.return_value = mock_store

            rag_config.vector_store_type = "memory"
            retriever = VectorRetriever(rag_config)
            retriever.vector_store = mock_store

            results = await retriever.similarity_search("test query", k=2)

            assert len(results) == 2
            assert results[0].score == 0.9
            mock_store.similarity_search.assert_called_once_with("test query", 2)


class TestHybridRetriever:
    """Test hybrid retrieval (vector + keyword)."""

    @pytest.fixture
    def rag_config(self):
        """Create RAG config for hybrid retrieval."""
        return RAGConfig(
            retrieval_method=RetrievalMethod.HYBRID, vector_store_type="memory"
        )

    def test_hybrid_retriever_initialization(self, rag_config):
        """Test hybrid retriever initialization."""
        retriever = HybridRetriever(rag_config)

        assert retriever.config == rag_config
        assert retriever.vector_retriever is not None
        assert retriever.keyword_retriever is not None

    @pytest.mark.asyncio
    async def test_hybrid_add_documents(self, rag_config):
        """Test adding documents to hybrid retriever."""
        with patch("src.ai.rag.retrievers.VectorRetriever") as mock_vector:
            with patch("src.ai.rag.retrievers.KeywordRetriever") as mock_keyword:
                mock_vector_instance = AsyncMock()
                mock_keyword_instance = AsyncMock()
                mock_vector.return_value = mock_vector_instance
                mock_keyword.return_value = mock_keyword_instance

                retriever = HybridRetriever(rag_config)
                retriever.vector_retriever = mock_vector_instance
                retriever.keyword_retriever = mock_keyword_instance

                documents = [Document("Test content", {"id": "1"})]

                await retriever.add_documents(documents)

                mock_vector_instance.add_documents.assert_called_once_with(documents)
                mock_keyword_instance.add_documents.assert_called_once_with(documents)

    @pytest.mark.asyncio
    async def test_hybrid_similarity_search(self, rag_config):
        """Test hybrid similarity search."""
        with patch("src.ai.rag.retrievers.VectorRetriever") as mock_vector:
            with patch("src.ai.rag.retrievers.KeywordRetriever") as mock_keyword:
                # Setup mock retrievers
                mock_vector_instance = AsyncMock()
                mock_keyword_instance = AsyncMock()

                vector_results = [
                    Document("Vector result 1", {"id": "1"}, score=0.9),
                    Document("Vector result 2", {"id": "2"}, score=0.8),
                ]
                keyword_results = [
                    Document("Keyword result 1", {"id": "1"}, score=0.7),
                    Document("Keyword result 3", {"id": "3"}, score=0.6),
                ]

                mock_vector_instance.similarity_search.return_value = vector_results
                mock_keyword_instance.keyword_search.return_value = keyword_results

                mock_vector.return_value = mock_vector_instance
                mock_keyword.return_value = mock_keyword_instance

                retriever = HybridRetriever(rag_config)
                retriever.vector_retriever = mock_vector_instance
                retriever.keyword_retriever = mock_keyword_instance

                results = await retriever.similarity_search("test query", k=3)

                assert len(results) <= 3
                # Should combine and rerank results
                assert all(hasattr(doc, "score") for doc in results)


class TestRAGPipeline:
    """Test complete RAG pipeline."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider."""
        mock_llm = AsyncMock(spec=LLMProvider)
        mock_llm.generate.return_value = "Generated answer based on context"
        return mock_llm

    @pytest.fixture
    def rag_config(self):
        """Create RAG config for pipeline testing."""
        return RAGConfig(
            chunk_size=200, chunk_overlap=50, vector_store_type="memory", top_k=2
        )

    def test_rag_pipeline_initialization(self, rag_config, mock_llm_provider):
        """Test RAG pipeline initialization."""
        pipeline = RAGPipeline(rag_config, mock_llm_provider)

        assert pipeline.config == rag_config
        assert pipeline.llm == mock_llm_provider
        assert pipeline.chunker is not None
        assert pipeline.retriever is not None

    @pytest.mark.asyncio
    async def test_ingest_documents(self, rag_config, mock_llm_provider):
        """Test document ingestion."""
        with patch("src.ai.rag.retrievers.VectorRetriever") as mock_retriever_class:
            mock_retriever = AsyncMock()
            mock_retriever_class.return_value = mock_retriever

            pipeline = RAGPipeline(rag_config, mock_llm_provider)
            pipeline.retriever = mock_retriever

            documents = [
                Document(
                    "Long document content that should be chunked into smaller pieces",
                    {"id": "1"},
                )
            ]

            await pipeline.ingest_documents(documents)

            # Should have called add_documents on retriever with chunks
            mock_retriever.add_documents.assert_called_once()
            # Check that documents were chunked
            call_args = mock_retriever.add_documents.call_args[0][0]
            assert len(call_args) >= 1  # Should have at least one chunk

    @pytest.mark.asyncio
    async def test_query_pipeline(self, rag_config, mock_llm_provider):
        """Test complete query pipeline."""
        with patch("src.ai.rag.retrievers.VectorRetriever") as mock_retriever_class:
            mock_retriever = AsyncMock()
            retrieved_docs = [
                Document("Relevant context 1", {"id": "1"}),
                Document("Relevant context 2", {"id": "2"}),
            ]
            mock_retriever.similarity_search.return_value = retrieved_docs
            mock_retriever_class.return_value = mock_retriever

            pipeline = RAGPipeline(rag_config, mock_llm_provider)
            pipeline.retriever = mock_retriever

            answer = await pipeline.query("What is machine learning?")

            assert answer == "Generated answer based on context"
            mock_retriever.similarity_search.assert_called_once()
            mock_llm_provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_no_documents_found(self, rag_config, mock_llm_provider):
        """Test query when no relevant documents are found."""
        with patch("src.ai.rag.retrievers.VectorRetriever") as mock_retriever_class:
            mock_retriever = AsyncMock()
            mock_retriever.similarity_search.return_value = []  # No documents found
            mock_retriever_class.return_value = mock_retriever

            pipeline = RAGPipeline(rag_config, mock_llm_provider)
            pipeline.retriever = mock_retriever

            answer = await pipeline.query("What is machine learning?")

            assert "couldn't find relevant information" in answer.lower()

    @pytest.mark.asyncio
    async def test_retrieve_method(self, rag_config, mock_llm_provider):
        """Test retrieve method."""
        with patch("src.ai.rag.retrievers.VectorRetriever") as mock_retriever_class:
            mock_retriever = AsyncMock()
            retrieved_docs = [Document("Test doc", {"id": "1"})]
            mock_retriever.similarity_search.return_value = retrieved_docs
            mock_retriever_class.return_value = mock_retriever

            pipeline = RAGPipeline(rag_config, mock_llm_provider)
            pipeline.retriever = mock_retriever

            results = await pipeline.retrieve("test query", k=1)

            assert len(results) == 1
            assert results[0].content == "Test doc"

    @pytest.mark.asyncio
    async def test_evaluate_retrieval(self, rag_config, mock_llm_provider):
        """Test retrieval evaluation."""
        with patch("src.ai.rag.retrievers.VectorRetriever") as mock_retriever_class:
            mock_retriever = AsyncMock()

            # Mock retrieval results with proper async mock
            async def mock_similarity_search(query, k):
                if "query1" in query:
                    doc = Document("doc1", {"id": "doc1"})
                    doc.id = "doc1"  # Set ID directly
                    return [doc]
                elif "query2" in query:
                    doc = Document("doc2", {"id": "doc2"})
                    doc.id = "doc2"  # Set ID directly
                    return [doc]
                return []

            mock_retriever.similarity_search = mock_similarity_search
            mock_retriever_class.return_value = mock_retriever

            pipeline = RAGPipeline(rag_config, mock_llm_provider)
            pipeline.retriever = mock_retriever

            queries = ["query1", "query2"]
            ground_truth = [["doc1"], ["doc2"]]

            results = await pipeline.evaluate_retrieval(queries, ground_truth)

            assert "precision" in results
            assert "recall" in results
            assert "f1" in results
            # Perfect retrieval should give high scores
            assert results["precision"] > 0.5
            assert results["recall"] > 0.5


class TestRAGIntegration:
    """Integration tests for RAG components."""

    @pytest.mark.asyncio
    async def test_end_to_end_rag_pipeline(self):
        """Test complete RAG pipeline integration."""
        # Create config
        config = RAGConfig(
            chunk_size=100, chunk_overlap=20, vector_store_type="memory", top_k=2
        )

        # Create mock LLM
        mock_llm = AsyncMock(spec=LLMProvider)
        mock_llm.generate.return_value = "AI is artificial intelligence."

        # Create pipeline
        pipeline = RAGPipeline(config, mock_llm)

        # Create test documents
        documents = [
            Document(
                "Artificial Intelligence (AI) is the simulation of human intelligence in machines.",
                {"source": "ai_intro.txt", "topic": "AI"},
            ),
            Document(
                "Machine Learning is a subset of AI that enables computers to learn from data.",
                {"source": "ml_intro.txt", "topic": "ML"},
            ),
        ]

        # Test the pipeline
        await pipeline.ingest_documents(documents)
        answer = await pipeline.query("What is AI?")

        assert answer == "AI is artificial intelligence."
        mock_llm.generate.assert_called_once()

    def test_chunking_strategies_integration(self):
        """Test different chunking strategies work together."""
        _ = Document(
            "This is sentence one. This is sentence two. This is sentence three.",
            {"id": "test"},
        )

        strategies = [
            ChunkingStrategy.FIXED_SIZE,
            ChunkingStrategy.SENTENCE,
            ChunkingStrategy.PARAGRAPH,
        ]

        for strategy in strategies:
            chunker = DocumentChunker(chunk_size=50, strategy=strategy)
            # Should not raise errors
            assert chunker.strategy == strategy


if __name__ == "__main__":
    pytest.main([__file__])
