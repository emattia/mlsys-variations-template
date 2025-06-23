"""Document chunking strategies for RAG systems."""

import re
from enum import Enum

from .base import Document


class ChunkingStrategy(Enum):
    """Different chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"


class DocumentChunker:
    """Handles document chunking with various strategies."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

    async def chunk_document(self, document: Document) -> list[Document]:
        """Chunk a document into smaller pieces."""
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._fixed_size_chunking(document)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            return self._sentence_chunking(document)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            return self._paragraph_chunking(document)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            return await self._semantic_chunking(document)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

    def _fixed_size_chunking(self, document: Document) -> list[Document]:
        """Split document into fixed-size chunks with overlap."""
        text = document.content
        chunks = []

        start = 0
        chunk_id = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to break at word boundaries
            if end < len(text):
                # Look for the last space before the end
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_metadata = document.metadata.copy()
                chunk_metadata.update(
                    {
                        "chunk_id": chunk_id,
                        "chunk_start": start,
                        "chunk_end": end,
                        "parent_id": document.id,
                    }
                )

                chunks.append(
                    Document(
                        content=chunk_text,
                        metadata=chunk_metadata,
                        id=f"{document.id}_chunk_{chunk_id}"
                        if document.id
                        else f"chunk_{chunk_id}",
                    )
                )

                chunk_id += 1

            # Move start position with overlap
            start = max(start + self.chunk_size - self.chunk_overlap, start + 1)

        return chunks

    def _sentence_chunking(self, document: Document) -> list[Document]:
        """Split document into chunks based on sentences."""
        # Simple sentence splitting (can be enhanced with spaCy or NLTK)
        sentences = re.split(r"(?<=[.!?])\s+", document.content)

        chunks = []
        current_chunk = ""
        chunk_id = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunk_metadata = document.metadata.copy()
                    chunk_metadata.update(
                        {
                            "chunk_id": chunk_id,
                            "chunking_method": "sentence",
                            "parent_id": document.id,
                        }
                    )

                    chunks.append(
                        Document(
                            content=current_chunk.strip(),
                            metadata=chunk_metadata,
                            id=f"{document.id}_chunk_{chunk_id}"
                            if document.id
                            else f"chunk_{chunk_id}",
                        )
                    )

                    chunk_id += 1

                current_chunk = sentence

        # Add the last chunk
        if current_chunk:
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunking_method": "sentence",
                    "parent_id": document.id,
                }
            )

            chunks.append(
                Document(
                    content=current_chunk.strip(),
                    metadata=chunk_metadata,
                    id=f"{document.id}_chunk_{chunk_id}"
                    if document.id
                    else f"chunk_{chunk_id}",
                )
            )

        return chunks

    def _paragraph_chunking(self, document: Document) -> list[Document]:
        """Split document into chunks based on paragraphs."""
        paragraphs = document.content.split("\n\n")

        chunks = []
        current_chunk = ""
        chunk_id = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(current_chunk) + len(paragraph) + 2 <= self.chunk_size:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            else:
                if current_chunk:
                    chunk_metadata = document.metadata.copy()
                    chunk_metadata.update(
                        {
                            "chunk_id": chunk_id,
                            "chunking_method": "paragraph",
                            "parent_id": document.id,
                        }
                    )

                    chunks.append(
                        Document(
                            content=current_chunk.strip(),
                            metadata=chunk_metadata,
                            id=f"{document.id}_chunk_{chunk_id}"
                            if document.id
                            else f"chunk_{chunk_id}",
                        )
                    )

                    chunk_id += 1

                current_chunk = paragraph

        # Add the last chunk
        if current_chunk:
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunking_method": "paragraph",
                    "parent_id": document.id,
                }
            )

            chunks.append(
                Document(
                    content=current_chunk.strip(),
                    metadata=chunk_metadata,
                    id=f"{document.id}_chunk_{chunk_id}"
                    if document.id
                    else f"chunk_{chunk_id}",
                )
            )

        return chunks

    async def _semantic_chunking(self, document: Document) -> list[Document]:
        """
        Split document based on semantic similarity (placeholder).

        This would require sentence embeddings to group semantically
        similar sentences together. For now, falls back to sentence chunking.
        """
        # TODO: Implement semantic chunking with sentence embeddings
        # This would involve:
        # 1. Split into sentences
        # 2. Generate embeddings for each sentence
        # 3. Group semantically similar sentences
        # 4. Create chunks from groups

        # For now, fall back to sentence chunking
        return self._sentence_chunking(document)
