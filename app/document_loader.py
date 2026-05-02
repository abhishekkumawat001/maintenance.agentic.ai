"""
Document Loader — Ingests PDF, DOCX, and TXT files into the vector store.
Handles text extraction, chunking, and embedding pipeline.
"""

import hashlib
import logging
import os
from typing import List, Tuple

from app.config import EmbeddingConfig
from app.knowledge_base import MaintenanceKnowledgeBase
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads documents, chunks them, and ingests into ChromaDB."""

    def __init__(self, vector_store: VectorStore,
                 knowledge_base: MaintenanceKnowledgeBase):
        self.vector_store = vector_store
        self.knowledge_base = knowledge_base
        self.chunk_size = EmbeddingConfig.CHUNK_SIZE
        self.chunk_overlap = EmbeddingConfig.CHUNK_OVERLAP

    # ─── Text Extraction ───

    def load_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            return text.strip()
        except Exception as e:
            logger.error("Error reading PDF %s: %s", file_path, e)
            return ""

    def load_docx(self, file_path: str) -> str:
        """Extract text from a DOCX file."""
        try:
            import docx
            doc = docx.Document(file_path)
            text = "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())
            return text.strip()
        except Exception as e:
            logger.error("Error reading DOCX %s: %s", file_path, e)
            return ""

    def load_txt(self, file_path: str) -> str:
        """Read plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error("Error reading TXT %s: %s", file_path, e)
            return ""

    def load_file(self, file_path: str) -> str:
        """Auto-detect file type and extract text."""
        ext = os.path.splitext(file_path)[1].lower()
        loaders = {
            '.pdf': self.load_pdf,
            '.docx': self.load_docx,
            '.txt': self.load_txt,
            '.md': self.load_txt,
        }
        loader = loaders.get(ext)
        if not loader:
            logger.warning("Unsupported file type: %s", ext)
            return ""
        return loader(file_path)

    # ─── Chunking ───

    def chunk_text(self, text: str, chunk_size: int = None,
                   overlap: int = None) -> List[str]:
        """
        Split text into overlapping chunks using recursive character splitting.
        Tries to split on paragraph breaks first, then sentences, then words.
        """
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap

        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        # Split by paragraphs first
        separators = ["\n\n", "\n", ". ", " "]
        chunks = []
        current_chunk = ""

        # Use paragraph-level splitting
        paragraphs = text.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk += ("\n\n" + para if current_chunk else para)
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # If a single paragraph is too long, split by sentences
                if len(para) > chunk_size:
                    sentences = para.replace('. ', '.\n').split('\n')
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                            current_chunk += (" " + sentence if current_chunk else sentence)
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence
                else:
                    current_chunk = para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Apply overlap: prepend the tail of the previous chunk
        if overlap > 0 and len(chunks) > 1:
            overlapped_chunks = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_tail = chunks[i - 1][-overlap:]
                overlapped_chunks.append(prev_tail + " " + chunks[i])
            return overlapped_chunks

        return chunks

    # ─── Ingestion Pipeline ───

    def _generate_chunk_id(self, filename: str, chunk_index: int) -> str:
        """Generate a deterministic ID for a document chunk."""
        raw = f"{filename}::chunk_{chunk_index}"
        return hashlib.md5(raw.encode()).hexdigest()

    def ingest_file(self, file_path: str, metadata: dict = None) -> int:
        """
        Full ingestion pipeline: load → chunk → embed → store.
        Returns the number of chunks ingested.
        """
        filename = os.path.basename(file_path)

        # Check if already ingested
        if self.knowledge_base.is_document_ingested(filename):
            logger.info("Document '%s' already ingested, skipping", filename)
            return 0

        # Extract text
        text = self.load_file(file_path)
        if not text:
            logger.warning("No text extracted from %s", file_path)
            return 0

        # Chunk
        chunks = self.chunk_text(text)
        if not chunks:
            return 0

        # Prepare for ChromaDB
        base_metadata = {
            'source': filename,
            'file_type': os.path.splitext(file_path)[1].lower(),
        }
        if metadata:
            base_metadata.update(metadata)

        ids = [self._generate_chunk_id(filename, i) for i in range(len(chunks))]
        metadatas = [{**base_metadata, 'chunk_index': i} for i in range(len(chunks))]

        # Embed and store
        self.vector_store.add_documents(texts=chunks, metadatas=metadatas, ids=ids)

        # Track in SQLite
        self.knowledge_base.track_document(
            filename=filename,
            file_path=file_path,
            file_type=base_metadata['file_type'],
            chunk_count=len(chunks)
        )

        logger.info("Ingested '%s': %d chunks", filename, len(chunks))
        return len(chunks)

    def ingest_directory(self, dir_path: str, metadata: dict = None) -> int:
        """Ingest all supported files in a directory. Returns total chunks."""
        if not os.path.isdir(dir_path):
            logger.warning("Directory not found: %s", dir_path)
            return 0

        total_chunks = 0
        supported_extensions = {'.pdf', '.docx', '.txt', '.md'}

        for filename in sorted(os.listdir(dir_path)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in supported_extensions:
                file_path = os.path.join(dir_path, filename)
                total_chunks += self.ingest_file(file_path, metadata)

        logger.info("Ingested directory '%s': %d total chunks", dir_path, total_chunks)
        return total_chunks

    def ingest_text_directly(self, text: str, source_name: str,
                              metadata: dict = None) -> int:
        """Ingest raw text directly (for pasted content or API data)."""
        if self.knowledge_base.is_document_ingested(source_name):
            return 0

        chunks = self.chunk_text(text)
        if not chunks:
            return 0

        base_metadata = {'source': source_name, 'file_type': 'text', **(metadata or {})}
        ids = [self._generate_chunk_id(source_name, i) for i in range(len(chunks))]
        metadatas = [{**base_metadata, 'chunk_index': i} for i in range(len(chunks))]

        self.vector_store.add_documents(texts=chunks, metadatas=metadatas, ids=ids)
        self.knowledge_base.track_document(
            filename=source_name, file_path="direct_input",
            file_type="text", chunk_count=len(chunks)
        )

        return len(chunks)
