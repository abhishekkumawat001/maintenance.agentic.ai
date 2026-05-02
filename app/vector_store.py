"""
Vector Store — ChromaDB wrapper with sentence-transformers embeddings.
Provides semantic search over maintenance documents and entity memory.
"""

import logging
import os
from typing import Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import EmbeddingConfig, VectorStoreConfig

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-backed vector store with local sentence-transformer embeddings."""

    def __init__(self, persist_dir: str = None, model_name: str = None):
        self.persist_dir = persist_dir or VectorStoreConfig.PERSIST_DIR
        self.model_name = model_name or EmbeddingConfig.MODEL_NAME

        # Ensure persistence directory exists
        os.makedirs(self.persist_dir, exist_ok=True)

        # Initialize embedding model
        logger.info("Loading embedding model: %s", self.model_name)
        self.embedding_model = SentenceTransformer(self.model_name)
        logger.info("Embedding model loaded successfully")

        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # Main document collection
        self.collection = self.client.get_or_create_collection(
            name=VectorStoreConfig.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        # Entity memory collection
        self.entity_collection = self.client.get_or_create_collection(
            name=VectorStoreConfig.ENTITY_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(
            "VectorStore initialized — %d document chunks, %d entity entries",
            self.collection.count(), self.entity_collection.count()
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    # ─── Document Methods ───

    def add_documents(self, texts: List[str], metadatas: List[Dict],
                      ids: List[str]) -> None:
        """Add document chunks to the vector store."""
        if not texts:
            return
        embeddings = self._embed(texts)
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        logger.info("Added %d document chunks to vector store", len(texts))

    def query(self, query_text: str, n_results: int = None,
              where: Optional[Dict] = None) -> List[Dict]:
        """
        Semantic search over document chunks.
        Returns list of {content, metadata, distance} dicts, sorted by relevance.
        """
        n_results = n_results or VectorStoreConfig.N_RESULTS

        # Don't query more results than we have documents
        doc_count = self.collection.count()
        if doc_count == 0:
            return []
        n_results = min(n_results, doc_count)

        query_embedding = self._embed([query_text])

        kwargs = {
            "query_embeddings": query_embedding,
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        # Flatten ChromaDB's nested result format
        documents = []
        for i in range(len(results['ids'][0])):
            documents.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                'distance': results['distances'][0][i] if results['distances'] else 0.0,
                'id': results['ids'][0][i]
            })

        return documents

    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        return {
            'document_chunks': self.collection.count(),
            'entity_entries': self.entity_collection.count(),
            'embedding_model': self.model_name,
            'persist_dir': self.persist_dir
        }

    def delete_collection(self) -> None:
        """Delete all document chunks (for reset/testing)."""
        self.client.delete_collection(VectorStoreConfig.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=VectorStoreConfig.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Document collection cleared")

    # ─── Entity Memory Methods ───

    def add_entity(self, entity_id: str, text: str, metadata: Dict) -> None:
        """Add or update an entity memory entry."""
        embedding = self._embed([text])
        # Upsert: add or update
        self.entity_collection.upsert(
            documents=[text],
            embeddings=embedding,
            metadatas=[metadata],
            ids=[entity_id]
        )

    def query_entities(self, query_text: str, n_results: int = 3) -> List[Dict]:
        """Search entity memory."""
        entity_count = self.entity_collection.count()
        if entity_count == 0:
            return []

        n_results = min(n_results, entity_count)
        query_embedding = self._embed([query_text])

        results = self.entity_collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        entities = []
        for i in range(len(results['ids'][0])):
            entities.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                'id': results['ids'][0][i]
            })
        return entities
