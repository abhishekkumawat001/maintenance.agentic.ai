"""
Centralized configuration for the Maintenance AI system.
All environment variables, paths, and tunable parameters live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class LLMConfig:
    """LLM provider configuration"""
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


class EmbeddingConfig:
    """Embedding model configuration"""
    MODEL_NAME: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))


class VectorStoreConfig:
    """ChromaDB vector store configuration"""
    PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
    COLLECTION_NAME: str = "maintenance_docs"
    ENTITY_COLLECTION_NAME: str = "entity_memory"
    N_RESULTS: int = int(os.getenv("RAG_TOP_K", "5"))


class DatabaseConfig:
    """SQLite database configuration"""
    DB_PATH: str = os.getenv("DB_PATH", "data/maintenance.db")


class MemoryConfig:
    """Conversation memory configuration"""
    MAX_TURNS: int = int(os.getenv("MEMORY_MAX_TURNS", "20"))
    SUMMARY_THRESHOLD: int = int(os.getenv("MEMORY_SUMMARY_THRESHOLD", "15"))
    MAX_CONTEXT_CHARS: int = int(os.getenv("MEMORY_MAX_CONTEXT_CHARS", "3000"))


class SensorThresholds:
    """Default sensor anomaly thresholds"""
    THRESHOLDS = {
        'temperature': {'min': 0, 'max': 80},
        'vibration': {'min': 0, 'max': 10},
        'pressure': {'min': 0, 'max': 100},
        'humidity': {'min': 0, 'max': 100}
    }


class AppConfig:
    """Top-level application config"""
    APP_NAME: str = "Maintenance Agentic AI"
    APP_VERSION: str = "2.0.0"
    UPLOAD_DIR: str = "uploads"
    VISUAL_UPLOAD_DIR: str = "visual_uploads"
    SAMPLE_DOCS_DIR: str = "data/sample_docs"
    DATA_DIR: str = "data"
