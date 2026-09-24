"""
Centralized Application Configuration Module.
Provides authoritative configuration management via Pydantic BaseSettings.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Base Directory Resolution
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
VECTOR_DB_DIR = DATA_DIR / "chroma_db"

# Ensure essential data directories exist at startup
CACHE_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


def _find_default_pdf() -> Path:
    """Auto-detects Harrison's PDF file in BASE_DIR or DATA_DIR if present."""
    default_target = DATA_DIR / "harrison.pdf"
    if default_target.exists():
        return default_target
    
    for search_dir in [BASE_DIR, DATA_DIR]:
        pdfs = list(search_dir.glob("*.pdf"))
        if pdfs:
            return pdfs[0]
            
    return default_target


class Config(BaseModel):
    """
    Central Authoritative Configuration Model.
    """
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = DATA_DIR
    CACHE_DIR: Path = CACHE_DIR
    VECTOR_DB_DIR: Path = VECTOR_DB_DIR

    # -----------------------------------------------------------------------------
    # Ingestion & Heading Detection Thresholds
    # -----------------------------------------------------------------------------
    PDF_PATH: Path = Field(default_factory=_find_default_pdf)
    MIN_HEADING_FONT_SIZE: float = 12.0
    REQUIRE_BOLD_FOR_SMALL_HEADINGS: bool = True
    SMALL_HEADING_FONT_SIZE: float = 11.0
    CHUNKS_CACHE_FILE: Path = CACHE_DIR / "processed_chunks.json"

    # -----------------------------------------------------------------------------
    # Structure-Aware Chunking Strategy
    # -----------------------------------------------------------------------------
    TARGET_CHUNK_TOKENS: int = 400
    TOKEN_OVERLAP: int = 75
    FORCE_SPLIT_MAX_TOKENS: int = 800

    # -----------------------------------------------------------------------------
    # Embedding & Vector Database
    # -----------------------------------------------------------------------------
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    CHROMA_COLLECTION_NAME: str = "harrison_medical_textbook"
    EMBEDDING_BATCH_SIZE: int = 64

    # -----------------------------------------------------------------------------
    # Hybrid Retrieval (Dense + Sparse BM25 + RRF)
    # -----------------------------------------------------------------------------
    DENSE_TOP_K: int = 30
    BM25_TOP_K: int = 30
    RRF_K: float = 60.0
    SHORTLIST_SIZE: int = 12
    BM25_CACHE_FILE: Path = CACHE_DIR / "bm25_index.pkl"

    # -----------------------------------------------------------------------------
    # Cross-Encoder Reranking
    # -----------------------------------------------------------------------------
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"
    TOP_N_RERANKED: int = 8

    # -----------------------------------------------------------------------------
    # Jina AI Cloud Embeddings API Configuration
    # -----------------------------------------------------------------------------
    USE_JINA_EMBEDDINGS: bool = True
    JINA_API_KEY: str = Field(default_factory=lambda: os.getenv("JINA_API_KEY", ""))
    JINA_EMBEDDING_MODEL: str = "jina-embeddings-v3"
    JINA_EMBEDDING_DIM: int = 384

    # -----------------------------------------------------------------------------
    # Groq Cloud LLM API Configuration
    # -----------------------------------------------------------------------------
    GROQ_API_KEY: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    GROQ_MODEL: str = "openai/gpt-oss-120b"

    # -----------------------------------------------------------------------------
    # Local LLM Generation (Ollama API / Local Runtime)
    # -----------------------------------------------------------------------------
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 2048

    # -----------------------------------------------------------------------------
    # API & Service Configuration
    # -----------------------------------------------------------------------------
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000


# Global Singleton Config Instance
config = Config()
