import os
from pathlib import Path
from pydantic import BaseModel

# Suppress Hugging Face Windows Symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Base Directory Paths
BASE_DIR = Path(__file__).resolve().parent
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
    
    # Check for any PDF file in BASE_DIR or DATA_DIR
    for search_dir in [BASE_DIR, DATA_DIR]:
        pdfs = list(search_dir.glob("*.pdf"))
        if pdfs:
            return pdfs[0]
            
    return default_target

class Config(BaseModel):
    """
    Central Configuration Module.
    Allows tuning chunk sizes, model selections, hybrid weights,
    and server settings in a single authoritative location.
    """
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = DATA_DIR
    CACHE_DIR: Path = CACHE_DIR
    VECTOR_DB_DIR: Path = VECTOR_DB_DIR

    # -----------------------------------------------------------------------------
    # Ingestion & Heading Detection Thresholds
    # -----------------------------------------------------------------------------
    # Auto-detected path for Harrison's Principles of Internal Medicine PDF
    PDF_PATH: Path = _find_default_pdf()
    
    # Minimum font size (in pt) to consider a block as a potential heading.
    # Standard body text in Harrison's is ~9-10pt, section titles are >= 12-14pt.
    MIN_HEADING_FONT_SIZE: float = 12.0
    
    # Require bold font weight flag for sub-headings if font size is close to body text
    REQUIRE_BOLD_FOR_SMALL_HEADINGS: bool = True
    SMALL_HEADING_FONT_SIZE: float = 11.0

    # Cache file location for processed chunks
    CHUNKS_CACHE_FILE: Path = CACHE_DIR / "processed_chunks.json"

    # -----------------------------------------------------------------------------
    # Structure-Aware Chunking Strategy
    # -----------------------------------------------------------------------------
    # Target token length for standard clinical text chunks (~300-500 tokens).
    # Preserves clinical context without overloading model context windows.
    TARGET_CHUNK_TOKENS: int = 400
    
    # Sliding window token overlap (~15-20% of target tokens).
    # Ensures symptoms, drug dosages, or criteria spanning boundaries aren't split.
    TOKEN_OVERLAP: int = 75
    
    # Hard upper bound for oversized structures (e.g. massive diagnostic tables/lists).
    # Prevents pathologically large chunks from causing embedding or memory issues.
    FORCE_SPLIT_MAX_TOKENS: int = 800

    # -----------------------------------------------------------------------------
    # Embedding & Vector Database
    # -----------------------------------------------------------------------------
    # High-performance, free open-source embedding model from Hugging Face
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    
    # Chroma collection identifier
    CHROMA_COLLECTION_NAME: str = "harrison_medical_textbook"
    
    # Batch size for vector store embedding generation (auto GPU/CPU)
    EMBEDDING_BATCH_SIZE: int = 64

    # -----------------------------------------------------------------------------
    # Hybrid Retrieval (Dense + Sparse BM25 + RRF)
    # -----------------------------------------------------------------------------
    # Number of candidate passages retrieved from Dense (Vector) search
    DENSE_TOP_K: int = 30
    
    # Number of candidate passages retrieved from Sparse (BM25) search
    BM25_TOP_K: int = 30
    
    # Reciprocal Rank Fusion constant k (standard default: 60)
    RRF_K: float = 60.0
    
    # Shortlist candidate pool size sent to expensive Cross-Encoder Reranker
    SHORTLIST_SIZE: int = 12

    # Cache file for BM25 corpus & tokenized index
    BM25_CACHE_FILE: Path = CACHE_DIR / "bm25_index.pkl"

    # -----------------------------------------------------------------------------
    # Cross-Encoder Reranking
    # -----------------------------------------------------------------------------
    # Free, local cross-encoder model for high-precision query-passage rescoring
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"
    
    # Top N final reranked context passages passed to LLM generation (5-8 recommended)
    TOP_N_RERANKED: int = 8

    # -----------------------------------------------------------------------------
    # Jina AI Cloud Embeddings API Configuration (Free High-Speed Vector API)
    # Set USE_JINA_EMBEDDINGS = False to fallback to local BGE SentenceTransformer
    # -----------------------------------------------------------------------------
    USE_JINA_EMBEDDINGS: bool = True
    JINA_API_KEY: str = os.getenv("JINA_API_KEY", "")
    JINA_EMBEDDING_MODEL: str = "jina-embeddings-v3"
    JINA_EMBEDDING_DIM: int = 384

    # -----------------------------------------------------------------------------
    # Groq Cloud LLM API Configuration (Free Ultra-Fast Inference)
    # -----------------------------------------------------------------------------
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = "openai/gpt-oss-120b"

    # -----------------------------------------------------------------------------
    # Local LLM Generation (Ollama API / Local Runtime)
    # -----------------------------------------------------------------------------
    # Base URL for local Ollama server
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # Local open-source LLM model name served by Ollama (e.g., llama3.2, mistral, medllama)
    OLLAMA_MODEL: str = "llama3.2"
    
    # Strict low temperature for deterministic, factual clinical answers
    LLM_TEMPERATURE: float = 0.0
    
    # Max tokens for LLM generation output (2048 tokens ensures complete medical reports and tables)
    LLM_MAX_TOKENS: int = 2048 

    # -----------------------------------------------------------------------------
    # API & Service Configuration
    # -----------------------------------------------------------------------------
    API_HOST: str = "0.0.0.0"
    API_PORT: int = int(os.getenv("PORT", 8000))


# Instantiated global configuration object
config = Config()
