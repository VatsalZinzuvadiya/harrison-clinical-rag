"""
Core Module: Application configuration, logging, and custom exception definitions.
"""

from app.core.config import config, Config
from app.core.exceptions import (
    ClinicalRAGException,
    DocumentIngestionError,
    VectorStoreError,
    RetrievalError,
    RerankerError,
    LLMGenerationError,
    PDFRenderingError,
)

__all__ = [
    "config",
    "Config",
    "ClinicalRAGException",
    "DocumentIngestionError",
    "VectorStoreError",
    "RetrievalError",
    "RerankerError",
    "LLMGenerationError",
    "PDFRenderingError",
]
