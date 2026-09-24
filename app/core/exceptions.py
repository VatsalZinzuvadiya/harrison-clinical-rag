"""
Custom Clinical RAG Domain Exceptions.
Provides explicit exception taxonomy for graceful error handling across pipeline layers.
"""

class ClinicalRAGException(Exception):
    """Base exception for all Clinical RAG domain errors."""
    def __init__(self, message: str, details: str = ""):
        super().__init__(message)
        self.message = message
        self.details = details


class DocumentIngestionError(ClinicalRAGException):
    """Raised when PDF ingestion or layout parsing fails."""
    pass


class VectorStoreError(ClinicalRAGException):
    """Raised when ChromaDB persistent operations or embedding generation fails."""
    pass


class RetrievalError(ClinicalRAGException):
    """Raised during dense/sparse retrieval or Reciprocal Rank Fusion errors."""
    pass


class RerankerError(ClinicalRAGException):
    """Raised when Cross-Encoder rescoring fails."""
    pass


class LLMGenerationError(ClinicalRAGException):
    """Raised when Groq API or Ollama local inference fails."""
    pass


class PDFRenderingError(ClinicalRAGException):
    """Raised when PyMuPDF fails to render page image or dynamic highlights."""
    pass
