"""
Dependency Injection Container for Clinical RAG Pipeline Services.
Pre-loads singleton PyTorch, ChromaDB, and BM25 instances at server startup.
"""

from typing import Dict, Any
from app.services.vector_store import LocalVectorStore
from app.services.retriever import HybridRetriever
from app.services.reranker import LocalReranker
from app.services.generator import LocalClinicalGenerator
from app.services.pdf_service import PDFPageService

# Container Dictionary for Singletons
_pipeline_services: Dict[str, Any] = {}


def initialize_pipeline_services() -> None:
    """Pre-loads heavy models once at server startup."""
    print("\n=====================================================================")
    print(" FASTAPI STARTUP: Pre-loading Clinical RAG Pipeline Models...")
    print("=====================================================================")

    v_store = LocalVectorStore()
    retriever = HybridRetriever(vector_store=v_store)
    reranker = LocalReranker()
    generator = LocalClinicalGenerator()
    pdf_service = PDFPageService()

    _pipeline_services["vector_store"] = v_store
    _pipeline_services["retriever"] = retriever
    _pipeline_services["reranker"] = reranker
    _pipeline_services["generator"] = generator
    _pipeline_services["pdf_service"] = pdf_service

    print(" -> All pipeline models successfully initialized & ready.")
    print("=====================================================================\n")


def shutdown_pipeline_services() -> None:
    """Cleanly closes open handles on shutdown."""
    _pipeline_services.clear()
    print("[FastAPI] Pipeline services shut down cleanly.")


def get_vector_store() -> LocalVectorStore:
    return _pipeline_services.get("vector_store")


def get_retriever() -> HybridRetriever:
    return _pipeline_services.get("retriever")


def get_reranker() -> LocalReranker:
    return _pipeline_services.get("reranker")


def get_generator() -> LocalClinicalGenerator:
    return _pipeline_services.get("generator")


def get_pdf_service() -> PDFPageService:
    return _pipeline_services.get("pdf_service")
