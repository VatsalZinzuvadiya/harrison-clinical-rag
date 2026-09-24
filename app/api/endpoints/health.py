"""
Health Diagnostics API Endpoint Controller.
"""

from fastapi import APIRouter, HTTPException
from app.schemas.health import HealthCheckResponse, VectorStoreStatus, BM25Status, LLMStatus
from app.api import dependencies as deps

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse, summary="Pipeline Diagnostics & Health Status")
async def health_check():
    """Returns diagnostic health status of vector database, BM25 index, and LLM providers."""
    v_store = deps.get_vector_store()
    retriever = deps.get_retriever()
    generator = deps.get_generator()

    if not v_store or not retriever or not generator:
        raise HTTPException(status_code=503, detail="Services not fully initialized")

    llm_status = generator.check_llm_status()

    return HealthCheckResponse(
        status="healthy",
        vector_store=VectorStoreStatus(
            indexed=v_store.is_indexed(),
            count=v_store.get_count(),
            device=v_store.device
        ),
        bm25_index=BM25Status(
            loaded=retriever.bm25_engine is not None,
            passages_count=len(retriever.chunk_ids_order)
        ),
        llm_status=LLMStatus(**llm_status),
        ollama_llm=LLMStatus(**llm_status)
    )
