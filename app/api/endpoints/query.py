"""
Clinical RAG Query Execution Controller.
"""

import time
from fastapi import APIRouter, HTTPException
from app.core.config import config
from app.schemas.query import QueryRequest, QueryResponse, SourcePassage, LatencyBreakdown
from app.api import dependencies as deps

router = APIRouter()


@router.post("/query", response_model=QueryResponse, summary="Submit Clinical Query / Patient Presentation")
async def query_clinical_rag(req: QueryRequest):
    """
    End-to-End Clinical Query Handler:
    1. Hybrid Search (Dense Vector + BM25 Sparse with RRF Fusion).
    2. PyTorch Multi-threaded Cross-Encoder Reranking over top candidates.
    3. Grounded Zero-Hallucination Clinical Generation with Page Citations.
    """
    t_start = time.time()

    retriever = deps.get_retriever()
    reranker = deps.get_reranker()
    generator = deps.get_generator()

    if not retriever or not reranker or not generator:
        raise HTTPException(status_code=500, detail="Pipeline engines unavailable")

    # Step 1: Hybrid Retrieval (Dense + BM25 + RRF)
    t0 = time.time()
    candidates = retriever.retrieve(
        query=req.question,
        dense_top_k=config.DENSE_TOP_K,
        bm25_top_k=config.BM25_TOP_K,
        shortlist_size=config.SHORTLIST_SIZE
    )
    t_retrieval = (time.time() - t0) * 1000.0

    # Step 2: Cross-Encoder Reranking
    t1 = time.time()
    top_n_requested = req.top_n if req.top_n else config.TOP_N_RERANKED
    reranked_chunks = reranker.rerank(
        query=req.question,
        candidates=candidates,
        top_n=top_n_requested
    )
    t_rerank = (time.time() - t1) * 1000.0

    # Step 3: LLM Clinical Generation
    t2 = time.time()
    answer_text = generator.generate(
        query=req.question,
        context_chunks=reranked_chunks
    )
    t_gen = (time.time() - t2) * 1000.0

    total_time = (time.time() - t_start) * 1000.0

    # Format Source Passages for output
    sources_output = []
    if req.include_sources:
        for c in reranked_chunks:
            sources_output.append(SourcePassage(
                chunk_id=c.get("chunk_id", ""),
                page_numbers=c.get("page_numbers", []),
                heading_trail=c.get("heading_trail", "General"),
                text_preview=c.get("text", "")[:250] + ("..." if len(c.get("text", "")) > 250 else ""),
                rerank_score=c.get("rerank_score"),
                rrf_score=c.get("rrf_score")
            ))

    return QueryResponse(
        question=req.question,
        answer=answer_text,
        sources=sources_output,
        latency=LatencyBreakdown(
            retrieval_ms=round(t_retrieval, 2),
            rerank_ms=round(t_rerank, 2),
            generation_ms=round(t_gen, 2),
            total_ms=round(total_time, 2)
        )
    )
