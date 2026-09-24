import warnings
warnings.filterwarnings("ignore")

import time
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import config
from vector_store import LocalVectorStore
from hybrid_retriever import HybridRetriever
from reranker import LocalReranker
from generator import LocalClinicalGenerator

# Global Singleton Services
services: Dict[str, Any] = {}

app = FastAPI(
    title="Harrison's Principles of Internal Medicine - Clinical RAG API",
    description="Production-grade, local, zero-cost clinical decision support RAG engine.",
    version="1.0.0"
)

@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_dashboard():
    """Serves the professional HTML/CSS Clinical Dashboard."""
    index_path = Path(__file__).resolve().parent / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="index.html dashboard not found")

@app.on_event("startup")
async def startup_event():
    """
    Startup Event Handler.
    Pre-loads heavy PyTorch, ChromaDB, and BM25 models once at server launch.
    """
    print("\n=====================================================================")
    print(" FASTAPI STARTUP: Pre-loading Clinical RAG Models...")
    print("=====================================================================")

    # Initialize Vector Store & Embedder
    v_store = LocalVectorStore()
    
    # Initialize Hybrid Retriever (Dense + BM25)
    retriever = HybridRetriever(vector_store=v_store)
    
    # Initialize Cross-Encoder Reranker
    reranker = LocalReranker()
    
    # Initialize Clinical Generator
    generator = LocalClinicalGenerator()

    services["vector_store"] = v_store
    services["retriever"] = retriever
    services["reranker"] = reranker
    services["generator"] = generator

    print(" -> All pipeline models successfully initialized & ready.")
    print("=====================================================================\n")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown Event Handler."""
    services.clear()
    print("[FastAPI] Pipeline services shut down cleanly.")

# Enable CORS for Streamlit or external web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class QueryRequest(BaseModel):
    question: str = Field(..., description="Doctor's question or patient symptom presentation text", min_length=3)
    top_n: Optional[int] = Field(config.TOP_N_RERANKED, description="Number of final reranked passages to send to LLM context")
    include_sources: Optional[bool] = Field(True, description="Whether to include cited source passages in response")


class SourcePassage(BaseModel):
    chunk_id: str
    page_numbers: List[int]
    heading_trail: str
    text_preview: str
    rerank_score: Optional[float] = None
    rrf_score: Optional[float] = None


class LatencyBreakdown(BaseModel):
    retrieval_ms: float
    rerank_ms: float
    generation_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourcePassage]
    latency: LatencyBreakdown


@app.get("/health", summary="Pipeline Diagnostics & Health Status")
async def health_check():
    """
    Returns diagnostic health status of vector database, BM25 index, and Ollama LLM.
    """
    v_store: LocalVectorStore = services.get("vector_store")
    retriever: HybridRetriever = services.get("retriever")
    generator: LocalClinicalGenerator = services.get("generator")

    if not v_store or not retriever or not generator:
        raise HTTPException(status_code=503, detail="Services not fully initialized")

    llm_status = generator.check_llm_status()

    return {
        "status": "healthy",
        "vector_store": {
            "indexed": v_store.is_indexed(),
            "count": v_store.get_count(),
            "device": v_store.device
        },
        "bm25_index": {
            "loaded": retriever.bm25_engine is not None,
            "passages_count": len(retriever.chunk_ids_order)
        },
        "llm_status": llm_status,
        "ollama_llm": llm_status
    }


@app.post("/query", response_model=QueryResponse, summary="Submit Clinical Query / Patient Presentation")
async def query_clinical_rag(req: QueryRequest):
    """
    End-to-End Clinical Query Handler:
    1. Hybrid Search (Dense Vector + BM25 Sparse with RRF Fusion).
    2. Cross-Encoder Reranking over top candidates.
    3. Grounded Clinical Generation with Ollama LLM & Page Citations.
    """
    t_start = time.time()

    retriever: HybridRetriever = services.get("retriever")
    reranker: LocalReranker = services.get("reranker")
    generator: LocalClinicalGenerator = services.get("generator")

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

    # Step 3: Local LLM Clinical Generation
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host=config.API_HOST, port=config.API_PORT, reload=False)
