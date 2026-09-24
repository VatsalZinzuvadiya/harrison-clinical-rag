"""
Pydantic Schemas for Clinical Queries, Citations, and Performance Metrics.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from app.core.config import config


class QueryRequest(BaseModel):
    """Client request schema for submitting clinical queries."""
    question: str = Field(
        ...,
        description="Doctor's question or patient symptom presentation text",
        min_length=3,
        example="65-year-old male with hypertension presenting with retrosternal crushing pain radiating to left jaw."
    )
    top_n: Optional[int] = Field(
        default=config.TOP_N_RERANKED,
        description="Number of final reranked context passages sent to LLM context window"
    )
    include_sources: Optional[bool] = Field(
        default=True,
        description="Whether to include cited source textbook passages in response payload"
    )


class SourcePassage(BaseModel):
    """Cited Harrison textbook passage DTO."""
    chunk_id: str
    page_numbers: List[int]
    heading_trail: str
    text_preview: str
    rerank_score: Optional[float] = None
    rrf_score: Optional[float] = None


class LatencyBreakdown(BaseModel):
    """Performance metrics breakdown DTO in milliseconds."""
    retrieval_ms: float
    rerank_ms: float
    generation_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    """End-to-end clinical decision support response DTO."""
    question: str
    answer: str
    sources: List[SourcePassage]
    latency: LatencyBreakdown
