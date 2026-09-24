"""
Pydantic Schemas for System Diagnostics and Service Health Status.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class VectorStoreStatus(BaseModel):
    indexed: bool
    count: int
    device: str


class BM25Status(BaseModel):
    loaded: bool
    passages_count: int


class LLMStatus(BaseModel):
    available: bool
    provider: str
    models: List[str]


class HealthCheckResponse(BaseModel):
    status: str
    vector_store: VectorStoreStatus
    bm25_index: BM25Status
    llm_status: LLMStatus
    ollama_llm: LLMStatus
