"""
Master FastAPI Router combining health, query, and pdf controllers.
"""

from fastapi import APIRouter
from app.api.endpoints import health, query, pdf

api_router = APIRouter()

api_router.include_router(health.router, tags=["Diagnostics"])
api_router.include_router(query.router, tags=["Clinical RAG Engine"])
api_router.include_router(pdf.router, tags=["Textbook PDF Viewer"])
