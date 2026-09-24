"""
Main FastAPI Application Entrypoint.
Harrison's Principles of Internal Medicine - Production RAG Engine.
"""

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.core.config import config
from app.api.router import api_router
from app.api import dependencies as deps

app = FastAPI(
    title="Harrison's Principles of Internal Medicine - Clinical RAG API",
    description="Production-grade, zero-cost, local clinical decision support RAG engine.",
    version="1.0.0"
)

# Enable CORS for cross-origin frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount REST API Routes
app.include_router(api_router)


@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_dashboard():
    """Serves the professional HTML/CSS Clinical Dashboard."""
    index_path = config.BASE_DIR / "app" / "web" / "index.html"
    if not index_path.exists():
        index_path = config.BASE_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="index.html dashboard not found")


@app.on_event("startup")
async def startup_event():
    deps.initialize_pipeline_services()


@app.on_event("shutdown")
async def shutdown_event():
    deps.shutdown_pipeline_services()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.API_HOST, port=config.API_PORT, reload=False)
