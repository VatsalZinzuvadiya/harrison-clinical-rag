"""
PDF Page Rendering & File Streaming Controller.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse
from app.core.config import config
from app.core.exceptions import PDFRenderingError
from app.api import dependencies as deps

router = APIRouter()


@router.get("/pdf/page/{page_num}", summary="Render PDF Page as PNG Image with Text Highlighting")
async def get_pdf_page_image(page_num: int, highlight: Optional[str] = None, dpi: Optional[int] = 150):
    """
    Renders exact PDF page_num (1-indexed) as a crisp high-res PNG image.
    If `highlight` text snippet is provided, searches text on the page and applies
    glowing gold/yellow highlight annotations directly onto the textbook page.
    """
    pdf_service = deps.get_pdf_service()
    if not pdf_service:
        raise HTTPException(status_code=500, detail="PDF rendering service unavailable")

    try:
        png_bytes = pdf_service.render_page_image(page_num=page_num, highlight=highlight, dpi=dpi)
        return Response(content=png_bytes, media_type="image/png")
    except PDFRenderingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected rendering error: {e}")


@router.get("/pdf/file", summary="Stream Raw PDF File")
async def get_pdf_file():
    """
    Streams full Harrison PDF file for native browser viewer at #page=X.
    """
    pdf_path = config.PDF_PATH
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Harrison PDF file not found")
    return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_path.name)
