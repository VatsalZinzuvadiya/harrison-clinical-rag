"""
PDF Page Rendering & Text Highlighting Service

Renders high-resolution PNG images of PDF pages from Harrison's textbook and applies
dynamic yellow/gold highlight annotations over matching text bounding boxes.
"""

from pathlib import Path
from typing import Optional
from fastapi import HTTPException

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

from app.core.config import config
from app.core.exceptions import PDFRenderingError


class PDFPageService:
    """
    Domain service for rendering textbook PDF pages and text annotations.
    """

    def __init__(self, pdf_path: Optional[Path] = None):
        self.pdf_path = Path(pdf_path) if pdf_path else config.PDF_PATH
        if fitz is None:
            raise PDFRenderingError("PyMuPDF library is not installed.")

    def render_page_image(self, page_num: int, highlight: Optional[str] = None, dpi: int = 150) -> bytes:
        """
        Renders 1-indexed page_num to PNG bytes.
        If `highlight` text is provided, searches page for text rectangles and applies yellow highlights.
        """
        if not self.pdf_path.exists():
            raise PDFRenderingError(f"Harrison PDF file not found at: {self.pdf_path.resolve()}")

        try:
            doc = fitz.open(self.pdf_path)
            total_pages = len(doc)

            if page_num < 1 or page_num > total_pages:
                doc.close()
                raise PDFRenderingError(f"Page number {page_num} out of range (1 to {total_pages})")

            page = doc.load_page(page_num - 1)

            # Apply Dynamic Text Highlighting if requested
            if highlight and highlight.strip():
                clean_hl = highlight.strip()
                instances = page.search_for(clean_hl)

                # Fallback to phrase-window search if exact string spans line breaks
                if not instances and len(clean_hl.split()) > 2:
                    words = clean_hl.split()
                    for i in range(0, min(len(words), 25), 4):
                        phrase = " ".join(words[i:i + 5])
                        if len(phrase) > 10:
                            p_insts = page.search_for(phrase)
                            instances.extend(p_insts)

                for inst in instances:
                    annot = page.add_highlight_annot(inst)
                    if annot:
                        annot.set_colors(stroke=(1.0, 0.82, 0.0))  # Gold / Yellow highlight
                        annot.update()

            pix = page.get_pixmap(dpi=dpi)
            png_bytes = pix.tobytes("png")
            doc.close()
            return png_bytes

        except PDFRenderingError:
            raise
        except Exception as e:
            raise PDFRenderingError(f"Error rendering PDF page: {e}")
