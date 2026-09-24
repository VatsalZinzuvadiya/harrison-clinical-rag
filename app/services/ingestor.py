"""
PDF Ingestion & Font Inspection Module

Extracts page-by-page structured text blocks from large PDFs (2000+ pages) using PyMuPDF (fitz).
Preserves font metrics (font size, bold/italic flags, font names) and page numbers, enabling
structure-aware heading detection downstream.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter
from tqdm import tqdm

try:
    import pymupdf as fitz  # PyMuPDF
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

from app.core.config import config
from app.core.exceptions import DocumentIngestionError


class PDFIngestor:
    """
    Robust PDF Parser preserving layout and typography metadata.
    Designed to process 2000+ page medical reference texts with memory efficiency.
    """

    def __init__(self, pdf_path: Optional[Path] = None):
        self.pdf_path = Path(pdf_path) if pdf_path else config.PDF_PATH
        if fitz is None:
            raise DocumentIngestionError(
                "PyMuPDF is required for PDF ingestion. Install via `pip install PyMuPDF`."
            )

    def extract_blocks(self, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Parses PDF blocks page-by-page and extracts text along with dominant font metrics.

        Returns a list of block dictionaries:
        [
            {
                "page_num": int,
                "text": str,
                "font_size": float,
                "is_bold": bool,
                "is_italic": bool,
                "font_name": str
            }, ...
        ]
        """
        if not self.pdf_path.exists():
            raise DocumentIngestionError(
                f"PDF file not found at: {self.pdf_path.resolve()}\n"
                "Please place 'harrison.pdf' in the data directory or provide a valid --pdf path."
            )

        try:
            doc = fitz.open(self.pdf_path)
        except Exception as e:
            raise DocumentIngestionError(f"Failed to open PDF document: {e}")

        total_pages = len(doc)
        pages_to_process = min(total_pages, max_pages) if max_pages else total_pages

        print(f"[PDFIngestor] Opening '{self.pdf_path.name}' ({total_pages} total pages)...")
        print(f"[PDFIngestor] Extracting typography metadata for {pages_to_process} pages...")

        extracted_blocks: List[Dict[str, Any]] = []

        for page_idx in tqdm(range(pages_to_process), desc="Parsing PDF Pages"):
            page = doc[page_idx]
            page_num = page_idx + 1  # 1-indexed page numbering for citations
            
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue

                block_text_spans = []
                font_sizes = []
                bold_flags = []
                italic_flags = []
                font_names = []

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        span_text = span.get("text", "").strip()
                        if not span_text:
                            continue

                        block_text_spans.append(span.get("text", ""))
                        font_sizes.append(round(span.get("size", 0.0), 1))
                        
                        flags = span.get("flags", 0)
                        font_name = span.get("font", "").lower()

                        is_bold = bool(flags & 2**4) or ("bold" in font_name) or ("black" in font_name) or ("heavy" in font_name)
                        is_italic = bool(flags & 2**1) or ("italic" in font_name) or ("oblique" in font_name)

                        bold_flags.append(is_bold)
                        italic_flags.append(is_italic)
                        font_names.append(font_name)

                full_block_text = " ".join(block_text_spans).strip()
                if not full_block_text:
                    continue

                dominant_font_size = Counter(font_sizes).most_common(1)[0][0] if font_sizes else 10.0
                dominant_bold = (sum(bold_flags) / len(bold_flags)) > 0.4 if bold_flags else False
                dominant_italic = (sum(italic_flags) / len(italic_flags)) > 0.4 if italic_flags else False
                dominant_font_name = Counter(font_names).most_common(1)[0][0] if font_names else "unknown"

                extracted_blocks.append({
                    "page_num": page_num,
                    "text": full_block_text,
                    "font_size": dominant_font_size,
                    "is_bold": dominant_bold,
                    "is_italic": dominant_italic,
                    "font_name": dominant_font_name
                })

        doc.close()
        print(f"[PDFIngestor] Successfully extracted {len(extracted_blocks)} text blocks across {pages_to_process} pages.")
        return extracted_blocks
