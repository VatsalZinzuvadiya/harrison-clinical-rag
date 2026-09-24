"""
Structure-Aware Heading & Token Chunker Module

Groups PDF text blocks into topic-coherent sections using font typography metadata (headings).
Splits section texts into token-bounded chunks with overlap, attaches page numbers and
heading metadata, and caches intermediate results to disk.
"""

import json
import uuid
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

try:
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))
    def split_tokens(text: str, token_limit: int, overlap: int) -> List[str]:
        tokens = tokenizer.encode(text)
        sub_chunks = []
        start = 0
        while start < len(tokens):
            end = start + token_limit
            chunk_tokens = tokens[start:end]
            sub_chunks.append(tokenizer.decode(chunk_tokens))
            if end >= len(tokens):
                break
            start += (token_limit - overlap)
        return sub_chunks
except Exception:
    # Fallback to whitespace approximation if tiktoken encoding fails
    def count_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)
    def split_tokens(text: str, token_limit: int, overlap: int) -> List[str]:
        words = text.split()
        words_per_token = 0.75
        word_limit = max(10, int(token_limit * words_per_token))
        word_overlap = max(2, int(overlap * words_per_token))
        sub_chunks = []
        start = 0
        while start < len(words):
            end = start + word_limit
            chunk_words = words[start:end]
            sub_chunks.append(" ".join(chunk_words))
            if end >= len(words):
                break
            start += (word_limit - word_overlap)
        return sub_chunks

from config import config
from ingest import PDFIngestor


class StructureAwareChunker:
    """
    Structure-Aware Clinical Chunker.
    Ensures medical topics remain grouped together under their parent section/subsection headings,
    preventing context fragments across different diseases or diagnostic protocols.
    """

    def __init__(self, target_tokens: int = config.TARGET_CHUNK_TOKENS,
                 overlap_tokens: int = config.TOKEN_OVERLAP,
                 force_max_tokens: int = config.FORCE_SPLIT_MAX_TOKENS):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.force_max_tokens = force_max_tokens

    def _is_heading(self, block: Dict[str, Any]) -> bool:
        """
        Determines whether a block is a heading based on font size and boldness.
        """
        font_sz = block.get("font_size", 0.0)
        is_bold = block.get("is_bold", False)
        text = block.get("text", "").strip()

        # Ignore tiny or very long blocks (headings are usually concise)
        if len(text) > 200 or len(text) < 3:
            return False

        # Rule 1: Font size clearly above threshold
        if font_sz >= config.MIN_HEADING_FONT_SIZE:
            return True

        # Rule 2: Moderate font size + bold flag
        if config.REQUIRE_BOLD_FOR_SMALL_HEADINGS and font_sz >= config.SMALL_HEADING_FONT_SIZE and is_bold:
            return True

        # Rule 3: Text pattern matching (e.g. CHAPTER 12, PART 3, SECTION 4)
        if re.match(r'^(CHAPTER|PART|SECTION|CHAPTER\s+\d+|SECTION\s+\d+)', text, re.IGNORECASE):
            return True

        return False

    def chunk_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes parsed blocks into structured chunks.

        1. Maintains a breadcrumb trail of section headings.
        2. Groups blocks belonging to the same section.
        3. Splits sections into overlapping token-bounded chunks.
        4. Applies safety force-splitting to prevent pathologically large chunks.
        """
        print("[StructureAwareChunker] Grouping blocks into topic sections...")
        
        sections: List[Dict[str, Any]] = []
        current_heading_trail: List[str] = ["Harrison's Manual"]
        current_section_blocks: List[Dict[str, Any]] = []

        def flush_section():
            if current_section_blocks:
                sections.append({
                    "heading_trail": " > ".join(current_heading_trail),
                    "blocks": list(current_section_blocks)
                })

        for block in blocks:
            if self._is_heading(block):
                flush_section()
                heading_text = block["text"].strip()
                
                # Update heading trail context
                font_sz = block.get("font_size", 0.0)
                if font_sz >= config.MIN_HEADING_FONT_SIZE + 2.0:
                    # Major Heading (e.g. Chapter / Part) -> reset sub-trails
                    current_heading_trail = [heading_text]
                else:
                    # Sub-heading -> append to current trail (keep max 3 levels)
                    if len(current_heading_trail) >= 3:
                        current_heading_trail = current_heading_trail[:2] + [heading_text]
                    else:
                        current_heading_trail.append(heading_text)
                
                current_section_blocks = []
            else:
                current_section_blocks.append(block)

        flush_section()
        print(f"[StructureAwareChunker] Identified {len(sections)} topic sections across corpus.")

        final_chunks: List[Dict[str, Any]] = []

        for sec in tqdm(sections, desc="Generating Token Chunks"):
            heading_trail = sec["heading_trail"]
            sec_blocks = sec["blocks"]

            if not sec_blocks:
                continue

            # Accumulate text and page numbers spanned by section
            sec_text_parts = []
            page_numbers_set = set()

            for b in sec_blocks:
                sec_text_parts.append(b["text"])
                page_numbers_set.add(b["page_num"])

            full_sec_text = "\n".join(sec_text_parts).strip()
            if not full_sec_text:
                continue

            sorted_pages = sorted(list(page_numbers_set))
            total_tokens = count_tokens(full_sec_text)

            # Case A: Section fits comfortably in a single target chunk
            if total_tokens <= self.target_tokens:
                chunk_id = str(uuid.uuid4())
                final_chunks.append({
                    "chunk_id": chunk_id,
                    "text": full_sec_text,
                    "heading_trail": heading_trail,
                    "page_numbers": sorted_pages,
                    "token_count": total_tokens
                })
            else:
                # Case B: Section is larger than target chunk -> split with overlap
                sub_texts = split_tokens(full_sec_text, self.target_tokens, self.overlap_tokens)
                
                for sub_t in sub_texts:
                    sub_t_clean = sub_t.strip()
                    if not sub_t_clean:
                        continue
                    
                    sub_tok_count = count_tokens(sub_t_clean)

                    # Safety check: Force-split if sub-chunk is still pathologically large
                    if sub_tok_count > self.force_max_tokens:
                        micro_chunks = split_tokens(sub_t_clean, self.target_tokens // 2, self.overlap_tokens // 2)
                        for mc in micro_chunks:
                            mc_clean = mc.strip()
                            if mc_clean:
                                final_chunks.append({
                                    "chunk_id": str(uuid.uuid4()),
                                    "text": mc_clean,
                                    "heading_trail": heading_trail,
                                    "page_numbers": sorted_pages,
                                    "token_count": count_tokens(mc_clean)
                                })
                    else:
                        final_chunks.append({
                            "chunk_id": str(uuid.uuid4()),
                            "text": sub_t_clean,
                            "heading_trail": heading_trail,
                            "page_numbers": sorted_pages,
                            "token_count": sub_tok_count
                        })

        print(f"[StructureAwareChunker] Generated {len(final_chunks)} token-bounded chunks.")
        return final_chunks

    def load_or_create_chunks(self, force_reparse: bool = False, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Loads cached chunks from disk if available to save re-parsing 2000+ pages.
        Otherwise executes PDF ingestion, chunking, and saves to cache.
        """
        cache_path = config.CHUNKS_CACHE_FILE

        if not force_reparse and cache_path.exists():
            print(f"[StructureAwareChunker] Loading cached chunks from: {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_chunks = json.load(f)
            print(f"[StructureAwareChunker] Loaded {len(cached_chunks)} chunks from cache.")
            return cached_chunks

        print("[StructureAwareChunker] No valid cache found. Running full PDF ingestion & chunking...")
        ingestor = PDFIngestor(pdf_path=config.PDF_PATH)
        blocks = ingestor.extract_blocks(max_pages=max_pages)
        chunks = self.chunk_blocks(blocks)

        # Cache final chunks to disk
        print(f"[StructureAwareChunker] Caching processed chunks to: {cache_path}")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

        return chunks


if __name__ == "__main__":
    chunker = StructureAwareChunker()
    # Test sample run
    chunks = chunker.load_or_create_chunks(max_pages=15)
    if chunks:
        print("\n--- SAMPLE GENERATED CHUNK ---")
        print(f"ID: {chunks[0]['chunk_id']}")
        print(f"Heading Trail: {chunks[0]['heading_trail']}")
        print(f"Pages: {chunks[0]['page_numbers']}")
        print(f"Token Count: {chunks[0]['token_count']}")
        print(f"Text Snippet: {chunks[0]['text'][:150]}...")
