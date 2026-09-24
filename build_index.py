"""
End-to-End Indexing Pipeline Runner CLI

Executes the complete index creation flow:
1. PDF Ingestion with font layout profiling.
2. Structure-Aware Heading Detection & Token Chunking.
3. Intermediate Chunk JSON Caching.
4. Local Vector Database Embedding (ChromaDB).
5. Sparse BM25 Keyword Corpus Tokenization & Caching.
"""

import sys
import time
import argparse
from pathlib import Path

from config import config
from chunker import StructureAwareChunker
from vector_store import LocalVectorStore
from hybrid_retriever import HybridRetriever


def run_indexing_pipeline(pdf_path: Path = config.PDF_PATH,
                          max_pages: int = None,
                          force_rebuild: bool = False) -> None:
    """
    Executes the end-to-end indexing pipeline.
    """
    start_time = time.time()

    print("\n=====================================================================")
    print(" HARRISON'S CLINICAL RAG - END-TO-END INDEX BUILDER")
    print("=====================================================================\n")

    # 1. Structure-Aware PDF Ingestion & Chunking
    print("[Pipeline Step 1/4] Running Structure-Aware PDF Chunker...")
    chunker = StructureAwareChunker()
    chunks = chunker.load_or_create_chunks(force_reparse=force_rebuild, max_pages=max_pages)

    if not chunks:
        print("[Pipeline Error] No chunks were produced. Please check your PDF file.")
        sys.exit(1)

    avg_tokens = sum(c.get("token_count", 0) for c in chunks) / len(chunks)
    print(f" -> Generated {len(chunks)} chunks (Avg Token Size: {avg_tokens:.1f} tokens).")

    # 2. Local Vector Store Embedding (ChromaDB)
    print("\n[Pipeline Step 2/4] Building Local Vector Store (ChromaDB)...")
    vector_store = LocalVectorStore()
    vector_store.build_index(chunks=chunks, force_rebuild=force_rebuild)

    # 3. BM25 Sparse Index Building
    print("\n[Pipeline Step 3/4] Building BM25 Sparse Search Index...")
    hybrid_retriever = HybridRetriever(vector_store=vector_store)
    hybrid_retriever.build_bm25_index(chunks=chunks, force_rebuild=force_rebuild)

    elapsed = time.time() - start_time
    print("\n=====================================================================")
    print(f" INDEX BUILDING COMPLETE IN {elapsed:.2f} SECONDS")
    print(f" Total Chunks Indexed: {len(chunks)}")
    print(f" Chroma Vector Count:  {vector_store.get_count()}")
    print(f" BM25 Corpus Size:    {len(hybrid_retriever.chunk_ids_order)}")
    print("=====================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Clinical RAG Harrison Index")
    parser.add_argument("--pdf", type=str, default=None, help="Path to PDF file")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit number of pages for rapid testing")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild indices ignoring existing cache")

    args = parser.parse_args()

    pdf_p = Path(args.pdf) if args.pdf else config.PDF_PATH
    run_indexing_pipeline(pdf_path=pdf_p, max_pages=args.max_pages, force_rebuild=args.force_rebuild)
