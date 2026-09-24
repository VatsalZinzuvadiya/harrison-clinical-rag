"""
Hybrid Retrieval Engine (Dense + BM25 Sparse + Reciprocal Rank Fusion)

Combines semantic embedding search (ChromaDB) with keyword search (BM25Okapi).
Medical texts require both semantic understanding (symptom descriptions) and
exact keyword matching (drug names, lab thresholds, eponyms). Result lists are fused
using Reciprocal Rank Fusion (RRF) into a candidate shortlist.
"""

import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from config import config
from vector_store import LocalVectorStore


def tokenize_medical_text(text: str) -> List[str]:
    """
    Medical-tailored tokenizer for BM25 keyword matching.
    Preserves numbers, lab ranges, hyphenated drug names, and clinical abbreviations.
    """
    text = text.lower()
    # Replace non-alphanumeric characters (except hyphens and decimals in numbers) with whitespace
    tokens = re.findall(r'\b[a-z0-9]+(?:-[a-z0-9]+)*\b', text)
    # Remove extremely short tokens unless they are common medical abbreviations (e.g., mg, dl, iv, po, ct, mr, bp, hr)
    valid_short = {"mg", "g", "dl", "l", "iv", "po", "ct", "mr", "bp", "hr", "na", "k", "ca", "fe", "pr", "rr", "ab"}
    filtered_tokens = [t for t in tokens if len(t) > 2 or t in valid_short]
    return filtered_tokens


class HybridRetriever:
    """
    Hybrid Retriever executing parallel Dense + BM25 Sparse Retrieval
    and combining candidates using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, vector_store: Optional[LocalVectorStore] = None):
        if BM25Okapi is None:
            raise ImportError("rank_bm25 is required for hybrid retrieval. Install via `pip install rank-bm25`.")

        self.vector_store = vector_store if vector_store else LocalVectorStore()
        self.bm25_cache_file = config.BM25_CACHE_FILE
        
        self.bm25_engine: Optional[BM25Okapi] = None
        self.chunks_lookup: Dict[str, Dict[str, Any]] = {}
        self.chunk_ids_order: List[str] = []

        # Attempt to load cached BM25 index
        self._load_bm25_index()

    def build_bm25_index(self, chunks: List[Dict[str, Any]], force_rebuild: bool = False) -> None:
        """
        Tokenizes all text chunks and initializes the BM25 search index.
        Persists the tokenized corpus and lookup map to disk.
        """
        if self.bm25_engine is not None and not force_rebuild:
            print(f"[HybridRetriever] BM25 index already loaded with {len(self.chunk_ids_order)} passages.")
            return

        print(f"[HybridRetriever] Building BM25 index over {len(chunks)} chunks...")
        
        self.chunks_lookup = {c["chunk_id"]: c for c in chunks}
        self.chunk_ids_order = [c["chunk_id"] for c in chunks]

        # Tokenize corpus
        tokenized_corpus = []
        for c in tqdm(chunks, desc="Tokenizing Chunks for BM25"):
            combined_text = f"{c['heading_trail']} {c['text']}"
            tokenized_corpus.append(tokenize_medical_text(combined_text))

        self.bm25_engine = BM25Okapi(tokenized_corpus)

        # Cache to disk
        print(f"[HybridRetriever] Caching BM25 index to: {self.bm25_cache_file}")
        with open(self.bm25_cache_file, "wb") as f:
            pickle.dump({
                "chunk_ids_order": self.chunk_ids_order,
                "chunks_lookup": self.chunks_lookup,
                "bm25_engine": self.bm25_engine
            }, f)

        print("[HybridRetriever] BM25 indexing complete.")

    def _load_bm25_index(self) -> bool:
        """Loads cached BM25 index from disk if available."""
        if self.bm25_cache_file.exists():
            try:
                print(f"[HybridRetriever] Loading cached BM25 index from: {self.bm25_cache_file}")
                with open(self.bm25_cache_file, "rb") as f:
                    data = pickle.load(f)
                    self.chunk_ids_order = data["chunk_ids_order"]
                    self.chunks_lookup = data["chunks_lookup"]
                    self.bm25_engine = data["bm25_engine"]
                print(f"[HybridRetriever] Loaded BM25 index with {len(self.chunk_ids_order)} items.")
                return True
            except Exception as e:
                print(f"[HybridRetriever] Failed to load BM25 cache: {e}. Rebuild required.")
        return False

    def reciprocal_rank_fusion(self, dense_results: List[Dict[str, Any]],
                               sparse_results: List[Dict[str, Any]],
                               k: float = config.RRF_K,
                               top_n: int = config.SHORTLIST_SIZE) -> List[Dict[str, Any]]:
        """
        Executes Reciprocal Rank Fusion (RRF) over dense and sparse result lists.

        Formula: Score(d) = 1/(k + Rank_dense(d)) + 1/(k + Rank_sparse(d))
        """
        rrf_scores: Dict[str, float] = {}
        passage_map: Dict[str, Dict[str, Any]] = {}

        # 1. Process Dense Ranks
        for rank, item in enumerate(dense_results, start=1):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))
            passage_map[cid] = item

        # 2. Process Sparse BM25 Ranks
        for rank, item in enumerate(sparse_results, start=1):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))
            if cid not in passage_map:
                passage_map[cid] = item

        # 3. Sort by combined RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        fused_shortlist = []
        for cid in sorted_cids[:top_n]:
            entry = dict(passage_map[cid])
            entry["rrf_score"] = round(rrf_scores[cid], 5)
            fused_shortlist.append(entry)

        return fused_shortlist

    def retrieve(self, query: str,
                 dense_top_k: int = config.DENSE_TOP_K,
                 bm25_top_k: int = config.BM25_TOP_K,
                 shortlist_size: int = config.SHORTLIST_SIZE) -> List[Dict[str, Any]]:
        """
        Runs parallel Dense + Sparse BM25 searches and fuses candidate passages.
        Returns a fused shortlist ready for Cross-Encoder Reranking.
        """
        # 1. Dense Search via Vector Store
        dense_results = self.vector_store.dense_search(query=query, top_k=dense_top_k)

        # 2. Sparse Search via BM25
        sparse_results = []
        if self.bm25_engine is not None:
            tokenized_query = tokenize_medical_text(query)
            bm25_scores = self.bm25_engine.get_scores(tokenized_query)
            
            # Find top-k indices
            top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:bm25_top_k]

            for idx in top_bm25_indices:
                score = bm25_scores[idx]
                if score <= 0.0:
                    continue
                cid = self.chunk_ids_order[idx]
                chunk_data = self.chunks_lookup.get(cid, {})
                
                sparse_results.append({
                    "chunk_id": cid,
                    "text": chunk_data.get("text", ""),
                    "heading_trail": chunk_data.get("heading_trail", ""),
                    "page_numbers": chunk_data.get("page_numbers", []),
                    "bm25_score": round(float(score), 4)
                })

        # 3. Reciprocal Rank Fusion
        fused_candidates = self.reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=config.RRF_K,
            top_n=shortlist_size
        )

        return fused_candidates


if __name__ == "__main__":
    retriever = HybridRetriever()
    print("Hybrid Retriever Initialized.")
