"""
Cross-Encoder Reranker Module

Re-scores fused retrieval candidates using a local Cross-Encoder transformer model.
Cross-encoders perform full joint cross-attention over (query, passage) pairs, achieving
substantially higher precision than bi-encoder embedding similarity alone.
"""

import os
from typing import List, Dict, Any, Optional
import torch

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

from app.core.config import config
from app.core.exceptions import RerankerError


class LocalReranker:
    """
    Cross-Encoder Reranker evaluating query-passage relevance on a shortlist.
    Executes locally on GPU (if available) or optimized multi-core CPU.
    """

    def __init__(self, model_name: str = config.RERANKER_MODEL_NAME):
        if CrossEncoder is None:
            raise RerankerError("sentence-transformers CrossEncoder is required. Install via `pip install sentence-transformers`.")

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cpu":
            num_threads = min(8, os.cpu_count() or 4)
            torch.set_num_threads(num_threads)
            print(f"[LocalReranker] Loading Cross-Encoder '{self.model_name}' on CPU ({num_threads} threads).")
        else:
            print(f"[LocalReranker] Loading Cross-Encoder '{self.model_name}' on GPU (CUDA).")

        try:
            self.model = CrossEncoder(self.model_name, device=self.device)
        except Exception as e:
            raise RerankerError(f"Failed to load CrossEncoder model '{self.model_name}': {e}")

    def rerank(self, query: str,
               candidates: List[Dict[str, Any]],
               top_n: int = config.TOP_N_RERANKED) -> List[Dict[str, Any]]:
        """
        Re-scores a candidate shortlist (~10-15 items) and returns the top N passages.
        """
        if not candidates:
            return []

        shortlist = candidates[:15]
        pairs = []
        for c in shortlist:
            content_snippet = c['text'][:1000]
            passage_text = f"Heading: {c['heading_trail']}\nContent: {content_snippet}"
            pairs.append((query, passage_text))

        print(f"[LocalReranker] Re-scoring {len(pairs)} shortlist candidates...")
        try:
            with torch.inference_mode():
                scores = self.model.predict(
                    pairs,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
        except Exception as e:
            print(f"[LocalReranker] Error during Cross-Encoder prediction: {e}")
            return candidates[:top_n]

        reranked_candidates = []
        for c, score in zip(shortlist, scores):
            entry = dict(c)
            entry["rerank_score"] = round(float(score), 4)
            reranked_candidates.append(entry)

        reranked_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        selected = reranked_candidates[:top_n]
        print(f"[LocalReranker] Selected Top-{len(selected)} reranked context passages.")
        return selected
