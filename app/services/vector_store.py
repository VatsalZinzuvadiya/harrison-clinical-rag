"""
Embedding & Vector Store Repository Module

Supports high-speed Jina AI Cloud Embeddings API (jina-embeddings-v3) with
connection pooling and fallback to local Hugging Face SentenceTransformers (BGE family).
Persistent indexing and retrieval in local Chroma vector database.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import httpx
from tqdm import tqdm

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from app.core.config import config
from app.core.exceptions import VectorStoreError


class LocalVectorStore:
    """
    Vector Store repository utilizing ChromaDB with Jina AI API & local BGE model fallback.
    """

    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME,
                 persist_dir: Path = config.VECTOR_DB_DIR,
                 collection_name: str = config.CHROMA_COLLECTION_NAME):
        
        if chromadb is None:
            raise VectorStoreError("chromadb is required. Install via `pip install chromadb`.")

        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.model_name = model_name
        self.use_jina = config.USE_JINA_EMBEDDINGS and bool(config.JINA_API_KEY)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder: Optional[Any] = None

        # Persistent HTTP client for connection pooling (keeps TLS connection warm)
        self.http_client = httpx.Client(timeout=15.0, follow_redirects=True)

        if self.use_jina:
            print(f"[LocalVectorStore] Initialized using Jina AI Cloud API ('{config.JINA_EMBEDDING_MODEL}').")
        else:
            print(f"[LocalVectorStore] Loading local embedding model '{self.model_name}' on device: {self.device.upper()}")
            if SentenceTransformer is not None:
                self.embedder = SentenceTransformer(self.model_name, device=self.device)

        self.chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def is_indexed(self) -> bool:
        return self.collection.count() > 0

    def get_count(self) -> int:
        return self.collection.count()

    def _embed_texts_jina(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        url = "https://api.jina.ai/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {config.JINA_API_KEY}",
            "Content-Type": "application/json"
        }
        task = "retrieval.query" if is_query else "retrieval.passage"
        
        payload = {
            "model": config.JINA_EMBEDDING_MODEL,
            "task": task,
            "dimensions": config.JINA_EMBEDDING_DIM,
            "input": texts
        }

        try:
            response = self.http_client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json().get("data", [])
                return [item["embedding"] for item in data]
            else:
                print(f"[LocalVectorStore] Jina API returned HTTP {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[LocalVectorStore] Error calling Jina Embeddings API: {e}")

        return self._embed_texts_local(texts, is_query=is_query)

    def _embed_texts_local(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        if self.embedder is None:
            if SentenceTransformer is None:
                raise VectorStoreError("sentence-transformers is required for local embedding fallback.")
            self.embedder = SentenceTransformer(self.model_name, device=self.device)

        if is_query and "bge" in self.model_name.lower():
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]

        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()
        return embeddings

    def _embed_texts(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        if self.use_jina:
            return self._embed_texts_jina(texts, is_query=is_query)
        return self._embed_texts_local(texts, is_query=is_query)

    def build_index(self, chunks: List[Dict[str, Any]], force_rebuild: bool = False) -> None:
        current_count = self.collection.count()
        if current_count > 0 and not force_rebuild:
            print(f"[LocalVectorStore] Index already exists with {current_count} vectors. Skipping build.")
            return

        if force_rebuild and current_count > 0:
            print(f"[LocalVectorStore] Resetting collection '{self.collection_name}'...")
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

        batch_size = 128 if self.use_jina else config.EMBEDDING_BATCH_SIZE
        print(f"[LocalVectorStore] Embedding {len(chunks)} chunks (Using {'Jina Cloud API' if self.use_jina else 'Local Model'})...")

        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding Chunks"):
            batch = chunks[i : i + batch_size]
            
            texts_to_embed = [
                f"Section: {c['heading_trail']}\nContent: {c['text']}" for c in batch
            ]
            ids = [c["chunk_id"] for c in batch]
            
            embeddings = self._embed_texts(texts_to_embed, is_query=False)

            metadatas = [
                {
                    "page_numbers": json.dumps(c["page_numbers"]),
                    "heading_trail": c["heading_trail"],
                    "token_count": c["token_count"],
                    "text": c["text"]
                }
                for c in batch
            ]

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=[c["text"] for c in batch]
            )

        print(f"[LocalVectorStore] Vector store indexing complete. Total vectors: {self.collection.count()}")

    def dense_search(self, query: str, top_k: int = config.DENSE_TOP_K) -> List[Dict[str, Any]]:
        if self.collection.count() == 0:
            print("[LocalVectorStore] Warning: Vector store is empty.")
            return []

        query_embeddings = self._embed_texts([query], is_query=True)

        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if results and results.get("ids") and results["ids"][0]:
            ids = results["ids"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            for chunk_id, meta, dist in zip(ids, metadatas, distances):
                similarity_score = round(1.0 - float(dist), 4)
                page_nums = json.loads(meta.get("page_numbers", "[]")) if isinstance(meta.get("page_numbers"), str) else meta.get("page_numbers", [])
                
                formatted_results.append({
                    "chunk_id": chunk_id,
                    "text": meta.get("text", ""),
                    "heading_trail": meta.get("heading_trail", ""),
                    "page_numbers": page_nums,
                    "dense_score": similarity_score
                })

        return formatted_results
