# 🩺 Production-Grade Local Clinical RAG Engine
### *Harrison's Principles of Internal Medicine (20th Edition)*

A fast, accurate, scalable, zero-cost Retrieval-Augmented Generation (RAG) system built in Python to assist physicians and medical researchers. Answers clinical questions grounded strictly in textbook passages with exact page citations.

---

## 🏛️ Project Architecture & Folder Structure

```
clinical_rag_harrison/
├── config.py             # Central configuration (chunking, thresholds, models, API ports)
├── requirements.txt      # Pinned, 100% free open-source dependencies
├── ingest.py             # PyMuPDF parser & Font Inspection tool (--inspect-fonts)
├── chunker.py            # Structure-aware heading detector & token sliding-window chunker
├── vector_store.py       # ChromaDB local vector store & BGE embedding manager
├── hybrid_retriever.py   # Hybrid BM25 + Dense vector retrieval with Reciprocal Rank Fusion (RRF)
├── reranker.py           # Cross-Encoder (BGE Reranker) candidate rescoring module
├── generator.py          # Local LLM integration (Ollama API / fallback synthesis)
├── build_index.py        # CLI runner to execute end-to-end PDF ingestion and index creation
├── api.py                # FastAPI REST server (/query, /health) with startup model caching
├── ui.py                 # Streamlit clinical dashboard with citation cards & latency breakdown
└── README.md             # Complete setup, tuning reference & scaling guide
```

---

## 🚀 Quick Setup & Execution Guide

### Step 1: Install Dependencies
Create a clean virtual environment and install requirements:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Add PDF File
Place your 2000+ page medical textbook PDF at:
`clinical_rag_harrison/data/harrison.pdf`

*(Or specify a custom location via `--pdf /path/to/your.pdf` or `config.py`).*

### Step 3: Inspect Typography & Tune Heading Detection (Optional)
Run the font inspection tool on sample pages to view heading font sizes:
```bash
python ingest.py --inspect-fonts --sample-pages 50
```
*If necessary, adjust `MIN_HEADING_FONT_SIZE` in `config.py` based on the output.*

### Step 4: Build the Index
Process the PDF into structure-aware chunks, compute embeddings, build ChromaDB and BM25 indices:
```bash
python build_index.py
```
> **Note**: Intermediary chunks are cached to `data/cache/processed_chunks.json`. Re-running `build_index.py` loads instantly from cache unless `--force-rebuild` is passed.

### Step 5: (Optional) Start Local LLM Runtime (Ollama)
Install and run [Ollama](https://ollama.ai) locally:
```bash
ollama run llama3.2
```
*(If Ollama is not running, the application gracefully falls back to a direct grounded synthesis format).*

### Step 6: Launch FastAPI Backend Server
```bash
python api.py
# Server will start on http://0.0.0.0:8000
```
Verify status via browser: `http://localhost:8000/health`

### Step 7: Launch Streamlit Clinical Interface
In a separate terminal window:
```bash
python run_ui.py
```
*(Or `streamlit run ui.py` via entrypoint launcher)*

---

## 🛠️ Tuning & Troubleshooting Reference

### 1. Heading Detection Thresholds
- **Problem**: Heading breadcrumbs look messy or miss section titles.
- **Solution**: Run `python ingest.py --inspect-fonts`. Inspect the font size table. Update `MIN_HEADING_FONT_SIZE` (default: `12.0`) or `REQUIRE_BOLD_FOR_SMALL_HEADINGS` in `config.py`.

### 2. Retrieval Misses Obvious Answers
- **Exact terms (drug names, lab values) missed**: Increase `BM25_TOP_K` (e.g. from 30 to 50) or adjust `RRF_K` (default 60).
- **Semantic concepts missed**: Increase `DENSE_TOP_K` (e.g. from 30 to 50).
- **Truncated context**: Increase `TARGET_CHUNK_TOKENS` (default: 400) or `TOKEN_OVERLAP` (default: 75 / ~18%).

### 3. Latency Optimization
- **Reranker taking too long**: Reduce `SHORTLIST_SIZE` (e.g. from 25 down to 15-20). The Cross-Encoder only processes candidate pairs in the shortlist.
- **LLM generation latency**: Use a smaller local model in Ollama (`llama3.2:1b` or `mistral:7b-instruct-q4`), or lower `TOP_N_RERANKED` (e.g. 4-6 passages).

### 4. Memory Management (RAM / VRAM)
- **GPU Out-of-Memory (OOM)**: Lower `EMBEDDING_BATCH_SIZE` in `config.py` (e.g. from 64 to 16 or 32).
- **CPU execution**: PyTorch automatically detects CUDA; if GPU is unavailable, it runs seamlessly on CPU.

---

## 📈 Scaling Guide (Extending the Architecture)

This application is built with modular abstraction layers designed for linear scaling without rewriting core logic:

1. **Bigger Corpus (Multiple Medical Books / 100k+ Pages)**:
   - Replace Chroma local mode with a vector database cluster (**Qdrant** or **Milvus**).
   - Replace memory-bound BM25 with disk-backed **Pyserini / Lucene** sparse indexing.

2. **High Concurrent User Traffic**:
   - Deploy FastAPI with multiple worker processes: `uvicorn api:app --workers 4`.
   - Offload local LLM inference to a dedicated **vLLM** GPU inference server (`/v1/completions` endpoint) supporting high-throughput continuous batching.

3. **GPU Acceleration**:
   - Both embedding generation (`vector_store.py`) and reranking (`reranker.py`) leverage PyTorch `cuda` devices automatically. Adding additional GPUs accelerates indexing linearly.
