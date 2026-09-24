# Harrison's Principles of Internal Medicine - Enterprise Clinical Decision Support System (RAG)

An enterprise-grade, zero-cost, privacy-focused Clinical Decision Support RAG Application built on **Harrison's Principles of Internal Medicine (21st Edition)**. Designed with Clean Architecture (Domain-Driven Design), sub-second hybrid search, multi-model LLM failover, and interactive PDF page viewing with dynamic text highlighting.

---

## 🏛 Architecture & Project Structure

The project follows a **Layered Clean Architecture**:

```
clinical_rag_harrison/
├── app/
│   ├── api/                  # API Controllers & Route Handlers
│   │   ├── endpoints/
│   │   │   ├── health.py     # System diagnostic checks & latency status
│   │   │   ├── pdf.py        # PDF rendering & yellow-highlight endpoints
│   │   │   └── query.py      # Clinical inquiry & RAG workflow controller
│   │   ├── dependencies.py   # Dependency injection container & singleton pre-loader
│   │   └── router.py         # Master API router
│   ├── core/                 # Core Infrastructure & Configuration
│   │   ├── config.py         # Type-safe environment config (Pydantic BaseSettings)
│   │   ├── exceptions.py     # Custom domain exception taxonomy
│   │   └── logging.py        # Structured JSON/Console logger
│   ├── schemas/              # Pydantic Contracts & Data Transfer Objects (DTOs)
│   │   ├── health.py         # Health check response DTOs
│   │   └── query.py          # Request, Response, Citation, & Latency DTOs
│   ├── services/             # Core Business Logic & Domain Services
│   │   ├── chunker.py        # Structure-aware topic & token chunking service
│   │   ├── generator.py      # OpenRouter / Ollama LLM provider with failover
│   │   ├── ingestor.py       # PyMuPDF visual layout & typography parser
│   │   ├── pdf_service.py    # PyMuPDF 15ms PNG renderer with yellow highlighting
│   │   ├── reranker.py       # Cross-Encoder MS-MARCO Reranker (PyTorch multi-threaded)
│   │   ├── retriever.py      # Hybrid Dense + BM25Okapi + RRF engine
│   │   └── vector_store.py   # ChromaDB repository with Jina v3 / BGE embeddings
│   └── web/
│       └── index.html        # Glassmorphic Medical UI with PDF Page Modal Viewer
├── main.py                   # FastAPI Application Entrypoint
├── api.py                    # Legacy Proxy Entrypoint (Backward Compatibility)
└── requirements.txt          # Python Dependencies
```

---

## 🚀 Key Features

1. **Sub-Second Hybrid Search Engine**:
   - **Dense Vectors**: `jina-embeddings-v3` (384 Matryoshka dimensions via socket-pooled HTTP client). Local `bge-small-en-v1.5` fallback.
   - **Sparse Lexical Search**: BM25Okapi for precise medical terminology and abbreviation matching.
   - **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse ranks dynamically.

2. **PyTorch Multi-Threaded Cross-Encoder Reranker**:
   - Reranks top candidates using `ms-marco-MiniLM-L-6-v2`.
   - Accelerated via CPU thread tuning (`torch.set_num_threads(8)`), text truncation, and shortlisting (~1.8s execution).

3. **Multi-Model LLM Failover**:
   - Primary: `openai/gpt-oss-120b` (OpenRouter API)
   - Secondary: `qwen/qwen3.8-27b`
   - Tertiary: `openai/gpt-oss-20b`
   - Local Fallback: Ollama (`llama3.2:latest`)

4. **Interactive In-Browser PDF Viewer & Yellow Highlighting**:
   - Renders Harrison's textbook pages as 150 DPI PNGs in under **15ms**.
   - Applies precise yellow text highlight annotations on matching textbook text.

5. **Structured 4-Section Clinical Output**:
   - **Direct Answer & Diagnostic Summary**
   - **Key Symptoms, Signs & Diagnostic Criteria**
   - **Clinical Decision Support & Workup Guidelines**
   - **Grounded Harrison Page References**

---

## ⚡ Quick Start & Execution

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Launching the Application
Run via the main entrypoint:
```bash
python main.py
```
Or via the legacy entrypoint:
```bash
python api.py
```

### 3. Access the Dashboard & API Docs
- **Web UI Dashboard**: Navigate to `http://localhost:8000/`
- **Swagger API Docs**: Navigate to `http://localhost:8000/docs`

---

## 🛠 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the Glassmorphic Medical UI Dashboard |
| `POST` | `/api/v1/query` | Submits a clinical inquiry; returns grounded answer, citations, and latency metrics |
| `GET` | `/api/v1/pdf/page/{page_num}` | Renders a textbook page image with optional `highlight` text snippet |
| `GET` | `/api/v1/pdf/file` | Serves the original raw PDF file |
| `GET` | `/api/v1/health` | Diagnostic status check for ChromaDB, BM25, and LLM APIs |
