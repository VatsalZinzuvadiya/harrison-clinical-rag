# Starlette 0.45+ Compatibility Patch for Streamlit
import starlette.middleware.gzip as _gz
if not hasattr(_gz, "DEFAULT_EXCLUDED_CONTENT_TYPES"):
    setattr(_gz, "DEFAULT_EXCLUDED_CONTENT_TYPES", ("text/html", "text/css", "text/javascript", "text/plain", "application/json"))
if not hasattr(_gz, "IdentityResponder"):
    setattr(_gz, "IdentityResponder", type("IdentityResponder", (), {}))

import time
import httpx
import streamlit as st

# Page Configuration & Custom CSS Styling
st.set_page_config(
    page_title="Harrison's Clinical Decision Support (RAG)",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark glassmorphism aesthetic
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stAppHeader {
        background-color: transparent;
    }
    .medical-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .citation-card {
        background: rgba(0, 168, 204, 0.08);
        border-left: 4px solid #00a8cc;
        border-radius: 4px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .metric-box {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    h1, h2, h3 {
        color: #e0f2fe;
    }
</style>
""", unsafe_allow_html=True)

API_BASE_URL = "http://localhost:8000"


def check_api_health():
    try:
        resp = httpx.get(f"{API_BASE_URL}/health", timeout=2.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


# Sidebar Navigation & Settings
st.sidebar.title("🩺 Harrison's Clinical RAG")
st.sidebar.markdown("---")

health_data = check_api_health()

if health_data:
    st.sidebar.success("✅ API Server Online")
    vec_data = health_data.get("vector_store", {})
    bm25_data = health_data.get("bm25_index", {})
    ollama_data = health_data.get("ollama_llm", {})

    st.sidebar.markdown(f"**Vector Store**: `{vec_data.get('count', 0)}` chunks ({vec_data.get('device', 'cpu').upper()})")
    st.sidebar.markdown(f"**BM25 Index**: `{bm25_data.get('passages_count', 0)}` passages")
    
    if ollama_data.get("available"):
        st.sidebar.success(f"🤖 Ollama LLM Active (`{health_data.get('ollama_llm', {}).get('models', ['llama3.2'])[0]}`)")
    else:
        st.sidebar.warning("⚠️ Ollama LLM Offline (Fallback Mode)")
else:
    st.sidebar.error("❌ API Server Offline (`localhost:8000`)")
    st.sidebar.info("Launch API via: `uvicorn api:app --port 8000`")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Retrieval Tuning")
shortlist_k = st.sidebar.slider("Hybrid Shortlist Pool Size", min_value=10, max_value=50, value=25, step=5)
top_n_rerank = st.sidebar.slider("Top Reranked Context Chunks", min_value=3, max_value=12, value=6, step=1)


# Main Content Header
st.title("Harrison's Principles of Internal Medicine")
st.caption("Production-Grade Local RAG Engine for Evidence-Based Clinical Decision Support")

st.markdown("---")

# Quick Sample Patient Presentations
st.subheader("💡 Sample Patient Cases")
col1, col2, col3 = st.columns(3)

sample_case_1 = "65-year-old male with hypertension presenting with sudden retrosternal crushing chest pain radiating to left jaw, diaphoresis, and shortness of breath."
sample_case_2 = "42-year-old female presenting with 3-week history of fatigue, weight loss, heat intolerance, palpitation, and fine hand tremors."
sample_case_3 = "55-year-old male presenting with severe epigastric pain radiating straight to the back, nausea, vomiting, and elevated serum lipase."

selected_sample = None
if col1.button("🫀 Chest Pain Presentation"):
    selected_sample = sample_case_1
if col2.button("🦋 Thyroid Presentation"):
    selected_sample = sample_case_2
if col3.button("🩸 Epigastric Pain Presentation"):
    selected_sample = sample_case_3

# Input Text Area
query_input = st.text_area(
    "Enter Patient Symptoms or Clinical Question:",
    value=selected_sample if selected_sample else "",
    height=120,
    placeholder="e.g. Describe patient symptoms, laboratory findings, or clinical diagnostic questions..."
)

submit_btn = st.button("🔍 Submit Clinical Query", type="primary", use_container_width=True)

if submit_btn and query_input.strip():
    st.markdown("---")
    with st.spinner("Executing Hybrid Retrieval, Reranking, and Grounded Clinical Generation..."):
        t_start = time.time()
        
        # Try query via FastAPI backend
        try:
            resp = httpx.post(
                f"{API_BASE_URL}/query",
                json={
                    "question": query_input.strip(),
                    "top_n": top_n_rerank,
                    "include_sources": True
                },
                timeout=90.0
            )

            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "")
                sources = data.get("sources", [])
                latency = data.get("latency", {})

                # Render Answer Section
                st.subheader("📋 Clinical Guidance")
                st.markdown(answer)

                st.markdown("---")
                # Render Metrics
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Retrieval Latency", f"{latency.get('retrieval_ms', 0):.1f} ms")
                mcol2.metric("Rerank Latency", f"{latency.get('rerank_ms', 0):.1f} ms")
                mcol3.metric("LLM Generation", f"{latency.get('generation_ms', 0):.1f} ms")
                mcol4.metric("Total Latency", f"{latency.get('total_ms', 0):.1f} ms")

                st.markdown("---")
                # Render Cited Passages
                st.subheader("📚 Cited Textbook Passages & Sources")
                
                for idx, src in enumerate(sources, start=1):
                    pages_str = ", ".join(str(p) for p in src.get("page_numbers", []))
                    heading = src.get("heading_trail", "Harrison's Text")
                    r_score = src.get("rerank_score", "N/A")
                    
                    with st.expander(f"Source [{idx}] — Page(s) {pages_str} | Section: {heading}"):
                        st.markdown(f"**Reranker Relevance Score**: `{r_score}`")
                        st.markdown(f"**Text Excerpt**:\n>{src.get('text_preview')}")
            else:
                st.error(f"API Error {resp.status_code}: {resp.text}")

        except Exception as e:
            st.error(f"Could not connect to API server at `{API_BASE_URL}`: {e}")
            st.info("Make sure the API server is running (`python api.py` or `uvicorn api:app`).")
