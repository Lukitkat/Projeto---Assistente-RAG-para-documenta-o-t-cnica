"""Streamlit UI — Assistente RAG de Documentação Python.

Integra:
- src.pipeline.rag    (TODOs 1-3 implementados)
- src.pipeline.cache  (TODO 5 implementado)
- src.pipeline.routing (TODO 6 implementado)
- src.pipeline.tools  (TODO 4 implementado)
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# Adiciona o root do projeto no path para imports
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

load_dotenv()

import streamlit as st  # noqa: E402

from src.observability.trace import trace, log_event  # noqa: E402
from src.pipeline.cache import ExactCache, SemanticCache  # noqa: E402
from src.pipeline.rag import build_rag_pipeline  # noqa: E402
from src.pipeline.routing import classify_complexity  # noqa: E402
from src.pipeline.tools import search_python_definition, TOOLS  # noqa: E402


# ---------------------------------------------------------------- Streamlit UI
st.set_page_config(
    page_title="PyDoc Assistant — RAG with Groq",
    page_icon="🐍",
    layout="centered",
)

st.title("🐍 PyDoc Assistant")
st.caption("RAG Assistant for Python technical documentation · Powered by Groq + fastembed")


# Initialize pipeline and caches lazily
@st.cache_resource
def get_pipeline():
    return build_rag_pipeline(corpus_dir=str(_ROOT / "data" / "corpus"))


@st.cache_resource
def get_exact_cache():
    return ExactCache()


@st.cache_resource
def get_semantic_cache():
    return SemanticCache(threshold=0.93)


with st.spinner("🔄 Initializing RAG pipeline (first time might take ~30s)..."):
    pipeline = get_pipeline()
    exact_cache = get_exact_cache()
    semantic_cache = get_semantic_cache()


# Sidebar with metrics and debug controls
with st.sidebar:
    st.header("📊 Pipeline Metrics")
    st.metric("Indexed chunks", pipeline.collection.count())
    st.metric("Exact cache", exact_cache.stats()["size"])
    st.metric("Semantic cache", semantic_cache.stats()["size"])

    st.divider()
    st.subheader("⚙️ Controls")

    if st.button("🗑️ Clear caches"):
        get_exact_cache.clear()
        get_semantic_cache.clear()
        st.success("Caches cleared. Reload the page.")

    st.divider()
    st.subheader("🔧 Available tool")
    st.markdown(
        "**search_python_definition(term)**\n\n"
        "Fetches official definitions of Python terms.\n\n"
        "Examples: `list`, `decorator`, `generator`, `lambda`, `inheritance`"
    )

    st.divider()
    st.subheader("💡 Example questions")
    example_questions = [
        "What is a list in Python?",
        "Explain how decorators work",
        "Compare list comprehension with for loop",
        "How does inheritance work in Python?",
        "What is a generator?",
    ]
    for q in example_questions:
        if st.button(q, key=f"ex_{q[:20]}"):
            st.session_state["query_input"] = q


# Main chat interface
st.divider()

# Setup tabs for RAG and direct tool access
tab_rag, tab_tool = st.tabs(["💬 Ask the corpus", "🔧 Consult glossary"])

with tab_rag:
    query_default = st.session_state.get("query_input", "")
    query = st.text_input(
        "Your question about Python:",
        placeholder="Ex: How do decorators work in Python?",
        value=query_default,
        key="rag_query",
    )

    if query:
        with trace("query_handle", query=query) as ctx:
            trace_id = ctx["trace_id"]

            # check exact cache first
            cached = exact_cache.get(query)
            if cached:
                st.success("⚡ Cache hit (exact)")
                st.write(cached)
                log_event("cache_hit", trace_id=trace_id, layer="exact")
                st.stop()

            # fallback to semantic cache if possible
            try:
                cached = semantic_cache.get(query)
            except NotImplementedError:
                cached = None

            if cached:
                st.success("🧠 Cache hit (semantic — paraphrase detected)")
                st.write(cached)
                log_event("cache_hit", trace_id=trace_id, layer="semantic")
                st.stop()

            # determine the appropriate model based on query complexity
            try:
                decision = classify_complexity(query)
                complexity_emoji = "🟢" if decision.complexity == "simple" else "🔴"
                st.info(
                    f"{complexity_emoji} **Routing:** `{decision.complexity}` → "
                    f"`{decision.model}`\n\n_{decision.reason}_"
                )
                log_event("route_decision", trace_id=trace_id, **decision.__dict__)
            except NotImplementedError:
                st.warning("Routing not implemented. Using default model.")

            # run the full RAG pipeline
            with st.spinner("🔍 Searching the corpus and generating response..."):
                try:
                    result = pipeline.answer(query)
                except NotImplementedError as e:
                    st.error(f"Pipeline not implemented: {e}")
                    st.stop()
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.stop()

            # display the generated answer
            st.markdown("### 📝 Response")
            st.write(result["answer"])

            if result.get("sources"):
                with st.expander("📚 Cited sources"):
                    for source, page in result["sources"]:
                        st.write(f"- `{source}` · page {page}")

            # store the result in both caches
            exact_cache.put(query, result["answer"])
            semantic_cache.put(query, result["answer"])
            log_event("answer_generated", trace_id=trace_id, sources=len(result.get("sources", [])))


with tab_tool:
    st.markdown("### 🔧 Direct Python glossary lookup")
    st.caption("Fetches precise definitions of Python terms with examples and official references.")

    termo = st.text_input(
        "Enter a Python term:",
        placeholder="Ex: list, decorator, generator, lambda...",
        key="tool_termo",
    )

    termos_rapidos = ["list", "function", "dictionary", "class", "decorator", "generator", "lambda", "inheritance"]
    cols = st.columns(4)
    for i, t in enumerate(termos_rapidos):
        if cols[i % 4].button(t, key=f"termo_{t}"):
            termo = t

    if termo:
        resultado = search_python_definition(termo)
        st.markdown(resultado)


st.divider()
st.caption(
    "🐍 **PyDoc Assistant** · RAG with Groq (llama-3.1-8b-instant / llama-3.3-70b-versatile) "
    "+ fastembed embeddings (BAAI/bge-small-en-v1.5) · Streamlit"
)
