import streamlit as st
import os
import tempfile

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="DocMind · AI Document Intelligence",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,400;0,500;1,400&display=swap');

:root {
    --bg:       #0d0f14;
    --surface:  #161920;
    --border:   #252830;
    --accent:   #c8f04a;
    --accent2:  #4af0c8;
    --muted:    #555b6b;
    --text:     #e8eaf0;
    --text-dim: #9299ab;
    --radius:   12px;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── FIXED SIDEBAR — never collapses ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 280px !important;
    max-width: 280px !important;
    transform: none !important;
    visibility: visible !important;
    pointer-events: auto !important;
}
[data-testid="stSidebar"] > div:first-child {
    transform: none !important;
    visibility: visible !important;
}
/* Hide the collapse/expand arrow button */
[data-testid="collapsedControl"],
button[kind="header"],
[data-testid="stSidebarNavCollapseIcon"],
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

.block-container { padding: 2rem 2.5rem !important; max-width: 900px !important; }

.hero { display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem; }
.hero-icon { font-size: 2.8rem; line-height: 1; }
.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
    color: var(--text) !important;
    margin: 0 !important;
}
.hero-subtitle {
    font-size: 0.78rem;
    color: var(--text-dim);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 2px;
}
.accent { color: var(--accent); }

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-ready { background: #1e2f12; color: var(--accent); border: 1px solid #3a5a1a; }
.badge-wait  { background: #1e1e2f; color: #a0a8ff;        border: 1px solid #3a3a6a; }

.msg-wrapper { margin-bottom: 1.4rem; }
.msg-user { text-align: right; }
.msg-user .bubble {
    display: inline-block;
    background: #1a2e0a;
    border: 1px solid #3a5a1a;
    border-radius: var(--radius) var(--radius) 2px var(--radius);
    padding: 0.75rem 1rem;
    max-width: 75%;
    font-size: 0.875rem;
    color: #d4f577 !important;
    text-align: left;
}
.msg-ai .bubble {
    display: inline-block;
    background: #1c1f28;
    border: 1px solid #2e3340;
    border-radius: var(--radius) var(--radius) var(--radius) 2px;
    padding: 0.75rem 1rem;
    max-width: 85%;
    font-size: 0.875rem;
    line-height: 1.7;
    color: #e8eaf0 !important;
    text-align: left;
}
.msg-label {
    font-size: 0.68rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.source-grid { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.8rem; }
.source-card {
    background: #0d1a1f;
    border: 1px solid #1a3040;
    border-radius: 8px;
    padding: 5px 10px;
    font-size: 0.7rem;
    color: #4af0c8 !important;
}

.divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}

.stTextInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(200,240,74,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: #c8f04a !important;
    color: #0d0f14 !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover  { background-color: #b8e03a !important; color: #0d0f14 !important; }
.stButton > button:focus  { background-color: #c8f04a !important; color: #0d0f14 !important; box-shadow: none !important; }
.stButton > button:active { background-color: #a8d02a !important; color: #0d0f14 !important; }
.stButton > button p,
.stButton > button span,
.stButton > button div { color: #0d0f14 !important; font-weight: 700 !important; }

.stFormSubmitButton > button {
    background-color: #c8f04a !important;
    color: #0d0f14 !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
}
.stFormSubmitButton > button p,
.stFormSubmitButton > button span { color: #0d0f14 !important; font-weight: 700 !important; }
.stFormSubmitButton > button:hover { background-color: #b8e03a !important; color: #0d0f14 !important; }

.sidebar-label {
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
    font-family: 'Syne', sans-serif;
}
.stat-box {
    background: #0d0f14;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.8rem;
    color: var(--text) !important;
}
.stat-box b { color: var(--accent); }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for key, val in {
    "messages": [],
    "vectorstore": None,
    "doc_name": None,
    "chunk_count": 0,
    "page_count": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return MistralAIEmbeddings(model="mistral-embed")

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(model="mistral-small-2506")

def build_vectorstore(pdf_path: str):
    loader   = PyPDFLoader(pdf_path)
    docs     = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks   = splitter.split_documents(docs)
    vs       = Chroma.from_documents(documents=chunks, embedding=get_embedding_model())
    return vs, len(docs), len(chunks)

def answer_query(query: str, vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant.
Use ONLY the provided context to answer the question.
If the answer is not present in the context, say: "I could not find the answer in the document."
"""),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
    ])
    docs     = retriever.invoke(query)
    context  = "\n\n".join([d.page_content for d in docs])
    response = get_llm().invoke(prompt.invoke({"context": context, "question": query}))
    return response.content, docs


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-label">📂 Upload Document</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded:
        if st.button("📖  Process Document"):
            with st.spinner("Embedding document …"):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name

                    vs, pages, chunks = build_vectorstore(tmp_path)
                    os.unlink(tmp_path)

                    st.session_state.vectorstore = vs
                    st.session_state.doc_name    = uploaded.name
                    st.session_state.chunk_count = chunks
                    st.session_state.page_count  = pages
                    st.session_state.messages    = []
                    st.success("Document ready!")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if st.session_state.doc_name:
        st.markdown('<div class="sidebar-label">📊 Stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="stat-box">📄 <b>{st.session_state.doc_name}</b></div>
            <div class="stat-box">Pages · <b>{st.session_state.page_count}</b></div>
            <div class="stat-box">Chunks · <b>{st.session_state.chunk_count}</b></div>
        """, unsafe_allow_html=True)
        if st.button("🗑  Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    else:
        st.markdown(
            '<p style="font-size:0.78rem;color:#555b6b;">No document loaded yet.</p>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.7rem;color:#555b6b;">Mistral AI · LangChain · ChromaDB</p>',
        unsafe_allow_html=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-icon">📖</div>
  <div>
    <div class="hero-title">Doc<span class="accent">Mind</span></div>
    <div class="hero-subtitle">AI Document Intelligence · RAG · Mistral</div>
  </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.vectorstore:
    st.markdown(
        f'<span class="badge badge-ready">✓ {st.session_state.doc_name} ready</span>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<span class="badge badge-wait">⟳ Upload a document to begin</span>',
        unsafe_allow_html=True,
    )

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# Chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-wrapper msg-user">
            <div class="msg-label">You</div>
            <div class="bubble">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        sources_html = ""
        if msg.get("sources"):
            cards = "".join(
                f'<div class="source-card">📄 page {s.metadata.get("page", "?") + 1}</div>'
                for s in msg["sources"]
            )
            sources_html = f'<div class="source-grid">{cards}</div>'

        st.markdown(f"""
        <div class="msg-wrapper msg-ai">
            <div class="msg-label">DocMind</div>
            <div class="bubble">{msg["content"]}{sources_html}</div>
        </div>
        """, unsafe_allow_html=True)

# Input form
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "Ask",
            placeholder="Ask anything about your document …",
            label_visibility="collapsed",
            disabled=st.session_state.vectorstore is None,
        )
    with col2:
        submitted = st.form_submit_button(
            "Send",
            disabled=st.session_state.vectorstore is None,
        )

if submitted and query.strip():
    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner("Thinking …"):
        try:
            answer, source_docs = answer_query(query, st.session_state.vectorstore)
        except Exception as e:
            answer      = f"⚠️ Error: {e}"
            source_docs = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": source_docs,
    })
    st.rerun()

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0; color: #555b6b; font-size:0.85rem;">
        <div style="font-size:2.5rem; margin-bottom:0.5rem;">📖</div>
        Upload a PDF on the left, process it,<br>then ask anything about it here.
    </div>
    """, unsafe_allow_html=True)