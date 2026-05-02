"""
📄 Document Manager — View ingested documents, vector store stats, and manage knowledge base.
"""

import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import AppConfig
from app.knowledge_base import MaintenanceKnowledgeBase
from app.vector_store import VectorStore
from app.document_loader import DocumentLoader

st.set_page_config(page_title="Document Manager", page_icon="📄", layout="wide")


@st.cache_resource
def init_components():
    kb = MaintenanceKnowledgeBase()
    vs = VectorStore()
    doc_loader = DocumentLoader(vs, kb)
    return {'kb': kb, 'vs': vs, 'doc_loader': doc_loader}


components = init_components()

st.markdown("""
<style>
    .gradient-text {
        background: linear-gradient(90deg, #6C63FF, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 2rem;
    }
    .doc-card {
        background: rgba(108, 99, 255, 0.06);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 10px;
        padding: 16px;
        margin: 6px 0;
    }
    .stat-big {
        font-size: 2.5rem;
        font-weight: 800;
        color: #6C63FF;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="gradient-text">📄 Document Manager</p>', unsafe_allow_html=True)
st.caption("Manage your maintenance knowledge base — ingested documents, vector store stats, and semantic search testing")

# ─── Stats ───
stats = components['vs'].get_stats()
docs = components['kb'].get_ingested_documents()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f'<div class="doc-card"><p class="stat-big">{stats["document_chunks"]}</p>'
        f'<p style="color:#888">Document Chunks</p></div>',
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f'<div class="doc-card"><p class="stat-big">{len(docs)}</p>'
        f'<p style="color:#888">Documents</p></div>',
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        f'<div class="doc-card"><p class="stat-big">{stats["entity_entries"]}</p>'
        f'<p style="color:#888">Entity Memories</p></div>',
        unsafe_allow_html=True
    )
with col4:
    st.markdown(
        f'<div class="doc-card"><p style="font-size:1rem; color:#4ECDC4; margin:0">'
        f'{stats["embedding_model"]}</p>'
        f'<p style="color:#888">Embedding Model</p></div>',
        unsafe_allow_html=True
    )

st.divider()

# ─── Upload Section ───
st.subheader("📤 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload maintenance manuals, SOPs, equipment documentation",
    type=['pdf', 'docx', 'txt', 'md'],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        if not components['kb'].is_document_ingested(uploaded_file.name):
            upload_dir = AppConfig.UPLOAD_DIR
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"📥 Ingesting {uploaded_file.name}..."):
                chunks = components['doc_loader'].ingest_file(file_path)
                if chunks > 0:
                    st.success(f"✅ **{uploaded_file.name}**: {chunks} chunks ingested")
                else:
                    st.warning(f"⚠️ No text extracted from {uploaded_file.name}")
        else:
            st.info(f"ℹ️ {uploaded_file.name} already in knowledge base")

# ─── Paste Text ───
st.subheader("📝 Paste Text Directly")
with st.expander("Paste maintenance content to add to knowledge base"):
    text_name = st.text_input("Source Name", placeholder="e.g., Pump SOP v2.1")
    text_content = st.text_area("Content", height=200, placeholder="Paste maintenance procedures, SOPs, or equipment documentation here...")

    if st.button("Ingest Text", type="primary"):
        if text_name and text_content:
            chunks = components['doc_loader'].ingest_text_directly(text_content, text_name)
            if chunks > 0:
                st.success(f"✅ **{text_name}**: {chunks} chunks ingested")
                st.rerun()
            else:
                st.warning("No chunks created. Text may be too short or already ingested.")
        else:
            st.error("Both source name and content are required.")

st.divider()

# ─── Ingested Documents List ───
st.subheader("📚 Ingested Documents")

if docs:
    for doc in docs:
        st.markdown(
            f'<div class="doc-card">'
            f'📄 <strong style="color: #6C63FF">{doc["filename"]}</strong><br>'
            f'<span style="color: #4ECDC4">Type:</span> {doc["file_type"]} · '
            f'<span style="color: #4ECDC4">Chunks:</span> {doc["chunk_count"]} · '
            f'<span style="color: #888">Ingested: {doc["ingested_at"]}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
else:
    st.info("No documents in the knowledge base yet.")

st.divider()

# ─── Semantic Search Test ───
st.subheader("🔍 Test Semantic Search")
st.caption("Search your knowledge base to see what the RAG retriever finds")

search_query = st.text_input("Search Query", placeholder="e.g., pump vibration causes")

if search_query:
    results = components['vs'].query(search_query, n_results=5)

    if results:
        st.markdown(f"**Found {len(results)} relevant chunks:**")
        for i, result in enumerate(results, 1):
            relevance = (1 - result.get('distance', 0)) * 100
            source = result.get('metadata', {}).get('source', 'Unknown')

            st.markdown(
                f'<div class="doc-card">'
                f'<strong>#{i}</strong> · 📄 {source} · '
                f'<span style="color: #4ECDC4">Relevance: {relevance:.0f}%</span><br><br>'
                f'{result["content"][:300]}{"..." if len(result["content"]) > 300 else ""}'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.info("No results found. Try uploading more documents or adjusting your query.")

st.divider()

# ─── Danger Zone ───
with st.expander("⚠️ Danger Zone"):
    st.warning("These actions are destructive and cannot be undone.")
    if st.button("🗑️ Clear All Documents", type="secondary"):
        components['vs'].delete_collection()
        st.success("Vector store cleared. Documents will need to be re-ingested.")
        st.rerun()
