"""
🔧 Maintenance Agentic AI — Streamlit Chat Interface
RAG-powered industrial maintenance assistant with contextual memory.
"""

import asyncio
import os
import sys
import logging
import streamlit as st
from datetime import datetime

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import AppConfig, VectorStoreConfig
from app.llm_provider import LLMProvider
from app.knowledge_base import MaintenanceKnowledgeBase
from app.vector_store import VectorStore
from app.document_loader import DocumentLoader
from app.rag_chain import RAGChain
from app.memory import ConversationMemory, EntityMemory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Page Configuration ───
st.set_page_config(
    page_title="Maintenance AI Assistant",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS for premium look ───
st.markdown("""
<style>
    /* Main background gradient overlay */
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1A1F2E 50%, #0E1117 100%);
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 8px;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1F2E 0%, #0E1117 100%);
        border-right: 1px solid rgba(108, 99, 255, 0.2);
    }

    /* Headers with gradient text */
    .gradient-text {
        background: linear-gradient(90deg, #6C63FF, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2rem;
    }

    /* Source cards */
    .source-card {
        background: rgba(108, 99, 255, 0.08);
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 0.85rem;
    }

    /* Stats cards */
    .stat-card {
        background: rgba(78, 205, 196, 0.08);
        border: 1px solid rgba(78, 205, 196, 0.2);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin: 6px 0;
    }
    .stat-card h3 {
        margin: 0;
        color: #4ECDC4;
        font-size: 1.8rem;
    }
    .stat-card p {
        margin: 4px 0 0 0;
        color: #AAA;
        font-size: 0.8rem;
    }

    /* Subtle animation on load */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stChatMessage {
        animation: fadeIn 0.3s ease-out;
    }

    /* Upload area styling */
    .uploadedFile {
        border-radius: 8px;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid rgba(108, 99, 255, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        border-color: #6C63FF;
        box-shadow: 0 0 15px rgba(108, 99, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# ─── Async Helper ───
def run_async(coro):
    """Run an async function from synchronous Streamlit context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ─── Initialize Components (cached) ───
@st.cache_resource
def init_components():
    """Initialize all AI components (cached across reruns)."""
    llm = LLMProvider()
    kb = MaintenanceKnowledgeBase()
    vs = VectorStore()
    doc_loader = DocumentLoader(vs, kb)
    rag = RAGChain(vs, llm)
    entity_mem = EntityMemory(vs, llm)

    # Auto-ingest sample documents on first run
    sample_dir = AppConfig.SAMPLE_DOCS_DIR
    if os.path.isdir(sample_dir):
        count = doc_loader.ingest_directory(sample_dir)
        if count > 0:
            logger.info("Auto-ingested %d sample document chunks", count)

    return {
        'llm': llm,
        'kb': kb,
        'vs': vs,
        'doc_loader': doc_loader,
        'rag': rag,
        'entity_mem': entity_mem
    }


def get_memory(kb) -> ConversationMemory:
    """Get or create conversation memory for the current session."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(datetime.now().strftime('%Y%m%d_%H%M%S'))

    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationMemory(kb, st.session_state.session_id)

    return st.session_state.memory


# ─── Initialize ───
components = init_components()
memory = get_memory(components['kb'])

# Initialize chat messages in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    welcome_msg = (
        "👋 Hello! I'm your **AI Maintenance Assistant**, powered by RAG and contextual memory.\n\n"
        "I can help you with:\n"
        "- 🔍 **Equipment troubleshooting** — describe symptoms, I'll diagnose\n"
        "- 📖 **Maintenance procedures** — ask about any equipment type\n"
        "- 📊 **Sensor data analysis** — check the Sensor Analysis page\n"
        "- 📄 **Document knowledge** — upload manuals and I'll learn from them\n\n"
        "Try asking: *\"What are common causes of pump vibration?\"*"
    )
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg, "sources": []})


# ─── Sidebar ───
with st.sidebar:
    st.markdown('<p class="gradient-text">🔧 Maintenance AI</p>', unsafe_allow_html=True)
    st.caption(f"v{AppConfig.APP_VERSION} • RAG + Memory")

    st.divider()

    # ── Vector Store Stats ──
    stats = components['vs'].get_stats()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="stat-card"><h3>{stats["document_chunks"]}</h3><p>Doc Chunks</p></div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div class="stat-card"><h3>{stats["entity_entries"]}</h3><p>Entities</p></div>',
            unsafe_allow_html=True
        )

    st.divider()

    # ── Document Upload ──
    st.subheader("📁 Upload Documents")
    st.caption("Upload maintenance manuals, SOPs, or equipment docs")

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=['pdf', 'docx', 'txt', 'md'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if not components['kb'].is_document_ingested(uploaded_file.name):
                # Save uploaded file temporarily
                upload_dir = AppConfig.UPLOAD_DIR
                os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                # Ingest
                with st.spinner(f"📥 Ingesting {uploaded_file.name}..."):
                    chunks = components['doc_loader'].ingest_file(file_path)
                    if chunks > 0:
                        st.success(f"✅ {uploaded_file.name}: {chunks} chunks")
                    else:
                        st.warning(f"⚠️ No content extracted from {uploaded_file.name}")
            else:
                st.info(f"ℹ️ {uploaded_file.name} already ingested")

    st.divider()

    # ── Ingested Documents List ──
    st.subheader("📚 Knowledge Base")
    docs = components['kb'].get_ingested_documents()
    if docs:
        for doc in docs:
            st.markdown(
                f'<div class="source-card">📄 <b>{doc["filename"]}</b><br>'
                f'<span style="color:#888">{doc["chunk_count"]} chunks • '
                f'{doc["file_type"]}</span></div>',
                unsafe_allow_html=True
            )
    else:
        st.caption("No documents ingested yet. Upload files above or check the sample docs.")

    st.divider()

    # ── Equipment List ──
    st.subheader("🔧 Equipment")
    equipment = components['kb'].get_all_equipment()
    if equipment:
        for eq in equipment:
            st.markdown(f"• **{eq['id']}**: {eq['name']} ({eq['type']})")
    else:
        st.caption("No equipment registered. Use the Equipment Manager page.")

    st.divider()

    # ── Session Controls ──
    st.subheader("💬 Session")
    st.caption(f"Session: `{st.session_state.session_id}`")
    st.caption(f"Messages: {memory.get_message_count()}")

    if st.button("🗑️ New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop('memory', None)
        st.session_state.pop('session_id', None)
        st.rerun()


# ─── Main Chat Area ───
st.markdown('<p class="gradient-text">💬 Maintenance AI Chat</p>', unsafe_allow_html=True)
st.caption("Ask about equipment maintenance, troubleshooting, or procedures • Powered by RAG + Contextual Memory")

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🔧" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

        # Show sources if available
        sources = msg.get("sources", [])
        if sources:
            with st.expander(f"📚 {len(sources)} source(s) referenced", expanded=False):
                for src in sources:
                    st.markdown(
                        f'<div class="source-card">'
                        f'📄 <b>{src["source"]}</b> • Relevance: {src["relevance"]}<br>'
                        f'<span style="color:#999">{src["preview"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

# Chat input
if prompt := st.chat_input("Ask about maintenance, equipment, or procedures..."):
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

    # Add to memory
    memory.add_message("user", prompt)

    # Generate response
    with st.chat_message("assistant", avatar="🔧"):
        with st.spinner("🔍 Searching knowledge base & analyzing..."):
            # Get conversation context
            conversation_history = memory.get_context_string()

            # Get entity context
            entity_context = components['entity_mem'].get_entity_context(prompt)

            # RAG query
            result = run_async(components['rag'].query(
                user_query=prompt,
                conversation_history=conversation_history,
                entity_context=entity_context
            ))

            response = result['response']
            sources = result['sources']

        # Display response
        st.markdown(response)

        # Show sources
        if sources:
            with st.expander(f"📚 {len(sources)} source(s) referenced", expanded=False):
                for src in sources:
                    st.markdown(
                        f'<div class="source-card">'
                        f'📄 <b>{src["source"]}</b> • Relevance: {src["relevance"]}<br>'
                        f'<span style="color:#999">{src["preview"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    # Save to session state and memory
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources
    })
    memory.add_message("assistant", response, sources)

    # Background: extract entities and auto-summarize
    run_async(components['entity_mem'].extract_and_store_entities(prompt, response))
    run_async(memory.auto_summarize(components['llm']))
