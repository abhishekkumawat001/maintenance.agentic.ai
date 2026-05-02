# 🔧 Maintenance.Agentic.AI

**An AI-Powered Industrial Maintenance Assistant with RAG & Contextual Memory**

A production-grade maintenance chatbot powered by Google Gemini, ChromaDB vector search, and sentence-transformers. Features Retrieval-Augmented Generation (RAG) for grounded answers from maintenance documents, persistent contextual memory across conversations, and a modern Streamlit web interface.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Gemini](https://img.shields.io/badge/Gemini-1.5%20Flash-orange.svg)](https://ai.google.dev)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-red.svg)](https://www.trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌟 Key Features

### 🧠 RAG Pipeline (Retrieval-Augmented Generation)
- **ChromaDB Vector Store** — Semantic search over maintenance documents
- **sentence-transformers** (`all-MiniLM-L6-v2`) — Local embeddings, no API calls
- **Document Ingestion** — Upload PDFs, DOCX, TXT files; auto-chunked and embedded
- **Source Citations** — Every answer shows which documents were referenced
- **Hybrid Context** — Combines retrieved knowledge with Gemini's reasoning

### 💬 Contextual Memory System
- **Sliding Window History** — Remembers the last 20 messages in full detail
- **Auto-Summarization** — Older messages compressed by Gemini into summaries
- **Entity Memory** — Tracks facts about specific equipment across conversations
- **Persistent Storage** — Conversations saved to SQLite, survive app restarts
- **Cross-Session Recall** — Start new conversations with context from past sessions

### 🤖 AI-Powered Diagnostics
- **Gemini 1.5 Flash Integration** — Advanced reasoning for maintenance analysis
- **Sensor Data Analysis** — Temperature, vibration, pressure, humidity, sound level
- **Root Cause Analysis** — AI-driven fault diagnosis with confidence scoring
- **Multimodal Vision** — Send equipment images directly to Gemini for analysis

### 🖥️ Modern Web Interface
- **Streamlit Chat UI** — Native chat experience with streaming responses
- **Dark Theme** — Premium glassmorphism design with purple/teal accents
- **Multi-Page App** — Chat, Sensor Analysis, Equipment Manager, Document Manager
- **Drag & Drop Upload** — Upload maintenance manuals directly from the sidebar
- **Semantic Search Testing** — Test what the RAG retriever finds for any query

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Streamlit Web UI                           │
│  💬 Chat  │  📊 Sensors  │  🔧 Equipment  │  📄 Documents   │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│                   RAG Chain                                   │
│  Query → Embed → Retrieve → Augment Prompt → Generate        │
└──────┬───────────────┬──────────────────────┬────────────────┘
       │               │                      │
┌──────▼──────┐ ┌──────▼──────┐  ┌────────────▼───────────────┐
│  ChromaDB   │ │ Conversation │  │      Google Gemini         │
│ Vector Store│ │   Memory     │  │    1.5 Flash LLM           │
│ (embeddings)│ │ (SQLite +    │  │  (reasoning + generation)  │
│             │ │  summaries)  │  │                            │
└─────────────┘ └──────────────┘  └────────────────────────────┘
       │
┌──────▼──────┐
│  sentence-  │
│ transformers│
│ (local      │
│  embeddings)│
└─────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Google Gemini API key ([Get one free](https://ai.google.dev))

### Installation

```bash
# Clone
git clone https://github.com/abhishekkumawat001/maintenance.agentic.ai.git
cd maintenance.agentic.ai

# Virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install dependencies (includes PyTorch ~2GB first time)
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY
```

### Run

```bash
# Start the Streamlit web app
streamlit run streamlit_app.py

# Or run the legacy CLI interface
python agentic_ai_refactored.py
```

---

## 📁 Project Structure

```
maintenance.agentic.ai/
├── streamlit_app.py            # Main Streamlit chat interface
├── agentic_ai_refactored.py    # Legacy CLI interface
├── app/                        # Core application package
│   ├── config.py               # Centralized configuration
│   ├── models.py               # Data models (dataclasses)
│   ├── llm_provider.py         # Gemini API with retry logic
│   ├── knowledge_base.py       # SQLite storage (equipment, history, conversations)
│   ├── vector_store.py         # ChromaDB + sentence-transformers
│   ├── document_loader.py      # PDF/DOCX/TXT ingestion pipeline
│   ├── rag_chain.py            # Retrieval-Augmented Generation chain
│   ├── memory.py               # Conversation memory + entity tracking
│   ├── sensor_processor.py     # Sensor anomaly detection
│   ├── vision_processor.py     # OpenCV + Gemini multimodal vision
│   ├── diagnostic_engine.py    # AI fault diagnosis
│   └── maintenance_planner.py  # AI maintenance scheduling
├── pages/                      # Streamlit sub-pages
│   ├── sensor_analysis.py      # Sensor data input & analysis
│   ├── equipment_manager.py    # Equipment CRUD
│   └── document_manager.py     # Document upload & search testing
├── data/
│   ├── sample_docs/            # Pre-loaded maintenance documents
│   ├── chroma_db/              # Vector store (auto-generated)
│   └── maintenance.db          # SQLite database (auto-generated)
├── .streamlit/config.toml      # Dark theme configuration
├── requirements.txt
├── .env.example
└── README.md
```

---

## 💡 Usage

### Chat with the AI
Ask any maintenance question. The RAG pipeline searches your knowledge base and provides grounded answers with source citations.

### Upload Documents
Drag & drop PDFs, DOCX, or TXT files in the sidebar or Document Manager page. Documents are automatically chunked, embedded, and indexed for semantic search.

### Sensor Analysis
Enter real-time sensor readings (temperature, vibration, pressure, humidity, sound) and get AI-powered anomaly detection and fault diagnosis.

### Equipment Management
Register your equipment inventory. The AI provides type-specific maintenance recommendations on registration.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Google Gemini 1.5 Flash |
| **Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`) |
| **Vector Store** | ChromaDB (local, persistent) |
| **Database** | SQLite |
| **Web UI** | Streamlit |
| **Vision** | OpenCV + Gemini Multimodal |
| **Document Parsing** | PyPDF2, python-docx |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

**Made with ❤️ by Abhishek Kumawat**