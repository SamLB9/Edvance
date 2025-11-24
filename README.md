# 🎓 Edvance Study Coach

> **An AI-Powered RAG (Retrieval Augmented Generation) Platform for Personalized Learning**

A production-ready, multi-user educational platform that leverages advanced AI/ML techniques to provide adaptive learning experiences through intelligent document analysis, personalized quiz generation, and professional content summarization.

🌐 **[Live Demo](https://edvance-production.up.railway.app/)** | 📚 [Documentation](#-overview) | 🚀 [Quick Start](#-quick-start)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green.svg)](https://openai.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-orange.svg)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-purple.svg)](https://www.trychroma.com/)
[![Railway](https://img.shields.io/badge/Deployed%20on-Railway-black.svg)](https://railway.app)

---

## 🚀 Overview

**Edvance Study Coach** is a full-stack AI application deployed on [Railway](https://edvance-production.up.railway.app/) that demonstrates expertise in:

- **RAG Architecture**: Advanced retrieval-augmented generation with semantic search
- **Vector Embeddings**: ChromaDB-based vector store for efficient similarity search
- **LLM Orchestration**: Multi-step AI workflows with structured outputs
- **Adaptive Learning**: Machine learning-driven difficulty adjustment based on user performance
- **Production Engineering**: User authentication, multi-tenancy, error handling, and cloud deployment

### Key Technical Achievements

✅ **Semantic Search & RAG**: Implemented efficient document retrieval using OpenAI embeddings and ChromaDB  
✅ **Context-Aware Generation**: File-specific context filtering to prevent cross-contamination in multi-document scenarios  
✅ **Adaptive Difficulty Algorithm**: ML-based difficulty adjustment using historical performance data  
✅ **Multi-User Architecture**: Isolated user data with secure authentication and session management  
✅ **Production Deployment**: Configured for Railway with optimized dependency management  
✅ **Professional PDF Generation**: LaTeX-based document generation with custom branding and metadata

---

## 🏗️ Architecture & Technical Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | OpenAI GPT-4 / GPT-3.5 | Content generation, quiz creation, summarization |
| **Embeddings** | OpenAI `text-embedding-3-small` | Semantic document representation |
| **Vector Store** | ChromaDB | Efficient similarity search and retrieval |
| **Framework** | LangChain | LLM orchestration and document processing |
| **Frontend** | Streamlit | Interactive web interface |
| **Backend** | Python 3.10+ | Core application logic |
| **Authentication** | Streamlit-Authenticator | Secure user management |
| **PDF Generation** | LaTeX (pdflatex) | Professional document compilation |

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                       │
│      (Authentication, UI Components, State Management)      │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                  Application Layer (app.py)                 │
│                • User session management                    │
│                • Multi-tenant data isolation                │
│                • Background task orchestration              │
└────────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────┼───────────────┐
              │              │               │
      ┌───────▼──────┐ ┌─────▼─────┐ ┌───────▼─────┐
      │  RAG Engine  │ │  Quiz     │ │  Summary    │
      │ • Embeddings │ │• Adaptive │ │  Engine     │ 
      │ • Retrieval  │ │   Diff.   │ │ • LaTeX Gen │
      │ • Context    │ │           │ │ • PDF Comp. │
      └───────┬──────┘ └─────┬─────┘ └──────┬──────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │    ChromaDB Vector Store    │
              │   (Persistent embeddings)   │
              └─────────────────────────────┘
```

---

## 🔑 Core Features

### 1. **Intelligent Document Management**
- **Multi-format Support**: PDF, TXT, MD file ingestion
- **Automatic Chunking**: Recursive text splitting with configurable overlap
- **Semantic Indexing**: Real-time vector store updates with ChromaDB
- **User Isolation**: Per-user document storage and vector stores
- **Auto-topic Generation**: LLM-powered document topic extraction

### 2. **Adaptive Quiz Generation**
- **RAG-Powered Questions**: Context-aware question generation from uploaded documents
- **Adaptive Difficulty**: ML algorithm adjusts difficulty based on historical performance
- **Multiple Question Types**: MCQ (single/multi-select), True/False, Short Answer, Matching, Ordering
- **Intelligent Feedback**: LLM-generated personalized feedback for each answer
- **Performance Tracking**: Detailed analytics on response times, accuracy, and learning patterns

### 3. **Professional PDF Summarization**
- **Focus-Aware Summaries**: Generate summaries for specific topics or entire documents
- **File-Specific Context**: Strict document isolation prevents cross-contamination
- **LaTeX Compilation**: Professional PDF generation with custom branding
- **Metadata Management**: Proper PDF metadata for better document organization
- **Export Options**: PDF and Markdown export with session persistence

### 4. **Flashcard Generation**
- **Intelligent Card Creation**: Context-aware flashcard generation from documents
- **Multiple Formats**: Export to TSV, CSV (Quizlet), and Markdown
- **Customizable Content**: Formula-focused, definition-based, or example-driven cards
- **Cloze Deletion**: Automatic fill-in-the-blank card generation

### 5. **User Management & Analytics**
- **Secure Authentication**: Multi-user support with encrypted credentials
- **Progress Tracking**: Session history, accuracy metrics, and learning analytics
- **Frequently Missed Items**: AI-identified knowledge gaps for targeted review
- **Course Assignment**: Organize documents by course for better management

---

## 💻 Technical Implementation Highlights

### RAG (Retrieval Augmented Generation)

```python
# File-specific context retrieval prevents cross-contamination
def retrieve_context(vs, query: str, k: int = 6, source_path: str = None) -> str:
    """
    Retrieves top-k relevant chunks with optional source filtering.
    Ensures only content from the specified document is used.
    """
    if source_path:
        # Filter by source document for precise context
        results = vs.similarity_search_with_score(
            query, k=k, filter={"source": source_path}
        )
    else:
        results = vs.similarity_search_with_score(query, k=k)
    return format_context(results)
```

### Adaptive Difficulty Algorithm

```python
def get_adaptive_difficulty(topic: str) -> str:
    """
    ML-based difficulty adjustment using historical performance.
    Analyzes past accuracy, response times, and topic-specific patterns.
    """
    history = memory.get_topic_history(topic)
    if not history:
        return "medium"  # Default for new topics
    
    accuracy = calculate_accuracy(history)
    avg_response_time = calculate_avg_time(history)
    
    # Adaptive logic: adjust based on performance metrics
    if accuracy > 0.8 and avg_response_time < threshold:
        return "hard"
    elif accuracy < 0.5:
        return "easy"
    return "medium"
```

### Multi-User Data Isolation

```python
def get_user_notes_dir(username: str) -> Path:
    """Isolated per-user document storage"""
    return NOTES_DIR / username

def build_or_load_vectorstore(chunks: List[Document], persist_dir: str):
    """User-specific vector stores prevent data leakage"""
    user_vectorstore_dir = f"vectorstore/{username}"
    return Chroma.from_documents(chunks, embeddings, persist_directory=user_vectorstore_dir)
```

---

## 📁 Project Structure

```
Student_Coach_Q-A/
├── app.py                               # Main Streamlit application (3,900+ lines)
├── auth_config.yaml                     # User authentication configuration
├── requirements.txt                     # Python dependencies
├── .env                                 # Environment variables (API keys)
│
├── src/                                 # Core application modules
│   ├── config.py                        # Model configuration & settings
│   ├── ingest.py                        # Document loading & chunking
│   ├── retriever.py                     # Vector store & semantic search
│   ├── quiz_engine.py                   # Adaptive quiz generation
│   ├── evaluation.py                    # Answer grading & feedback
│   ├── memory.py                        # Performance tracking & analytics
│   ├── summary_engine.py                # PDF summary orchestration
│   ├── dynamic_latex_generator.py       # LaTeX document generation
│   ├── flashcards_engine.py             # Flashcard generation
│   └── auto_topic.py                    # Document topic extraction
│
├── data/
│   └── notes/                           # User-uploaded documents (per-user)
│       ├── {username}/                  # Isolated user directories
│       └── BayesTheorem.pdf             # Global shared document
│
├── vectorstore/                         # ChromaDB persistent storage
│   └── {user_id}/                       # Per-user vector stores
│
├── user_data/                           # User-specific data
│   ├── progress_{username}.json         # Performance tracking
│   ├── document_topics_{username}.json
│   └── saved_content_{username}.json
│
└── generated_summaries/                 # PDF outputs (per-user)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key
- LaTeX distribution (for PDF generation - optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/SamLB9/Edvance.git
cd Edvance/Student_Coach_Q-A

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-key-here
```

### Run Locally

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Deploy to Railway

1. Connect your GitHub repository to Railway
2. Set `OPENAI_API_KEY` as an environment variable
3. Railway will automatically detect and deploy the Streamlit app

---

## 🎯 Use Cases

### For Students
- **Personalized Study Plans**: Upload course materials and get AI-generated study content
- **Adaptive Practice**: Quizzes that adjust to your skill level
- **Knowledge Gap Analysis**: Identify areas needing more focus
- **Professional Summaries**: Generate publication-quality study notes

### For Educators
- **Content Analysis**: Automatically extract topics and themes from course materials
- **Assessment Generation**: Create diverse question sets from documents
- **Progress Monitoring**: Track student performance and learning patterns

---

## 🔧 Key Technical Decisions

### Why ChromaDB?
- **Lightweight**: No external dependencies, embedded database
- **Fast Retrieval**: Optimized for similarity search
- **Persistent Storage**: Survives application restarts
- **Metadata Filtering**: Enables file-specific context retrieval

### Why LangChain?
- **Abstraction Layer**: Simplifies LLM orchestration
- **Document Processing**: Built-in chunking and loading utilities
- **Future-Proof**: Easy to swap LLM providers
- **Structured Outputs**: Type-safe prompt engineering

### Why Streamlit?
- **Rapid Development**: Fast iteration for AI applications
- **Built-in Components**: Charts, file uploads, authentication
- **State Management**: Handles complex UI state
- **Deployment Ready**: One-command deployment to cloud

---

## 📊 Performance Optimizations

- **Background Embedding Pre-loading**: Thread-based async initialization
- **Vector Store Caching**: Persistent ChromaDB to avoid rebuilds
- **PDF Preview Caching**: Base64 encoding with size limits
- **Lazy Loading**: Vector stores and topics generated on-demand
- **Session State Management**: Efficient cleanup to prevent memory leaks

---

## 🔒 Security & Privacy

- **User Authentication**: Secure credential management with Streamlit-Authenticator
- **Data Isolation**: Complete separation of user data and vector stores
- **API Key Security**: Environment variable management
- **No Data Sharing**: Users can only access their own documents and progress

---

## 🛠️ Development

### Running Tests

```bash
# Run quiz generation test
python -m src.main --topic "Bayes theorem" --n 4 --feedback immediate
```

### Code Quality

- Type hints throughout codebase
- Modular architecture for maintainability
- Comprehensive error handling
- Performance monitoring and optimization

---

## 📈 Future Enhancements

- [ ] Multi-language support
- [ ] Collaborative study groups
- [ ] Advanced analytics dashboard
- [ ] Mobile app integration
- [ ] Real-time collaborative editing
- [ ] Integration with learning management systems

---

## 👤 Author

**Sam Laborde-Balen**  
AI Engineer | ML Engineer | Full-Stack Developer

- GitHub: [@SamLB9](https://github.com/SamLB9)
- Email: labordebalensam@gmail.com

---

## 📄 License

This project is proprietary and confidential.

---

## 🙏 Acknowledgments

- OpenAI for GPT models and embeddings
- LangChain team for the excellent framework
- Streamlit for the amazing web framework
- ChromaDB for efficient vector storage

---

**Built with ❤️ using Python, LangChain, and OpenAI**
