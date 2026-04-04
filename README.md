# DeskFit AI - Micro-Fitness for the Asian Workaholic

> A RAG-powered AI wellness assistant that retrieves ultra-short desk stretches, posture corrections, and wellness tips tailored for professionals working long hours in high-stress corporate environments.

## Demo Video

> **[Video Link]** *(to be added after recording)*

---

## Main Idea

Professionals in high-pressure industries (tech, finance, consulting) across Asia regularly work 10-14 hour days at their desks. This leads to chronic issues: neck/back pain, eye strain, carpal tunnel, poor posture, and burnout. **DeskFit AI** is a conversational assistant that provides instant, evidence-based micro-exercises and wellness tips that can be done at a desk in 1-5 minutes, without equipment, and without drawing attention in an open office.

The system uses **Retrieval-Augmented Generation (RAG)** to ground its responses in a curated knowledge base of exercises, ergonomic guidelines, and wellness advice -- ensuring recommendations are specific, safe, and actionable rather than generic LLM-generated advice.

## Core Concepts

### Retrieval-Augmented Generation (RAG)
Instead of relying solely on the LLM's training data (which may be generic or outdated), RAG retrieves relevant documents from a curated knowledge base and injects them into the LLM prompt. This ensures:
- **Factual grounding**: responses reference specific exercises with exact steps and durations
- **Safety**: precautions and contraindications come from the knowledge base, not hallucination
- **Transparency**: users can see which sources informed each answer

### Vector Similarity Search
User queries are converted to embedding vectors and matched against pre-embedded knowledge base entries using cosine similarity. This enables semantic search -- "my wrists hurt from typing" matches exercises tagged for carpal tunnel prevention even without exact keyword overlap.

### Embedding Models
We use `all-MiniLM-L6-v2` from sentence-transformers, a lightweight (80MB) model that produces 384-dimensional vectors. It runs locally with no API costs and provides strong semantic similarity for English text.

## Architecture

```
User Question
     |
     v
+------------------+
|  Streamlit UI    |  Chat interface with filters & source visibility
+------------------+
     |
     v
+------------------+
|  RAG Pipeline    |  Orchestrates the full flow
+------------------+
     |
     +-------> [Embeddings Client] ---> 384-dim query vector
     |              (sentence-transformers, local)
     |
     +-------> [ChromaDB Vector Search] ---> Top-K relevant documents
     |              (cosine similarity + metadata filters)
     |
     +-------> [Prompt Builder] ---> System prompt + context + user query
     |
     +-------> [LLM Client (Groq)] ---> Streamed response
                    (Llama 3.3 70B via OpenAI-compatible API)
     |
     v
+------------------+
|  Response + Sources displayed in UI  |
+------------------+
```

## Dataset

The knowledge base contains **~100 curated entries** across three categories:

| Category | Count | Description |
|----------|-------|-------------|
| **Exercises** | ~50 | Desk stretches, breathing exercises, eye care routines, hand/wrist exercises, micro-strength movements |
| **Posture Tips** | ~25 | Ergonomic desk setup, sitting posture, monitor/keyboard positioning, standing desk guidelines |
| **Wellness Advice** | ~25 | Sleep optimization, hydration, nutrition for desk workers, mental health breaks, screen fatigue management |

Each entry includes rich metadata: body area, duration, difficulty level, equipment requirements, "best for" scenarios, benefits, and precautions. See [data/README.md](data/README.md) for full schema documentation.

## Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| **Language** | Python 3.14 | Modern Python with latest features |
| **Vector Database** | ChromaDB | Pure Python, no Docker/server needed, built-in vector search |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Free, local, fast, 384-dim vectors |
| **LLM Provider** | Groq (Llama 3.3 70B Versatile) | Free tier available, fast inference, OpenAI-compatible API |
| **UI Framework** | Streamlit | Fast prototyping, modern chat interface, built-in state management |
| **LLM Client** | openai Python package | Groq uses OpenAI-compatible endpoints |

## System Technical Details

### Embedding Pipeline
1. Each knowledge base entry is converted to a **text representation** combining title, description, steps, benefits, and "best for" fields
2. Text is embedded using `all-MiniLM-L6-v2` (384 dimensions, cosine similarity)
3. Vectors + metadata are stored in ChromaDB's persistent storage

### Query Pipeline
1. User question is embedded using the same model
2. ChromaDB performs cosine similarity search, returning top-K results (default: 5)
3. Optional metadata filters (body area, category, difficulty) narrow the search
4. Retrieved documents are formatted and injected into the LLM prompt
5. Groq (Llama 3.3 70B) generates a grounded response, streamed to the UI

### Data Storage
- ChromaDB runs as an embedded database (no separate server process)
- Persistent storage in `chroma_db/` directory
- Each document stored with: ID, embedding vector, text content, metadata dict

## Requirements

### Software
- Python 3.10+ (tested on 3.14)
- pip package manager

### Python Dependencies
```
chromadb>=0.5.0
sentence-transformers>=3.0.0
openai>=1.0.0
streamlit>=1.35.0
python-dotenv>=1.0.0
```

### API Keys
- Groq API key (free at [console.groq.com](https://console.groq.com), set in `.env` file)

### Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your Groq API key

# 3. Populate the vector database
python -m src.populate_db

# 4. Launch the application
streamlit run app.py
```

## Limitations

1. **English only**: the dataset and embedding model are optimized for English text
2. **Not medical advice**: exercises are for general wellness; users with injuries should consult a professional
3. **Local embedding model**: all-MiniLM-L6-v2 is lightweight but less capable than large models like text-embedding-3-large
4. **Dataset size**: ~100 entries is representative but not exhaustive; a production system would need 1000+ entries
5. **No personalization**: the system does not track user history or adapt to individual conditions
6. **ChromaDB scale**: suitable for demo/small datasets; production would need a dedicated vector DB like Qdrant or Pinecone
7. **Single-turn context**: each query is independent; the system does not maintain multi-turn exercise program context
8. **LLM dependency**: requires network access to Groq API; no offline LLM fallback
