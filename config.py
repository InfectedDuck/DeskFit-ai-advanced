"""Central configuration for DeskFit AI."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

# Embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ChromaDB
COLLECTION_NAME = "deskfit_knowledge"

# EPAM DIAL (OpenAI-compatible)
DIAL_API_BASE = os.getenv("DIAL_API_BASE", "https://dial.epam.com/openai/deployments/gpt-4o-mini/chat/completions")
DIAL_API_KEY = os.getenv("DIAL_API_KEY", "")
DIAL_MODEL_NAME = os.getenv("DIAL_MODEL_NAME", "gpt-4o-mini")

# RAG parameters
TOP_K_RESULTS = 5
MAX_CONTEXT_LENGTH = 3000

# Hybrid retrieval
USE_HYBRID_RETRIEVAL = True
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VECTOR_CANDIDATE_K = 15
BM25_CANDIDATE_K = 15
