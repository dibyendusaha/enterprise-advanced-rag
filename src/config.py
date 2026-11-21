import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DOCS_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
CHROMA_DIR = VECTORSTORE_DIR / "chroma_db"
BM25_INDEX_PATH = VECTORSTORE_DIR / "bm25_chunks.pkl"

# Chunking parameters
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

# 100% OpenAI Models
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
RERANKER_LLM_MODEL = "gpt-4o-mini"          # Used by OpenAI structured relevance reranker

# Retrieval hyperparameters
DENSE_TOP_K = 10
SPARSE_TOP_K = 10
HYBRID_WEIGHTS = [0.45, 0.55]               # [BM25_weight, Dense_weight]
RERANK_TOP_N = 6                            # Final chunks passed to LLM after reranking
MIN_RELEVANCE_SCORE = 0.35                  # Minimum OpenAI relevance score (0.0 - 1.0) to keep a chunk

# Evaluates incoming user queries through Gaurdrails
MODERATION_MODEL = "omni-moderation-latest"
MODERATION_SCORE_THRESHOLD = 0.50           # Calibrated confidence threshold (0.0 - 1.0)