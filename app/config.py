"""
app/config.py — Central configuration for the Streamlit chatbot.

All constants are defined here so changing a model name, path, or
parameter only requires editing one file. Every other module imports
from here instead of hardcoding values.
"""

from pathlib import Path

# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

# Root of the project — parent of the app/ folder
PROJECT_ROOT = Path(__file__).parent.parent

# Path to the persistent ChromaDB built in Phase 6
VECTORDB_PATH = PROJECT_ROOT / "vectordb"

# Path to the .env file containing GROQ_API_KEY
ENV_PATH = PROJECT_ROOT / ".env"

# -------------------------------------------------------------------
# Model settings
# -------------------------------------------------------------------

# Sentence-transformer used in Phase 5 — MUST match exactly
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Groq LLM — fast LPU-based inference
GROQ_MODEL = "qwen/qwen3.8-27b"

# -------------------------------------------------------------------
# Retrieval settings
# -------------------------------------------------------------------

# Number of chunks retrieved per query
TOP_K = 5

# ChromaDB collection name — must match Phase 6
COLLECTION_NAME = "finance_rag"

# -------------------------------------------------------------------
# LLM settings
# -------------------------------------------------------------------

# 0.0 = deterministic, no hallucination risk for financial figures
TEMPERATURE = 0.0

# Max tokens in the LLM response
MAX_TOKENS = 1024

# -------------------------------------------------------------------
# UI settings
# -------------------------------------------------------------------

APP_TITLE       = "Finance RAG Chatbot"
APP_ICON        = "📊"
APP_DESCRIPTION = "Ask questions about financial reports. Answers are grounded in the source documents."
ASSISTANT_AVATAR = "🤖"
USER_AVATAR      = "🧑"
