"""
app/pipeline.py — Cached RAG pipeline wrapper for Streamlit.

Streamlit reruns the entire script on every user interaction.
Without caching, the ChromaDB connection and embedding model would
reload on every message — adding 5–10 seconds of latency per query.

@st.cache_resource solves this: it runs the initialisation once,
stores the result in memory, and reuses it across all reruns for
the lifetime of the browser session.

This module exposes two things to app.py:
  1. get_pipeline() — returns a cached, ready-to-use RAGPipeline
  2. RAGPipeline.query() — takes a question, returns answer + chunks
"""

import os
import re
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Add project root to path so app/ can import the pipeline modules
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieve_chunks import (
    load_chroma_collection,
    load_embedding_model,
    FinanceRetriever,
    build_filter,
)
from rag_pipeline import (
    analyse_query,
    build_prompt_template,
    format_context,
    get_llm_answer,
    KNOWN_COMPANIES,
)
from app.config import (
    VECTORDB_PATH,
    ENV_PATH,
    EMBEDDING_MODEL,
    GROQ_MODEL,
    COLLECTION_NAME,
    TOP_K,
    TEMPERATURE,
    MAX_TOKENS,
)


# ---------------------------------------------------------------------------
# Result dataclass — structured return type from query()
# ---------------------------------------------------------------------------

class QueryResult:
    """
    Container for one complete RAG query result.

    Bundles the LLM answer together with the retrieved chunks so the
    UI can display both without needing separate function calls.

    Attributes:
        answer          : Final LLM response string.
        docs_and_scores : List of (Document, similarity_score) pairs
                          from the retriever, ordered best-first.
    """

    def __init__(
        self,
        answer: str,
        docs_and_scores: list[tuple[Document, float]],
    ) -> None:
        self.answer          = answer
        self.docs_and_scores = docs_and_scores


# ---------------------------------------------------------------------------
# RAG pipeline class
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Self-contained RAG pipeline that initialises once and handles
    all queries for the lifetime of the Streamlit session.

    Initialisation loads:
      - ChromaDB collection (HNSW index + SQLite from disk)
      - SentenceTransformer embedding model (from local HF cache)
      - ChatGroq LLM client (authenticated via GROQ_API_KEY)
      - FinanceRetriever (wraps ChromaDB + embedding model)
      - PromptTemplate (instructions + {context} + {question})

    After __init__, query() is the only method the UI needs to call.
    """

    def __init__(self) -> None:
        """
        Load all pipeline components.
        Called once via get_pipeline() and cached by Streamlit.
        """
        # Load GROQ_API_KEY from .env into os.environ
        load_dotenv(dotenv_path=ENV_PATH, override=False)
        api_key = os.getenv("GROQ_API_KEY", "").strip()

        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError(
                "GROQ_API_KEY not set in .env\n"
                "Get your key at: https://console.groq.com/keys"
            )

        # ChromaDB — reads the persistent vector store from disk
        collection = load_chroma_collection(VECTORDB_PATH, COLLECTION_NAME)

        # Embedding model — loaded from local HuggingFace cache
        emb_model = load_embedding_model(EMBEDDING_MODEL)

        # Retriever — wraps ChromaDB + embedding model
        self._retriever = FinanceRetriever(
            collection=collection,
            model=emb_model,
            top_k=TOP_K,
        )

        # ChatGroq LLM client
        self._llm = ChatGroq(
            api_key=api_key,
            model_name=GROQ_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        # PromptTemplate — instructions baked in, placeholders filled at query time
        self._prompt_template = build_prompt_template()

    # -----------------------------------------------------------------------
    # Prompt template — reused from rag_pipeline.py
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_prompt_template() -> PromptTemplate:
        return build_prompt_template()

    # -----------------------------------------------------------------------
    # Context formatter — reused from rag_pipeline.py
    # -----------------------------------------------------------------------

    @staticmethod
    def _format_context(docs_and_scores: list[tuple[Document, float]]) -> str:
        return format_context(docs_and_scores)

    # -----------------------------------------------------------------------
    # Public query method
    # -----------------------------------------------------------------------

    def query(self, question: str) -> QueryResult:
        """
        Run the full RAG pipeline for one user question.

        Detects trend queries (multi-year) and routes them through
        per-year metadata-filtered retrieval. Single queries use
        standard top-K with optional company/year filter.
        """
        parsed = analyse_query(question)

        # Route retrieval based on query type
        if parsed["is_trend"] and parsed["company"] and parsed["years"]:
            docs_and_scores = self._retriever.get_trend_documents(
                query=question,
                company=parsed["company"],
                years=parsed["years"],
                chunks_per_year=2,
            )
        else:
            docs_and_scores = self._retriever.get_relevant_documents(
                query=question,
                company=parsed["company"],
                year=parsed["years"][0] if len(parsed["years"]) == 1 else None,
            )

        context      = format_context(docs_and_scores)
        answer       = get_llm_answer(self._llm, self._prompt_template, context, question)

        return QueryResult(answer=answer, docs_and_scores=docs_and_scores)


# ---------------------------------------------------------------------------
# Cached initialiser — called by app.py
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading pipeline...")
def get_pipeline() -> RAGPipeline:
    """
    Initialise and cache the RAGPipeline for the Streamlit session.

    @st.cache_resource ensures this function runs exactly once per
    session regardless of how many times the Streamlit script reruns.
    The cached RAGPipeline object is reused for every subsequent query,
    keeping ChromaDB and the embedding model loaded in memory.

    Returns:
        Ready-to-use RAGPipeline instance.
    """
    return RAGPipeline()
