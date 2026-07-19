"""
app/chat_engine.py — Chat history management and query handling.

Owns two responsibilities:
  1. Managing st.session_state — initialising it, appending messages,
     switching sessions, and saving to disk after every message.
  2. Running one query through the RAGPipeline and returning the result.

Session state keys used:
    "session_id"  : str          — current session UUID
    "messages"    : list[dict]   — current session messages
"""

import streamlit as st
from langchain_core.documents import Document

from app.pipeline import RAGPipeline, QueryResult
from app.chat_history import (
    save_session,
    load_session,
    new_session_id,
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_chat_history() -> None:
    """
    Ensure session_id and messages exist in st.session_state.

    Called once at the top of app.py. On first load both keys are
    missing so we create a fresh session. On subsequent reruns within
    the same browser session the keys already exist and this is a no-op.
    """
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = new_session_id()
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


# ---------------------------------------------------------------------------
# Session switching
# ---------------------------------------------------------------------------

def switch_session(session_id: str) -> None:
    """
    Load a previously saved session into st.session_state.

    Called when the user clicks a session in the sidebar history list.
    Replaces the current messages with the saved ones and updates the
    active session_id so subsequent messages are saved to the right file.

    Args:
        session_id: UUID string of the session to switch to.
    """
    data = load_session(session_id)
    if data:
        st.session_state["session_id"] = session_id
        st.session_state["messages"]   = data["messages"]


def start_new_session() -> None:
    """
    Start a fresh conversation by resetting session state.

    Creates a new session_id and clears messages. The previous session
    is already saved on disk from the last message, so nothing is lost.
    """
    st.session_state["session_id"] = new_session_id()
    st.session_state["messages"]   = []


# ---------------------------------------------------------------------------
# Message accessors
# ---------------------------------------------------------------------------

def get_messages() -> list[dict]:
    """Return the current session's message list."""
    return st.session_state.get("messages", [])


def get_session_id() -> str:
    """Return the current session ID."""
    return st.session_state.get("session_id", "")


def add_user_message(question: str) -> None:
    """
    Append a user message and save the session to disk.

    Args:
        question: The user's question string.
    """
    st.session_state["messages"].append({
        "role"           : "user",
        "content"        : question,
        "docs_and_scores": [],
    })
    _persist()


def add_assistant_message(
    answer: str,
    docs_and_scores: list[tuple[Document, float]],
) -> None:
    """
    Append an assistant message with sources and save to disk.

    docs_and_scores is stored in session state as Python objects for
    the current session UI rendering, and serialised to JSON by
    save_session() for persistent storage.

    Args:
        answer          : LLM response string.
        docs_and_scores : List of (Document, similarity_score) pairs.
    """
    st.session_state["messages"].append({
        "role"           : "assistant",
        "content"        : answer,
        "docs_and_scores": docs_and_scores,
    })
    _persist()


def _persist() -> None:
    """
    Save the current session state to disk.
    Called after every message so history is never lost.
    """
    save_session(
        session_id=st.session_state["session_id"],
        messages=st.session_state["messages"],
    )


# ---------------------------------------------------------------------------
# Query handler
# ---------------------------------------------------------------------------

def handle_query(pipeline: RAGPipeline, question: str) -> QueryResult:
    """
    Run one user question through the RAG pipeline.

    Args:
        pipeline : Cached RAGPipeline instance from get_pipeline().
        question : User's plain-text question.

    Returns:
        QueryResult with answer and retrieved chunks.
    """
    return pipeline.query(question)
