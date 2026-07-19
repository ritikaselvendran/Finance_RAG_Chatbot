"""
app.py — Main Streamlit entry point for the Finance RAG Chatbot.

Run with:
    streamlit run app.py

Flow on each rerun:
  1. set_page_config      — must be the very first st call
  2. render_header()      — title + pipeline info
  3. render_sidebar()     — session history list + controls
  4. get_pipeline()       — cached RAGPipeline (loads once)
  5. init_chat_history()  — ensure session state keys exist
  6. render_chat_history()— replay current session messages
  7. st.chat_input()      — wait for user message
  8. On new message:
       a. show user bubble immediately
       b. run pipeline with spinner
       c. save user + assistant messages to state + disk
       d. show assistant bubble with sources
"""

import streamlit as st

# MUST be the first Streamlit call
st.set_page_config(
    page_title="Finance RAG Chatbot",
    page_icon="📊",
    layout="wide",
)

from app.ui_components import (
    render_header,
    render_sidebar,
    render_chat_history,
    render_user_message,
    render_assistant_message,
    render_error,
)
from app.pipeline import get_pipeline
from app.chat_engine import (
    init_chat_history,
    get_messages,
    get_session_id,
    add_user_message,
    add_assistant_message,
    handle_query,
    switch_session,
    start_new_session,
)
from app.chat_history import delete_session


# ---------------------------------------------------------------------------
# Static UI — header
# ---------------------------------------------------------------------------

render_header()


# ---------------------------------------------------------------------------
# Sidebar — pass callbacks so ui_components stays decoupled from state
# ---------------------------------------------------------------------------

render_sidebar(
    on_new_session    = start_new_session,
    on_switch_session = switch_session,
    on_delete_session = delete_session,
    current_session_id= get_session_id(),
)


# ---------------------------------------------------------------------------
# Pipeline — cached, loads once per browser session
# ---------------------------------------------------------------------------

pipeline = get_pipeline()


# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------

init_chat_history()


# ---------------------------------------------------------------------------
# Replay current session messages
# ---------------------------------------------------------------------------

render_chat_history(get_messages())


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

question = st.chat_input("Ask a question about the financial reports...")

if question:
    # Show user message immediately (before pipeline runs)
    render_user_message(question)

    try:
        with st.spinner("Retrieving relevant chunks and generating answer..."):
            result = handle_query(pipeline, question)

        # Persist both messages to state + disk
        add_user_message(question)
        add_assistant_message(result.answer, result.docs_and_scores)

        # Render the assistant response
        render_assistant_message(result.answer, result.docs_and_scores)

    except Exception as e:
        add_user_message(question)
        render_error(e)
