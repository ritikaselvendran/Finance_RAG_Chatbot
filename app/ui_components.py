"""
app/ui_components.py — Reusable Streamlit UI rendering functions.

Each function renders one self-contained piece of the UI.
app.py calls these functions — it never calls st.* directly.
"""

from pathlib import Path

import streamlit as st
from langchain_core.documents import Document

from app.config import APP_TITLE, APP_ICON, APP_DESCRIPTION, GROQ_MODEL, TOP_K


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

def render_header() -> None:
    """
    Render the top-of-page title, subtitle, and pipeline info banner.
    """
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption(APP_DESCRIPTION)

    with st.expander("ℹ️ Pipeline details", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("LLM Model",   GROQ_MODEL)
        col2.metric("Embedding",   "all-MiniLM-L6-v2")
        col3.metric("Top-K Chunks", TOP_K)

    st.divider()


# ---------------------------------------------------------------------------
# Sidebar — session history + controls
# ---------------------------------------------------------------------------

def render_sidebar(
    on_new_session,
    on_switch_session,
    on_delete_session,
    current_session_id: str,
) -> None:
    """
    Render the sidebar with:
      - New Chat button
      - Saved session history list (click to load, trash to delete)
      - Example questions
      - Clear current chat button

    Each saved session is shown as a button with its auto-generated title
    (first 50 chars of the first user message). Clicking it loads that
    session's messages into the main chat area.

    Args:
        on_new_session     : Callback — called when "New Chat" is clicked.
        on_switch_session  : Callback(session_id) — called when a session is clicked.
        on_delete_session  : Callback(session_id) — called when trash is clicked.
        current_session_id : The active session ID (highlighted in the list).
    """
    from app.chat_history import list_sessions, delete_session

    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")
        st.divider()

        # New Chat button
        if st.button("➕  New Chat", use_container_width=True, type="primary"):
            on_new_session()
            st.rerun()

        st.divider()

        # --- Session history list ---
        st.markdown("**💬 Chat History**")
        sessions = list_sessions()

        if not sessions:
            st.caption("No saved sessions yet.")
        else:
            for session in sessions:
                sid   = session["session_id"]
                title = session["title"]
                date  = session["updated_at"][:10]   # show date only

                # Highlight the active session
                is_active = sid == current_session_id

                col_btn, col_del = st.columns([5, 1])

                with col_btn:
                    label = f"{'▶ ' if is_active else ''}{title}"
                    if st.button(
                        label,
                        key=f"session_{sid}",
                        use_container_width=True,
                        help=f"Last updated: {date}",
                        type="secondary",
                    ):
                        if not is_active:
                            on_switch_session(sid)
                            st.rerun()

                with col_del:
                    if st.button(
                        "🗑",
                        key=f"del_{sid}",
                        help="Delete this session",
                    ):
                        on_delete_session(sid)
                        # If deleting the active session, start fresh
                        if is_active:
                            on_new_session()
                        st.rerun()

        st.divider()

        # --- Example questions ---
        st.markdown("**💡 Example questions**")
        st.markdown(
            """
- What was 3M's total net sales in 2015?
- What was Amazon's operating income in 2022?
- How much did Apple spend on R&D in 2021?
- What was Microsoft's cloud revenue in 2022?
- Amazon net income trend from 2018 to 2022
- Compare Apple and Microsoft revenue in 2022
- How many employees did Nike have in 2020?
            """
        )

        st.divider()
        st.caption("Answers are grounded in retrieved document chunks only.")


# ---------------------------------------------------------------------------
# Individual chat messages
# ---------------------------------------------------------------------------

def render_user_message(question: str) -> None:
    """Render a user message bubble."""
    with st.chat_message("user"):
        st.markdown(question)


def render_assistant_message(
    answer: str,
    docs_and_scores: list[tuple[Document, float]],
) -> None:
    """
    Render the assistant answer bubble with a collapsible sources panel.

    Sources panel shows each retrieved chunk with its metadata and text.
    Collapsed by default to keep the chat readable.
    """
    with st.chat_message("assistant"):
        st.markdown(answer)

        if docs_and_scores:
            with st.expander(
                f"📄 Retrieved Sources ({len(docs_and_scores)} chunks)",
                expanded=False,
            ):
                for rank, (doc, score) in enumerate(docs_and_scores, start=1):
                    render_source_chunk(rank, doc, score)


def render_source_chunk(rank: int, doc: Document, score: float) -> None:
    """
    Render one retrieved chunk inside the sources expander.

    Shows metadata (section, source, company, year, type, score)
    and the full chunk text.
    """
    m           = doc.metadata
    source_name = Path(m.get("source", "N/A")).name
    section     = m.get("section", "N/A")
    label       = (
        f"Chunk {rank} — "
        f"{section[:55]}{'...' if len(section) > 55 else ''}"
        f"  |  Score: {score:.4f}"
    )

    with st.expander(label, expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Section:** {section}")
            st.markdown(f"**Source:** `{source_name}`")
        with col2:
            st.markdown(f"**Company:** {m.get('company', 'N/A')}")
            st.markdown(
                f"**Year:** {m.get('year', 'N/A')}  |  "
                f"**Type:** {m.get('report_type', 'N/A')}"
            )
        st.markdown(f"**Similarity Score:** `{score:.4f}`")
        st.divider()
        st.markdown("**Chunk text:**")
        st.markdown(doc.page_content)


# ---------------------------------------------------------------------------
# Chat history replay
# ---------------------------------------------------------------------------

def render_chat_history(messages: list[dict]) -> None:
    """
    Re-render all previous messages on each Streamlit rerun.

    Streamlit reruns the script on every interaction. Without replaying
    the history here, old messages would disappear after each new message.

    Args:
        messages: List of message dicts from st.session_state["messages"].
    """
    for msg in messages:
        if msg["role"] == "user":
            render_user_message(msg["content"])
        else:
            render_assistant_message(
                answer=msg["content"],
                docs_and_scores=msg.get("docs_and_scores", []),
            )


# ---------------------------------------------------------------------------
# Error display
# ---------------------------------------------------------------------------

def render_error(error: Exception) -> None:
    """Render a styled error message when the pipeline fails."""
    with st.chat_message("assistant"):
        st.error(f"Something went wrong: {error}")
        st.info("Check that your GROQ_API_KEY is set and vectordb/ exists.")
