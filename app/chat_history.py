"""
app/chat_history.py — Persistent chat session storage.

Each conversation is saved as a JSON file on disk under:
    chat_sessions/<session_id>.json

File structure:
    {
        "session_id"  : "abc123",
        "title"       : "Amazon revenue 2018-2022",
        "created_at"  : "2025-07-18 10:30:00",
        "updated_at"  : "2025-07-18 10:35:00",
        "messages"    : [
            {
                "role"    : "user",
                "content" : "What was Amazon's net income...?",
                "sources" : []          // empty for user messages
            },
            {
                "role"    : "assistant",
                "content" : "Amazon's net income in 2018...",
                "sources" : [           // serialised metadata only (no vectors)
                    {
                        "section"     : "...",
                        "source"      : "AMAZON_2018_10K.md",
                        "company"     : "AMAZON",
                        "year"        : "2018",
                        "report_type" : "10K",
                        "score"       : 0.68,
                        "text"        : "..."
                    }
                ]
            }
        ]
    }

Why JSON?
---------
JSON is human-readable, easy to inspect and backup, and requires no
database. Each session is one file so listing, loading, and deleting
sessions are simple file operations.

docs_and_scores (LangChain Documents + floats) cannot be serialised
directly to JSON, so we extract just the metadata + text + score into
plain dicts for storage. On load, we reconstruct Document objects.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document

# Sessions stored here — created automatically
SESSIONS_DIR = Path(__file__).parent.parent / "chat_sessions"


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _docs_to_sources(
    docs_and_scores: list[tuple[Document, float]],
) -> list[dict]:
    """
    Convert (Document, score) pairs to JSON-serialisable dicts.

    Extracts metadata fields + page_content + score. Drops the embedding
    vector (not needed for display) so files stay small.

    Args:
        docs_and_scores: Retriever output.

    Returns:
        List of plain dicts safe to write to JSON.
    """
    sources = []
    for doc, score in docs_and_scores:
        m = doc.metadata
        sources.append({
            "section"     : m.get("section",      "N/A"),
            "source"      : Path(m.get("source",  "N/A")).name,
            "company"     : m.get("company",      "N/A"),
            "year"        : m.get("year",          "N/A"),
            "report_type" : m.get("report_type",  "N/A"),
            "score"       : round(score, 4),
            "text"        : doc.page_content,
        })
    return sources


def _sources_to_docs(
    sources: list[dict],
) -> list[tuple[Document, float]]:
    """
    Reconstruct (Document, score) pairs from serialised source dicts.

    Called when loading a saved session so the UI can re-render the
    source expanders with full metadata and chunk text.

    Args:
        sources: List of dicts loaded from JSON.

    Returns:
        List of (Document, similarity_score) tuples.
    """
    docs_and_scores = []
    for s in sources:
        doc = Document(
            page_content=s.get("text", ""),
            metadata={
                "section"     : s.get("section",     "N/A"),
                "source"      : s.get("source",      "N/A"),
                "company"     : s.get("company",     "N/A"),
                "year"        : s.get("year",        "N/A"),
                "report_type" : s.get("report_type", "N/A"),
            },
        )
        docs_and_scores.append((doc, s.get("score", 0.0)))
    return docs_and_scores


# ---------------------------------------------------------------------------
# Session title generator
# ---------------------------------------------------------------------------

def _generate_title(messages: list[dict]) -> str:
    """
    Auto-generate a session title from the first user message.

    Takes the first user message and truncates it to 50 characters.
    This gives the session a meaningful name without requiring the user
    to name it manually.

    Args:
        messages: List of message dicts in the session.

    Returns:
        Title string (max 50 chars).
    """
    for msg in messages:
        if msg["role"] == "user":
            text = msg["content"].strip()
            return text[:50] + ("..." if len(text) > 50 else "")
    return "New conversation"


# ---------------------------------------------------------------------------
# Core CRUD operations
# ---------------------------------------------------------------------------

def save_session(
    session_id: str,
    messages: list[dict],
) -> None:
    """
    Save the current conversation to a JSON file on disk.

    Creates the chat_sessions/ directory if it doesn't exist.
    If the session file already exists it is overwritten (update).

    The messages list from session state contains docs_and_scores as
    LangChain Document objects — these are serialised to plain dicts
    before writing to JSON.

    Args:
        session_id: Unique identifier for this session (UUID string).
        messages  : List of message dicts from st.session_state.
    """
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session_file = SESSIONS_DIR / f"{session_id}.json"

    # Convert messages — serialise docs_and_scores to plain dicts
    serialisable_messages = []
    for msg in messages:
        entry = {
            "role"    : msg["role"],
            "content" : msg["content"],
            "sources" : _docs_to_sources(msg.get("docs_and_scores", [])),
        }
        serialisable_messages.append(entry)

    # Build or update the session file
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if session_file.exists():
        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["messages"]   = serialisable_messages
        data["updated_at"] = now
        data["title"]      = _generate_title(serialisable_messages)
    else:
        data = {
            "session_id" : session_id,
            "title"      : _generate_title(serialisable_messages),
            "created_at" : now,
            "updated_at" : now,
            "messages"   : serialisable_messages,
        }

    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_session(session_id: str) -> dict | None:
    """
    Load a saved session from disk by its ID.

    Reconstructs docs_and_scores as (Document, score) tuples so the
    UI can render source expanders for historical messages.

    Args:
        session_id: UUID string of the session to load.

    Returns:
        Session dict with messages list, or None if file not found.
    """
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if not session_file.exists():
        return None

    with open(session_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct docs_and_scores for each assistant message
    for msg in data.get("messages", []):
        if msg["role"] == "assistant":
            msg["docs_and_scores"] = _sources_to_docs(msg.get("sources", []))
        else:
            msg["docs_and_scores"] = []

    return data


def list_sessions() -> list[dict]:
    """
    Return all saved sessions sorted by most recently updated.

    Reads only the metadata (title, dates) from each file — not the
    full messages — so listing is fast even with many sessions.

    Returns:
        List of dicts with keys: session_id, title, created_at, updated_at.
        Ordered newest-first.
    """
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = []

    for path in SESSIONS_DIR.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({
                "session_id" : data.get("session_id", path.stem),
                "title"      : data.get("title",      "Untitled"),
                "created_at" : data.get("created_at", ""),
                "updated_at" : data.get("updated_at", ""),
            })
        except Exception:
            continue

    # Sort newest first by updated_at
    sessions.sort(key=lambda x: x["updated_at"], reverse=True)
    return sessions


def delete_session(session_id: str) -> None:
    """
    Delete a saved session file from disk.

    Args:
        session_id: UUID string of the session to delete.
    """
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if session_file.exists():
        session_file.unlink()


def new_session_id() -> str:
    """
    Generate a new unique session ID.

    Returns:
        Short 8-character hex string (subset of UUID4).
    """
    return uuid.uuid4().hex[:8]
