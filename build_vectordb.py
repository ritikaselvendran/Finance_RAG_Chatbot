"""
Finance RAG - Phase 6: Build Persistent ChromaDB Vector Store

Loads paired (Document + embedding) data from Phase 5 and inserts
everything into a persistent ChromaDB collection saved to disk.

How ChromaDB stores data:
--------------------------
ChromaDB uses two storage layers:

  1. SQLite (chroma.sqlite3)
     Stores chunk text (documents), metadata dicts, and chunk IDs.
     Lightweight, human-inspectable, fast for small lookups.

  2. HNSW binary index (<collection-uuid>/data_level0.bin)
     Stores the raw embedding vectors in a Hierarchical Navigable Small
     World graph. Each vector is a node; edges connect nearby vectors.
     At query time, the graph is traversed to find nearest neighbours
     without comparing to every stored vector — O(log n) instead of O(n).

When you query ChromaDB:
  query embedding → HNSW finds nearest vector IDs
                  → SQLite returns text + metadata for those IDs
                  → LangChain returns them as Document objects

Persistence: because we pass a path to chromadb.PersistentClient(), the
database survives across Python sessions. Phase 7 loads the same path
and the data is already there — no re-embedding needed.

Input  : embeddings/<CompanyName>/<pdf_name>_embeddings.pkl
Output : vectordb/   (persistent ChromaDB on disk)
"""

import pickle
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COLLECTION_NAME = "finance_rag"   # logical name for the ChromaDB collection


# ---------------------------------------------------------------------------
# Step 1 — Load embeddings from pickle
# ---------------------------------------------------------------------------

def load_paired_embeddings(pkl_path: Path) -> list[dict]:
    """
    Load the paired (Document + embedding) list saved by Phase 5.

    Each entry is a dict:
        {
            "document" : Document(page_content=..., metadata={...}),
            "embedding": [float, ...]    # 384 floats
        }

    Args:
        pkl_path: Path to the _embeddings.pkl file from Phase 5.

    Returns:
        List of paired dicts.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {pkl_path}\n"
            "Run generate_embeddings.py first."
        )
    with open(pkl_path, "rb") as f:
        paired = pickle.load(f)
    print(f"  Loaded {len(paired)} paired entries from {pkl_path.name}")
    return paired


# ---------------------------------------------------------------------------
# Step 2 — Create or connect to ChromaDB
# ---------------------------------------------------------------------------

def get_chroma_collection(
    db_path: Path,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """
    Create a persistent ChromaDB client and return the collection.

    PersistentClient saves everything to disk at db_path so the database
    survives between Python sessions. If the collection already exists
    (re-run scenario), get_or_create_collection returns the existing one
    without duplicating data — we handle deduplication via chunk_id later.

    ChromaDB folder structure after creation:
        vectordb/
        ├── chroma.sqlite3           ← text, metadata, IDs
        └── <uuid>/
            └── data_level0.bin      ← HNSW vector index

    Args:
        db_path:         Folder path where ChromaDB persists its files.
        collection_name: Logical name for the collection (like a table name).

    Returns:
        ChromaDB Collection object ready for insert/query operations.
    """
    db_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(anonymized_telemetry=False),  # no usage tracking
    )

    # Delete existing collection so we always start fresh on rebuild.
    # This prevents stale chunks from previous runs accumulating in the store.
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection '{collection_name}' for clean rebuild.")

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # use cosine similarity for search
    )

    return collection


# ---------------------------------------------------------------------------
# Step 3 — Build metadata dict for ChromaDB
# ---------------------------------------------------------------------------

def build_chroma_metadata(doc: Document) -> dict:
    """
    Extract and normalise metadata fields for ChromaDB storage.

    ChromaDB metadata values must be str, int, float, or bool — no nested
    dicts or lists. This function pulls the required fields from the
    LangChain Document's metadata dict and renames them to the final
    schema used in Phase 7 for filtering.

    Required output fields:
        company     : company name (e.g. "3M")
        year        : report year  (e.g. "2015")
        report_type : filing type  (e.g. "10K")
        section     : heading path (e.g. "MD&A > Revenue")
        chunk_id    : UUID string  (used as ChromaDB document ID)
        source      : original file path for citation

    Any field missing from the Document metadata falls back to "unknown"
    so the insert never crashes on incomplete metadata.

    Args:
        doc: LangChain Document with metadata from Phases 3 and 4.

    Returns:
        Flat dict of ChromaDB-compatible metadata values.
    """
    m = doc.metadata
    return {
        "company"     : str(m.get("company_name", "unknown")),
        "year"        : str(m.get("report_year",  "unknown")),
        "report_type" : str(m.get("report_type",  "unknown")),
        "section"     : str(m.get("section",      "unknown")),
        "chunk_id"    : str(m.get("chunk_id",      "unknown")),
        "source"      : str(m.get("source_file",  "unknown")),
        "chunk_number": int(m.get("chunk_number", 0)),
        "total_chunks": int(m.get("total_chunks", 0)),
    }


# ---------------------------------------------------------------------------
# Step 4 — Insert data into ChromaDB in batches
# ---------------------------------------------------------------------------

def insert_into_chroma(
    collection: chromadb.Collection,
    paired: list[dict],
    batch_size: int = 500,
) -> None:
    """
    Insert all chunks (text + embedding + metadata) into ChromaDB.

    ChromaDB's add() method takes parallel lists:
        ids        : unique string ID per chunk  (we use chunk_id from metadata)
        documents  : the chunk text              (page_content)
        embeddings : the 384-float vectors       (from Phase 5)
        metadatas  : list of flat metadata dicts (company, year, section, ...)

    We insert in batches of 500 to avoid memory spikes on large datasets.
    upsert() is used instead of add() so re-running the script updates
    existing entries rather than raising a duplicate-ID error.

    Args:
        collection: ChromaDB Collection to insert into.
        paired:     List of {document, embedding} dicts from Phase 5.
        batch_size: Number of chunks per insert call.
    """
    total   = len(paired)
    batches = (total + batch_size - 1) // batch_size   # ceiling division

    print(f"\nInserting {total} chunks into ChromaDB "
          f"(batch_size={batch_size}, {batches} batches)...")

    for batch_idx in range(batches):
        start = batch_idx * batch_size
        end   = min(start + batch_size, total)
        batch = paired[start:end]

        ids        = []
        documents  = []
        embeddings = []
        metadatas  = []

        for entry in batch:
            doc      = entry["document"]
            vec      = entry["embedding"]
            meta     = build_chroma_metadata(doc)

            ids.append(meta["chunk_id"])
            documents.append(doc.page_content)
            embeddings.append(vec)
            metadatas.append(meta)

        # upsert = insert if new, update if ID already exists
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(f"  Batch {batch_idx + 1}/{batches} — "
              f"chunks {start + 1}–{end} inserted.")

    print(f"\nAll {total} chunks stored in ChromaDB.")


# ---------------------------------------------------------------------------
# Step 5 — Verify and print summary
# ---------------------------------------------------------------------------

def print_db_summary(
    collection: chromadb.Collection,
    db_path: Path,
    sample_index: int = 5,
) -> None:
    """
    Query ChromaDB to confirm data was stored and print a summary.

    Fetches the total count of stored chunks directly from the collection,
    then retrieves one sample entry to display its stored text and metadata.
    This confirms the round-trip: insert → persist → fetch works correctly.

    Args:
        collection:   ChromaDB Collection to inspect.
        db_path:      Folder where the database is persisted.
        sample_index: Offset into the collection for the sample entry.
    """
    total_stored = collection.count()

    print("\n" + "=" * 60)
    print("CHROMADB SUMMARY")
    print("=" * 60)
    print(f"  Database location : {db_path.resolve()}")
    print(f"  Collection name   : {collection.name}")
    print(f"  Total chunks stored: {total_stored}")

    # Fetch one sample to verify stored content
    results = collection.get(
        limit=1,
        offset=min(sample_index, max(0, total_stored - 1)),
        include=["documents", "metadatas", "embeddings"],
    )

    if results["ids"]:
        print(f"\n--- Sample entry (offset {sample_index}) ---")
        print(f"  chunk_id  : {results['ids'][0]}")
        print(f"\n  Metadata:")
        for key, value in results["metadatas"][0].items():
            print(f"    {key:<14}: {value}")
        print(f"\n  Text preview (first 200 chars):")
        print(f"    {results['documents'][0][:200]}...")
        vec_preview = [round(v, 6) for v in results["embeddings"][0][:6]]
        print(f"\n  Embedding preview (first 6 floats):")
        print(f"    {vec_preview}  ...")

    print("=" * 60)

    # Show disk usage
    sqlite_file = db_path / "chroma.sqlite3"
    if sqlite_file.exists():
        size_mb = sqlite_file.stat().st_size / (1024 * 1024)
        print(f"\n  SQLite size : {size_mb:.2f} MB  ({sqlite_file})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root   = Path(__file__).parent
    embeddings_dir = project_root / "embeddings"
    db_path        = project_root / "vectordb"

    # Discover all embedding pickle files from Phase 5
    embedding_files = sorted(embeddings_dir.rglob("*_embeddings.pkl"))

    if not embedding_files:
        print(f"No embedding files found under {embeddings_dir}")
        print("Run generate_embeddings.py first.")
        raise SystemExit(1)

    print(f"Found {len(embedding_files)} embedding file(s).\n")

    # Connect to (or create) the persistent ChromaDB collection
    collection = get_chroma_collection(db_path)
    print(f"ChromaDB collection '{COLLECTION_NAME}' ready at: {db_path}\n")

    # Load and insert all embedding files
    print("=" * 60)
    for emb_file in embedding_files:
        print(f"Processing: {emb_file.name}")
        paired = load_paired_embeddings(emb_file)
        insert_into_chroma(collection, paired)
        print("=" * 60)

    # Final verification
    print_db_summary(collection, db_path)
