"""
Finance RAG - Phase 4: Hierarchical Chunking

Why hierarchical chunking beats RecursiveCharacterTextSplitter alone:
----------------------------------------------------------------------
RecursiveCharacterTextSplitter splits purely by character count with no
understanding of document structure. It will:
  - Cut mid-sentence or mid-table
  - Separate a heading from its content into different chunks
  - Produce a chunk with numbers but no context about what they mean

Hierarchical chunking works in two passes:
  Pass 1 — MarkdownHeaderTextSplitter
            Splits at heading boundaries (# ## ###), so each chunk is a
            complete, self-contained section. A chunk titled "Revenue" will
            contain all the revenue content, never be split from its heading.

  Pass 2 — RecursiveCharacterTextSplitter
            Any section larger than 800 chars is further split by size.
            This keeps chunks within embedding model token limits while
            still starting at a logical boundary.

Result: every chunk is topically coherent AND within size limits.
The section heading is stored in metadata so the LLM always knows what
topic a chunk belongs to, even after splitting.

Input  : processed/<CompanyName>/<pdf_name>.md  (via load_documents.py)
Output : chunks/<CompanyName>/<pdf_name>_chunks.pkl
"""

import pickle
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import Phase 3 loader so we reuse the same document loading logic
from load_documents import load_all_documents


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Heading levels MarkdownHeaderTextSplitter will split on.
# Each tuple is (markdown_marker, metadata_key_name).
HEADERS_TO_SPLIT_ON = [
    ("#",   "heading_1"),
    ("##",  "heading_2"),
    ("###", "heading_3"),
]

# RecursiveCharacterTextSplitter settings for oversized sections
CHUNK_SIZE    = 2000  # larger size keeps tables attached to their headings
CHUNK_OVERLAP = 200   # enough overlap so context isn't lost at boundaries


# ---------------------------------------------------------------------------
# Pass 1 — split by markdown headings
# ---------------------------------------------------------------------------

def split_by_headings(document: Document) -> list[Document]:
    """
    Split a single Document into sections using markdown headings.

    Uses MarkdownHeaderTextSplitter which understands # ## ### markers.
    Each resulting Document contains one logical section of the report.
    The heading text is stored in the Document's metadata under the keys
    defined in HEADERS_TO_SPLIT_ON (heading_1, heading_2, heading_3).

    The original document's metadata (company_name, report_year, etc.)
    is merged into every section Document so traceability is preserved.

    Args:
        document: A full-document LangChain Document from Phase 3.

    Returns:
        List of section-level Documents, one per heading-delimited section.
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,   # keep the heading line inside page_content
    )

    sections = splitter.split_text(document.page_content)

    # Merge original document metadata into each section
    enriched = []
    for section in sections:
        combined_metadata = {**document.metadata, **section.metadata}
        enriched.append(
            Document(
                page_content=section.page_content,
                metadata=combined_metadata,
            )
        )

    return enriched


# ---------------------------------------------------------------------------
# Pass 2 — split oversized sections by character count
# ---------------------------------------------------------------------------

def split_large_sections(
    sections: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Further split any section that exceeds chunk_size characters.

    Sections that are already within the size limit are returned as-is.
    Sections that exceed it are run through RecursiveCharacterTextSplitter,
    which tries to split at paragraph → sentence → word boundaries in that
    order to avoid cutting mid-sentence.

    All metadata from the parent section (including heading info) is copied
    to every sub-chunk produced, so context is never lost.

    Args:
        sections:      List of section Documents from split_by_headings().
        chunk_size:    Maximum characters allowed per final chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of final chunk Documents within the size limit.
    """
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # try these in order
    )

    final_chunks = []
    for section in sections:
        if len(section.page_content) <= chunk_size:
            # Section is small enough — keep it whole
            final_chunks.append(section)
        else:
            # Section too large — split further, carry metadata forward
            sub_chunks = char_splitter.split_documents([section])
            final_chunks.extend(sub_chunks)

    return final_chunks


# ---------------------------------------------------------------------------
# Pass 3 — filter out empty / near-empty chunks
# ---------------------------------------------------------------------------

def filter_empty_chunks(
    chunks: list[Document],
    min_chars: int = 80,
) -> list[Document]:
    """
    Remove chunks whose page_content is too short to be useful.

    MarkdownHeaderTextSplitter sometimes produces chunks that contain
    only a heading line and nothing else, e.g.:
        "## ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS"
    These heading-only chunks embed poorly — their vector represents the
    heading label, not financial content — so the retriever never finds
    them for data queries. Storing them wastes index space and pollutes
    results with empty matches.

    A chunk is kept only if its text (after stripping markdown symbols
    and whitespace) is at least min_chars characters long.

    Args:
        chunks:    List of chunks after both splitting passes.
        min_chars: Minimum meaningful character count (default 80).

    Returns:
        Filtered list with only substantive chunks.
    """
    import re
    kept    = []
    dropped = 0
    for chunk in chunks:
        clean = re.sub(r"[#*_\-|`\s]", "", chunk.page_content)
        if len(clean) >= min_chars:
            kept.append(chunk)
        else:
            dropped += 1
    if dropped:
        print(f"  Filtered {dropped} empty/near-empty chunks.")
    return kept


# ---------------------------------------------------------------------------
# Metadata enrichment
# ---------------------------------------------------------------------------

def add_chunk_metadata(chunks: list[Document]) -> list[Document]:
    """
    Add chunk-level metadata to every Document after splitting is complete.

    New fields added:
      section       : human-readable label built from heading_1 / heading_2
                      e.g. "3M COMPANY FORM 10-K > Management Discussion"
      chunk_id      : unique UUID string for this chunk (stable identifier
                      for vector store upserts and deduplication)
      chunk_number  : 1-based position of this chunk in the full list
      total_chunks  : total number of chunks across the entire document

    These fields are used downstream for:
      - chunk_id      → vector store document ID
      - section       → LLM prompt context ("this chunk is from section X")
      - chunk_number  → ordering / pagination in the UI

    Args:
        chunks: List of chunks after both splitting passes.

    Returns:
        Same list with metadata enriched in place.
    """
    total = len(chunks)

    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata

        # Build a readable section label from available heading metadata
        heading_parts = []
        for key in ("heading_1", "heading_2", "heading_3"):
            val = meta.get(key, "").strip()
            # Strip markdown bold markers (**) for cleaner display
            val = val.replace("**", "").strip()
            if val:
                heading_parts.append(val)

        section_label = " > ".join(heading_parts) if heading_parts else "General"

        chunk.metadata["section"]      = section_label
        chunk.metadata["chunk_id"]     = str(uuid.uuid4())
        chunk.metadata["chunk_number"] = i
        chunk.metadata["total_chunks"] = total

    return chunks


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_chunks(chunks: list[Document], output_path: Path) -> None:
    """
    Persist the list of chunk Documents to disk using pickle.

    Pickle is used here because LangChain Document objects are Python
    objects — pickle serialises them exactly, preserving all metadata
    and page_content without any conversion. The file can be loaded back
    with a single pickle.load() call in Phase 5.

    Args:
        chunks:      List of chunk Documents to save.
        output_path: Destination .pkl file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved {len(chunks)} chunks → {output_path}")


def load_chunks(pkl_path: Path) -> list[Document]:
    """
    Load previously saved chunk Documents from a pickle file.

    Provided here so Phase 5 can import and reuse this function
    without reimplementing the load logic.

    Args:
        pkl_path: Path to the .pkl file saved by save_chunks().

    Returns:
        List of LangChain Document objects.
    """
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Sample printer
# ---------------------------------------------------------------------------

def print_sample_chunk(chunks: list[Document], index: int = 0) -> None:
    """
    Print a formatted summary of one chunk for visual verification.

    Shows all metadata fields and the full page_content of the selected
    chunk so you can confirm headings, section labels, and content look
    correct before moving to embedding.

    Args:
        chunks: List of all chunk Documents.
        index:  Which chunk to display (default: first chunk).
    """
    chunk = chunks[index]
    print("\n" + "=" * 60)
    print(f"SAMPLE CHUNK (index {index})")
    print("=" * 60)
    print("\n--- Metadata ---")
    for key, value in chunk.metadata.items():
        print(f"  {key:<15}: {value}")
    print("\n--- page_content ---")
    print(chunk.page_content)
    print("=" * 60)


# ---------------------------------------------------------------------------
# Full pipeline for one document
# ---------------------------------------------------------------------------

def chunk_document(document: Document) -> list[Document]:
    """
    Run the full two-pass chunking pipeline on a single Document.

    Pass 1: split_by_headings    — logical section boundaries
    Pass 2: split_large_sections — size limit enforcement
    Pass 3: add_chunk_metadata   — section label, chunk_id, numbering

    Args:
        document: Full-document LangChain Document from Phase 3.

    Returns:
        List of final chunk Documents ready for embedding.
    """
    sections = split_by_headings(document)
    chunks   = split_large_sections(sections)
    chunks   = filter_empty_chunks(chunks)
    chunks   = add_chunk_metadata(chunks)
    return chunks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root  = Path(__file__).parent
    processed_dir = project_root / "processed"
    chunks_dir    = project_root / "chunks"

    # --- Load Phase 3 documents ---
    print("Loading documents from processed/...\n")
    documents = load_all_documents(processed_dir)
    print(f"\nTotal documents loaded: {len(documents)}\n")

    all_chunks: list[Document] = []

    for doc in documents:
        company = doc.metadata.get("company_name", "unknown")
        year    = doc.metadata.get("report_year",  "unknown")
        stem    = Path(doc.metadata["source_file"]).stem

        print(f"Chunking: {company} {year}")

        # Run two-pass chunking
        sections = split_by_headings(doc)
        chunks   = split_large_sections(sections)
        chunks   = filter_empty_chunks(chunks)
        chunks   = add_chunk_metadata(chunks)

        print(f"  Sections (after heading split) : {len(sections)}")
        print(f"  Final chunks (after size split): {len(chunks)}")

        # Save per-document pickle
        out_path = chunks_dir / company / f"{stem}_chunks.pkl"
        save_chunks(chunks, out_path)

        all_chunks.extend(chunks)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("CHUNKING SUMMARY")
    print("=" * 60)
    print(f"  Total documents : {len(documents)}")
    print(f"  Total chunks    : {len(all_chunks)}")
    avg = sum(len(c.page_content) for c in all_chunks) / len(all_chunks)
    print(f"  Avg chunk size  : {avg:.0f} characters")

    # --- Print a sample chunk ---
    print_sample_chunk(all_chunks, index=5)
