"""
Finance RAG - Phase 3: Load Cleaned Markdown into LangChain Documents

Reads cleaned markdown files from processed/ and wraps each file into a
LangChain Document object with structured metadata.

Why LangChain Document objects?
--------------------------------
A LangChain Document is a simple container with two fields:
  - page_content : the text the LLM will read
  - metadata     : a dict of structured info that travels with the text

Every step after this — chunking, embedding, vector storage, retrieval —
operates on Document objects. The metadata is preserved through all of
those steps, so when the retriever returns a chunk to the LLM, you always
know which company, year, and file it came from. Without metadata, a
retrieved chunk is anonymous and you cannot build a traceable RAG answer.

Input  : processed/<CompanyName>/<pdf_name>.md
Output : list of LangChain Document objects (in memory, ready for Phase 4)

PDF naming convention expected: <CompanyName>_<Year>_<ReportType>.pdf
Example: 3M_2015_10K.md → company=3M, year=2015, report_type=10K
"""

from pathlib import Path
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def parse_metadata_from_filename(md_path: Path) -> dict:
    """
    Derive structured metadata from the markdown filename.

    Expected filename format: <CompanyName>_<Year>_<ReportType>.md
    Example: 3M_2015_10K.md

    Fields extracted:
      source_file  : full resolved path as a string (for traceability)
      company_name : first segment before '_'  → "3M"
      report_year  : second segment            → "2015"
      report_type  : third segment             → "10K"

    If the filename does not follow the convention, company_name is set
    to the parent folder name and year/type are set to "unknown" so the
    pipeline does not crash on unexpected filenames.

    Args:
        md_path: Path to the cleaned .md file.

    Returns:
        Dictionary of metadata key-value pairs.
    """
    stem_parts = md_path.stem.split("_")

    if len(stem_parts) >= 3:
        company_name = stem_parts[0]
        report_year  = stem_parts[1]
        report_type  = stem_parts[2]
    else:
        # Fallback: use parent folder name as company, rest unknown
        company_name = md_path.parent.name
        report_year  = "unknown"
        report_type  = "unknown"

    return {
        "source_file"  : str(md_path.resolve()),
        "company_name" : company_name,
        "report_year"  : report_year,
        "report_type"  : report_type,
    }


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_markdown_text(md_path: Path) -> str:
    """
    Read the cleaned markdown file from disk and return its content.

    Args:
        md_path: Path to the .md file under processed/.

    Returns:
        Full file content as a UTF-8 string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not md_path.exists():
        raise FileNotFoundError(f"Processed markdown not found: {md_path}")
    return md_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Document creation
# ---------------------------------------------------------------------------

def create_document(md_path: Path) -> Document:
    """
    Wrap a single cleaned markdown file into a LangChain Document.

    The full file content goes into page_content.
    Metadata is parsed from the filename using parse_metadata_from_filename().

    At this stage (before chunking) each Document represents one complete
    financial report. Phase 4 will split these into smaller chunks while
    keeping the same metadata attached to every chunk.

    Args:
        md_path: Path to the cleaned .md file.

    Returns:
        A LangChain Document with page_content and metadata populated.
    """
    text     = load_markdown_text(md_path)
    metadata = parse_metadata_from_filename(md_path)

    return Document(
        page_content=text,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Batch loader
# ---------------------------------------------------------------------------

def load_all_documents(processed_dir: str | Path) -> list[Document]:
    """
    Discover and load every .md file under the processed/ directory tree.

    Walks the entire processed/ folder recursively so adding a new company
    or year is as simple as dropping a new file in the right subfolder —
    no code changes needed.

    Args:
        processed_dir: Root folder that contains processed markdown files.
                       Typically <project_root>/processed/

    Returns:
        List of LangChain Document objects, one per .md file found.

    Raises:
        FileNotFoundError: If processed_dir does not exist.
    """
    processed_dir = Path(processed_dir).resolve()

    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed directory not found: {processed_dir}\n"
            "Run preprocess_markdown.py first."
        )

    md_files = sorted(processed_dir.rglob("*.md"))

    if not md_files:
        raise FileNotFoundError(
            f"No .md files found under {processed_dir}\n"
            "Run preprocess_markdown.py first."
        )

    documents = []
    for md_file in md_files:
        doc = create_document(md_file)
        documents.append(doc)
        print(f"  Loaded: {md_file.relative_to(processed_dir)}  "
              f"({len(doc.page_content):,} chars)")

    return documents


# ---------------------------------------------------------------------------
# Sample printer
# ---------------------------------------------------------------------------

def print_sample_document(doc: Document, preview_chars: int = 500) -> None:
    """
    Print a readable summary of one Document to verify loading worked.

    Shows all metadata fields and the first N characters of page_content.
    This is useful to confirm the metadata is correctly parsed and the
    content looks clean before moving to the embedding step.

    Args:
        doc:           The Document to inspect.
        preview_chars: How many characters of page_content to show.
    """
    print("\n" + "=" * 60)
    print("SAMPLE DOCUMENT")
    print("=" * 60)
    print("\n--- Metadata ---")
    for key, value in doc.metadata.items():
        print(f"  {key:<15}: {value}")
    print(f"\n--- page_content preview (first {preview_chars} chars) ---")
    print(doc.page_content[:preview_chars])
    print("...")
    print(f"\nTotal page_content length: {len(doc.page_content):,} characters")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root  = Path(__file__).parent
    processed_dir = project_root / "processed"

    print("Loading documents from processed/...\n")

    documents = load_all_documents(processed_dir)

    print(f"\nTotal documents loaded: {len(documents)}")

    # Print a sample of the first document to verify correctness
    print_sample_document(documents[0])
