"""
Finance RAG - Phase 2: Markdown Preprocessor

Reads a raw extracted markdown file, applies a series of targeted cleaning
steps, and saves the cleaned version under the processed/ folder.

Input  : markdown/<CompanyName>/<pdf_name>.md
Output : processed/<CompanyName>/<pdf_name>.md

Each cleaning step is a separate function so it can be tested, adjusted,
or skipped independently.
"""

import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Individual cleaning functions
# ---------------------------------------------------------------------------

def remove_html_tags(text: str) -> str:
    """
    Remove any HTML tags left over from the PDF extraction.

    pymupdf4llm occasionally wraps underlined text in <u>...</u> tags.
    This strips all angle-bracket tags while keeping the inner text,
    so '<u>Table of Contents</u>' becomes 'Table of Contents'.

    Args:
        text: Raw markdown string.

    Returns:
        Markdown string with HTML tags removed.
    """
    return re.sub(r"<[^>]+>", "", text)


def remove_page_number_lines(text: str) -> str:
    """
    Remove lines that contain only a standalone page number.

    The extractor inserts bare integers on their own line as page markers
    (e.g. a line that is just '2' or '154'). These carry no content and
    clutter the document.

    A line is treated as a page-number line if, after stripping whitespace,
    it contains nothing but digits.

    Args:
        text: Markdown string.

    Returns:
        Markdown string with bare page-number lines removed.
    """
    cleaned_lines = []
    for line in text.splitlines():
        if re.fullmatch(r"\s*\d+\s*", line):
            continue            # drop the line
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def remove_repeated_navigation_lines(text: str) -> str:
    """
    Remove lines that repeat a navigation/header phrase across every page.

    After extraction, phrases like 'Table of Contents' appear as a
    standalone line at the top of almost every page. They add no value
    to a RAG corpus and should be removed.

    The check is case-insensitive and strips surrounding whitespace before
    comparing, so variations like '  Table of Contents  ' are also caught.

    Args:
        text: Markdown string.

    Returns:
        Markdown string with repeated navigation lines removed.
    """
    # Phrases to remove when they appear alone on a line (case-insensitive)
    nav_phrases = [
        "table of contents",
    ]

    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip().lower()
        if stripped in nav_phrases:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def remove_stray_single_words(text: str) -> str:
    """
    Remove lines that contain only a single short word with no semantic value.

    The extractor sometimes drops isolated words like 'low' or 'high' on
    their own line — artefacts from PDF layout columns or footnotes that
    lost their context. A line is considered stray if it:
      - contains only one word (no spaces)
      - is NOT a markdown heading (does not start with #)
      - is short enough to not be real content (≤ 6 characters)

    This is intentionally conservative to avoid removing valid short lines.

    Args:
        text: Markdown string.

    Returns:
        Markdown string with stray single-word lines removed.
    """
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # Keep markdown headings, table rows, list items, and blank lines
        if (
            stripped.startswith("#")
            or stripped.startswith("|")
            or stripped.startswith("-")
            or stripped.startswith("*")
            or stripped == ""
        ):
            cleaned_lines.append(line)
            continue

        # Drop if it's a single word of 6 characters or fewer
        words = stripped.split()
        if len(words) == 1 and len(stripped) <= 6:
            continue

        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def normalize_blank_lines(text: str) -> str:
    """
    Collapse runs of more than two consecutive blank lines into exactly one.

    The extractor inserts many blank lines between page sections. More than
    one blank line in a row is unnecessary in markdown and makes the file
    harder to read and chunk later.

    Preserves single blank lines (paragraph separators) and the one blank
    line required between a heading and body text.

    Args:
        text: Markdown string.

    Returns:
        Markdown string with excessive blank lines collapsed.
    """
    # Replace 3+ consecutive newlines with exactly 2 (= one blank line)
    return re.sub(r"\n{3,}", "\n\n", text)


def normalize_spaces(text: str) -> str:
    """
    Remove duplicate spaces within lines of text.

    Keeps the content of each line intact but replaces any run of 2+
    spaces with a single space. Does NOT touch leading indentation
    (important for nested lists) or table cell padding.

    Blank lines and markdown table rows are left unchanged.

    Args:
        text: Markdown string.

    Returns:
        Markdown string with duplicate inline spaces removed.
    """
    cleaned_lines = []
    for line in text.splitlines():
        # Leave table rows and blank lines as-is
        if line.strip().startswith("|") or line.strip() == "":
            cleaned_lines.append(line)
            continue
        # Preserve leading whitespace, clean the rest
        leading = len(line) - len(line.lstrip())
        prefix = line[:leading]
        body = re.sub(r" {2,}", " ", line[leading:])
        cleaned_lines.append(prefix + body)
    return "\n".join(cleaned_lines)


def strip_leading_trailing_whitespace(text: str) -> str:
    """
    Remove leading and trailing blank lines from the entire document.

    Ensures the file starts with real content and ends with a single newline.

    Args:
        text: Markdown string.

    Returns:
        Stripped markdown string ending with exactly one newline.
    """
    return text.strip() + "\n"


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def clean_markdown(raw_text: str) -> str:
    """
    Run all cleaning steps in sequence on the raw markdown text.

    Order matters:
    1. Strip HTML tags first so later steps see plain text.
    2. Remove page numbers before blank-line collapsing (they create blank lines).
    3. Remove repeated navigation lines.
    4. Remove stray single words.
    5. Collapse blank lines.
    6. Normalize spaces within lines.
    7. Final strip of document edges.

    Args:
        raw_text: Raw markdown string from extraction.

    Returns:
        Cleaned markdown string.
    """
    text = raw_text
    text = remove_html_tags(text)
    text = remove_page_number_lines(text)
    text = remove_repeated_navigation_lines(text)
    text = remove_stray_single_words(text)
    text = normalize_blank_lines(text)
    text = normalize_spaces(text)
    text = strip_leading_trailing_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_markdown(md_path: Path) -> str:
    """
    Read a markdown file from disk and return its content as a string.

    Args:
        md_path: Path to the .md file.

    Returns:
        File content as a UTF-8 string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")
    return md_path.read_text(encoding="utf-8")


def save_markdown(text: str, output_path: Path) -> None:
    """
    Write a cleaned markdown string to disk, creating directories as needed.

    Args:
        text:        Cleaned markdown content.
        output_path: Destination .md file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def preprocess(md_path: str | Path) -> Path:
    """
    Full preprocessing pipeline for a single markdown file.

    Derives the output path automatically:
        markdown/<Company>/<name>.md  →  processed/<Company>/<name>.md

    Args:
        md_path: Path to the raw markdown file under markdown/.

    Returns:
        Path to the saved cleaned file.
    """
    md_path = Path(md_path).resolve()

    # Derive output path: replace 'markdown' folder with 'processed'
    # Works regardless of how deep the path is
    parts = md_path.parts
    try:
        md_idx = next(i for i, p in enumerate(parts) if p == "markdown")
    except StopIteration:
        raise ValueError(
            f"Expected 'markdown' in path but got: {md_path}\n"
            "Place your file under a folder named 'markdown'."
        )

    output_path = Path(*parts[:md_idx]) / "processed" / Path(*parts[md_idx + 1:])

    print(f"Input     : {md_path}")
    print(f"Output    : {output_path}")
    print()

    raw_text = load_markdown(md_path)
    print(f"Raw size  : {len(raw_text):,} characters")

    cleaned_text = clean_markdown(raw_text)
    print(f"Clean size: {len(cleaned_text):,} characters")
    print(f"Reduction : {100 * (1 - len(cleaned_text) / len(raw_text)):.1f}%")

    save_markdown(cleaned_text, output_path)
    print(f"\nSaved to  : {output_path}")

    return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Explicit path provided
        target_files = [Path(sys.argv[1])]
    else:
        # Auto-discover all .md files under markdown/
        base_dir = Path(__file__).parent / "markdown"
        target_files = list(base_dir.rglob("*.md"))
        if not target_files:
            print(f"No .md files found under {base_dir}")
            sys.exit(1)

    print(f"Found {len(target_files)} markdown file(s) to process.\n")
    print("=" * 60)

    for md_file in target_files:
        try:
            preprocess(md_file)
        except Exception as e:
            print(f"ERROR processing {md_file.name}: {e}")
        print("=" * 60)
