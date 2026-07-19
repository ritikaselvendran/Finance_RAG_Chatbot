"""
Finance RAG - PDF to Markdown Extractor
Uses pymupdf4llm to extract full PDF content (text + tables) and saves as a single markdown file.

Output structure:
    markdown/<CompanyName>/<pdf_name>.md

PDF naming convention expected: <CompanyName>_<Year>_<ReportType>.pdf
Example: 3M_2015_10K.pdf  →  markdown/3M/3M_2015_10K.md
"""

from pathlib import Path
import pymupdf4llm


def extract_pdf_to_markdown(pdf_path: str | Path) -> Path:
    """
    Extract a PDF file to a single markdown file using pymupdf4llm.

    Args:
        pdf_path: Path to the input PDF file.

    Returns:
        Path to the saved markdown file.
    """
    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Parse company name from filename: 3M_2015_10K.pdf → company = "3M"
    company_name = pdf_path.stem.split("_")[0]

    # Output: <project_root>/markdown/<CompanyName>/<pdf_name>.md
    output_dir = pdf_path.parent.parent / "markdown" / company_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_md_path = output_dir / f"{pdf_path.stem}.md"

    print(f"PDF       : {pdf_path}")
    print(f"Company   : {company_name}")
    print(f"Output    : {output_md_path}")
    print()

    print("Converting PDF to markdown...")

    # pymupdf4llm extracts text, headings, and tables directly from the PDF
    # text layer — no vision models, no image rendering, no memory issues.
    # write_images=False ensures no bitmap rendering happens at all.
    md_text = pymupdf4llm.to_markdown(
        str(pdf_path),
        write_images=False,   # don't render/save page images
        show_progress=True,   # show page-by-page progress
    )

    output_md_path.write_text(md_text, encoding="utf-8")

    print(f"\nDone. Markdown saved to: {output_md_path}")
    print(f"File size : {output_md_path.stat().st_size / 1024:.1f} KB")

    return output_md_path


if __name__ == "__main__":
    import sys

    # If a path is passed as argument, use it. Otherwise process all PDFs in data/
    if len(sys.argv) > 1:
        pdf_files = [Path(sys.argv[1])]
    else:
        data_dir = Path(__file__).parent / "data"
        pdf_files = list(data_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {data_dir}")
            sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s) to process.\n")
    print("=" * 60)

    for pdf in pdf_files:
        try:
            extract_pdf_to_markdown(pdf)
        except Exception as e:
            print(f"ERROR processing {pdf.name}: {e}")
        print("=" * 60)
