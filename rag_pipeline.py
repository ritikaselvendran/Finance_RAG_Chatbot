"""
Finance RAG - Phase 9: Full RAG Pipeline with multi-year trend support.

Detects whether a query is asking for a trend across multiple years
and routes it through get_trend_documents() instead of the standard
top-K retriever.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from retrieve_chunks import (
    load_chroma_collection,
    load_embedding_model,
    FinanceRetriever,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOP_K           = 5
GROQ_MODEL      = "llama-3.3-70b-versatile"
COLLECTION_NAME = "finance_rag"

# All companies stored in the DB (uppercase, matches metadata)
KNOWN_COMPANIES = [
    "3M", "ACTIVISIONBLIZZARD", "ADOBE", "AES", "AMAZON", "AMCOR", "AMD",
    "AMERICANEXPRESS", "AMERICANWATERWORKS", "APPLE", "BESTBUY", "BLOCK",
    "BOEING", "BOSTONPROPERTIES", "COCACOLA", "CORNING", "COSTCO",
    "CVSHEALTH", "EBAY", "FEDEX", "FOOTLOCKER", "GENERALMILLS", "INTEL",
    "JOHNSON", "JPMORGAN", "KRAFTHEINZ", "LOCKHEEDMARTIN", "MCDONALDS",
    "MGMRESORTS", "MICROSOFT", "NETFLIX", "NIKE", "ORACLE", "PAYPAL",
    "PEPSICO", "PFIZER", "PG", "SALESFORCE", "ULTABEAUTY", "VERIZON",
    "WALMART",
]


# ---------------------------------------------------------------------------
# API key loader
# ---------------------------------------------------------------------------

def load_api_key() -> str:
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path, override=False)
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY not set in .env")
    return api_key


# ---------------------------------------------------------------------------
# Query analyser — detects company, years, and whether it's a trend query
# ---------------------------------------------------------------------------

def analyse_query(query: str) -> dict:
    """
    Parse the user query to detect company name, year range, and query type.

    Detects:
      - Company name: scans query tokens against KNOWN_COMPANIES list
      - Individual years: any 4-digit number between 2010–2030
      - Year range: "from YYYY to YYYY" or "YYYY to YYYY" or "YYYY-YYYY"
      - is_trend: True if 2+ years found or explicit range detected

    Returns a dict:
        {
            "company"  : str | None,   e.g. "AMAZON"
            "years"    : list[str],    e.g. ["2018","2019","2020","2021","2022"]
            "is_trend" : bool,         True if multi-year question
        }
    """
    q_upper = query.upper()

    # Detect company
    company = None
    for c in KNOWN_COMPANIES:
        if c in q_upper:
            company = c
            break

    # Detect year range: "from 2018 to 2022" or "2018-2022" or "2018 to 2022"
    years = []
    range_match = re.search(
        r'\b(20\d{2})\s*(?:to|-|through|–)\s*(20\d{2})\b',
        query, re.IGNORECASE
    )
    if range_match:
        start_yr = int(range_match.group(1))
        end_yr   = int(range_match.group(2))
        years    = [str(y) for y in range(start_yr, end_yr + 1)]
    else:
        # Detect individual years mentioned
        years = list(dict.fromkeys(re.findall(r'\b(20\d{2})\b', query)))

    is_trend = len(years) > 1 or range_match is not None

    return {"company": company, "years": years, "is_trend": is_trend}


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

def build_prompt_template() -> PromptTemplate:
    template = """You are an AI assistant specialized in analyzing financial reports.

Instructions:
- Answer ONLY using the information provided in the context below.
- Never hallucinate or add information not present in the context.
- If the answer is not available in the context, reply exactly:
  "The requested information is not available in the provided financial report."
- When the answer is available, always mention:
    * The section it came from
    * The source document name
- For trend questions covering multiple years, present the data year by year.
- Be concise and precise. For financial figures, include exact numbers and units.

---

Context (retrieved from financial reports):
{context}

---

Question: {question}

Answer:"""

    return PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )


# ---------------------------------------------------------------------------
# Context formatter
# ---------------------------------------------------------------------------

def format_context(docs_and_scores: list[tuple[Document, float]]) -> str:
    parts = []
    for i, (doc, score) in enumerate(docs_and_scores, start=1):
        m = doc.metadata
        block = (
            f"[Chunk {i} | Score: {score:.4f}]\n"
            f"Section  : {m.get('section', 'N/A')}\n"
            f"Source   : {Path(m.get('source', 'N/A')).name}\n"
            f"Company  : {m.get('company', 'N/A')}  |  "
            f"Year: {m.get('year', 'N/A')}  |  "
            f"Type: {m.get('report_type', 'N/A')}\n"
            f"{'-' * 40}\n"
            f"{doc.page_content.strip()}\n"
            f"{'=' * 40}"
        )
        parts.append(block)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_llm_answer(llm: ChatGroq, prompt_template: PromptTemplate,
                   context: str, question: str) -> str:
    final_prompt = prompt_template.format(context=context, question=question)
    response     = llm.invoke([HumanMessage(content=final_prompt)])
    return response.content


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_retrieved_chunks(docs_and_scores: list[tuple[Document, float]]) -> None:
    print(f"\n{'='*60}\nRETRIEVED CHUNKS (Top {len(docs_and_scores)})\n{'='*60}")
    for rank, (doc, score) in enumerate(docs_and_scores, start=1):
        m = doc.metadata
        print(f"\n  Rank {rank}  |  Score: {score:.4f}")
        print(f"  Section  : {m.get('section', 'N/A')}")
        print(f"  Source   : {Path(m.get('source', 'N/A')).name}")
        print(f"  Company  : {m.get('company')}  Year: {m.get('year')}  Type: {m.get('report_type')}")
        print(f"  Preview  : {doc.page_content[:120].strip()}...")


def print_final_answer(query: str, answer: str) -> None:
    print(f"\n{'='*60}\nFINAL ANSWER\n{'='*60}")
    print(f"  Question : {query}\n")
    for line in answer.strip().splitlines():
        print(f"  {line}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Full RAG pipeline
# ---------------------------------------------------------------------------

def run_rag_pipeline(query: str) -> tuple[str, list[tuple[Document, float]]]:
    """
    Run the full RAG pipeline for a query.

    Automatically detects trend queries and routes them through
    multi-year retrieval. Single-year or general queries use standard
    top-K retrieval with optional company/year metadata filtering.

    Returns:
        Tuple of (answer_string, docs_and_scores)
    """
    project_root = Path(__file__).parent
    db_path      = project_root / "vectordb"

    api_key    = load_api_key()
    collection = load_chroma_collection(db_path)
    emb_model  = load_embedding_model()

    llm = ChatGroq(
        api_key=api_key,
        model_name=GROQ_MODEL,
        temperature=0.0,
        max_tokens=1024,
    )

    retriever       = FinanceRetriever(collection=collection, model=emb_model, top_k=TOP_K)
    prompt_template = build_prompt_template()
    parsed          = analyse_query(query)

    print(f"\nQuery analysis: {parsed}")

    # Route to appropriate retrieval strategy
    if parsed["is_trend"] and parsed["company"] and parsed["years"]:
        print(f"Trend query detected — fetching per-year chunks for {parsed['years']}")
        docs_and_scores = retriever.get_trend_documents(
            query=query,
            company=parsed["company"],
            years=parsed["years"],
            chunks_per_year=3,
        )
    else:
        docs_and_scores = retriever.get_relevant_documents(
            query=query,
            company=parsed["company"],
            year=parsed["years"][0] if len(parsed["years"]) == 1 else None,
        )

    print_retrieved_chunks(docs_and_scores)

    context = format_context(docs_and_scores)
    answer  = get_llm_answer(llm, prompt_template, context, query)
    print_final_answer(query, answer)

    return answer, docs_and_scores


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    query = "What was Amazon's net income growth trend from 2018 to 2022?"
    run_rag_pipeline(query)
