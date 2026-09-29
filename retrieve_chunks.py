"""
Finance RAG - Phase 7: Retrieval from ChromaDB

Supports both simple queries (top-K across all documents) and
filtered queries (by company, year, or both) for multi-document
and multi-year trend analysis.
"""

from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Configuration — must match Phase 5 and Phase 6 exactly
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME      = "finance_rag"
TOP_K                = 5     # number of chunks to retrieve


# ---------------------------------------------------------------------------
# Step 1 — Load ChromaDB collection
# ---------------------------------------------------------------------------

def load_chroma_collection(
    db_path: Path,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """
    Connect to the existing persistent ChromaDB and return the collection.
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"Vector database not found at: {db_path}\n"
            "Run build_vectordb.py first."
        )

    client = chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(anonymized_telemetry=False),
    )

    collection = client.get_collection(name=collection_name)

    print(f"ChromaDB loaded from  : {db_path.resolve()}")
    print(f"Collection            : {collection_name}")
    print(f"Total chunks in store : {collection.count()}")

    return collection


# ---------------------------------------------------------------------------
# Step 2 — Load the embedding model
# ---------------------------------------------------------------------------

def load_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
) -> SentenceTransformer:
    """
    Load the same sentence-transformer model used in Phase 5.
    """
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Model ready. Dim: {model.get_sentence_embedding_dimension()}")
    return model


# ---------------------------------------------------------------------------
# Step 3 — Embed the query
# ---------------------------------------------------------------------------

def embed_query(query: str, model: SentenceTransformer) -> list[float]:
    """
    Convert the user's query string into a 384-dimensional vector.
    """
    vector = model.encode(query, normalize_embeddings=True)
    return vector.tolist()


# ---------------------------------------------------------------------------
# Step 4 — Query ChromaDB (with optional metadata filter)
# ---------------------------------------------------------------------------

def query_collection(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int = TOP_K,
    where: dict | None = None,
) -> dict:
    """
    Search ChromaDB for the top_k chunks most similar to the query vector.

    Supports optional ChromaDB `where` filter for metadata-based filtering.
    This allows restricting results to a specific company, year, or both.

    ChromaDB where filter examples:
        {"company": "AMAZON"}                          — one company
        {"year": "2022"}                               — one year
        {"$and": [{"company": "AMAZON"},
                  {"year": "2022"}]}                   — company + year

    Args:
        collection:      ChromaDB Collection to search.
        query_embedding: 384-float query vector.
        top_k:           Number of results to return.
        where:           Optional ChromaDB metadata filter dict.

    Returns:
        Raw ChromaDB result dict.
    """
    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    return collection.query(**kwargs)


# ---------------------------------------------------------------------------
# Step 5 — Convert results to LangChain Documents
# ---------------------------------------------------------------------------

def results_to_documents(results: dict) -> list[tuple[Document, float]]:
    """
    Convert raw ChromaDB results into (LangChain Document, score) pairs.
    similarity_score = 1 - cosine_distance  (higher = more relevant)
    """
    docs_and_scores = []

    ids        = results["ids"][0]
    documents  = results["documents"][0]
    metadatas  = results["metadatas"][0]
    distances  = results["distances"][0]

    for doc_id, text, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        similarity_score = round(1 - distance, 4)
        doc = Document(page_content=text, metadata=metadata)
        docs_and_scores.append((doc, similarity_score))

    return docs_and_scores


# ---------------------------------------------------------------------------
# Step 6 — Build ChromaDB where filter
# ---------------------------------------------------------------------------

def build_filter(
    company: str | None = None,
    year: str | None = None,
) -> dict | None:
    """
    Build a ChromaDB metadata filter dict from optional company and year.

    Returns None if neither is specified (no filtering = search all docs).
    Returns a simple equality filter if only one is specified.
    Returns a $and compound filter if both are specified.

    Args:
        company: Company name string e.g. "AMAZON" (must match stored metadata).
        year:    4-digit year string e.g. "2022".

    Returns:
        ChromaDB where filter dict, or None.
    """
    conditions = []
    if company:
        conditions.append({"company": company.upper()})
    if year:
        conditions.append({"year": str(year)})

    if len(conditions) == 0:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ---------------------------------------------------------------------------
# Step 6.5 — Query type detection and sub-query routing
# ---------------------------------------------------------------------------

# Keyword sets for each query type (lowercase for matching)
_FINANCIAL_KEYWORDS = {
    "revenue", "sales", "income", "profit", "loss", "earnings",
    "cash", "operating", "ebitda", "margin", "expense", "cost",
    "assets", "liabilities", "equity", "debt", "dividend", "eps",
    "shares", "capital", "expenditure", "capex", "gross", "net",
    "balance", "sheet", "cash flow", "free cash", "interest",
}

_GOVERNANCE_KEYWORDS = {
    "board", "director", "directors", "executive", "officer",
    "ceo", "cfo", "coo", "cto", "president", "chairman",
    "management", "governance", "team", "leadership", "founded",
    "members", "appointed", "compensation", "salary",
}

_RISK_KEYWORDS = {
    "risk", "risks", "threat", "uncertainty", "factor",
    "challenge", "concern", "exposure", "litigation", "lawsuit",
    "regulatory", "compliance", "competition", "cybersecurity",
}

_STRATEGY_KEYWORDS = {
    "strategy", "strategic", "growth", "plan", "outlook",
    "segment", "product", "service", "market", "business",
    "acquisition", "partnership", "innovation", "research",
    "development", "expansion", "initiative", "priority",
}

_LEGAL_KEYWORDS = {
    "legal", "lawsuit", "litigation", "settlement", "claim",
    "proceeding", "court", "regulatory", "investigation",
    "penalty", "fine", "compliance", "contingency",
}


def detect_query_type(query: str) -> str:
    """
    Detect the type of financial report question being asked.

    Scans the query for domain-specific keywords and returns one of:
        "financial"   — numbers, metrics, statements (default)
        "governance"  — board members, executives, leadership
        "risk"        — risk factors, threats, challenges
        "strategy"    — business strategy, growth, segments
        "legal"       — litigation, legal proceedings, compliance

    Used to select appropriate sub-queries for retrieval. Financial
    sub-queries steer toward income statements and balance sheets, while
    governance sub-queries steer toward Part III sections.

    Args:
        query: User's plain-text question (case-insensitive).

    Returns:
        Query type string.
    """
    q_lower = query.lower()
    words   = set(q_lower.split())

    # Score each category by keyword overlap
    scores = {
        "governance" : len(words & _GOVERNANCE_KEYWORDS),
        "risk"       : len(words & _RISK_KEYWORDS),
        "strategy"   : len(words & _STRATEGY_KEYWORDS),
        "legal"      : len(words & _LEGAL_KEYWORDS),
        "financial"  : len(words & _FINANCIAL_KEYWORDS),
    }

    # Also check multi-word phrases
    if any(p in q_lower for p in ["board of directors", "executive team",
                                   "executive officer", "chief executive",
                                   "board member"]):
        scores["governance"] += 3
    if any(p in q_lower for p in ["risk factor", "risk factors",
                                   "key risk", "major risk"]):
        scores["risk"] += 3
    if any(p in q_lower for p in ["cash flow", "net income", "net sales",
                                   "total revenue", "operating income",
                                   "gross profit", "earnings per share"]):
        scores["financial"] += 3

    # Return the highest scoring type, default to "financial"
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "financial"


def get_sub_queries(query: str, query_type: str) -> list[str]:
    """
    Return a list of sub-queries appropriate for the detected query type.

    The first entry is always the original query (highest semantic fidelity).
    The remaining entries are targeted phrasings that match the section
    headings used in 10-K filings for that topic area.

    Args:
        query:      Original user query.
        query_type: One of "financial", "governance", "risk",
                    "strategy", "legal".

    Returns:
        List of query strings to embed and search with.
    """
    sub_query_map = {
        "financial": [
            query,
            "cash flow from operating activities consolidated statements of cash flows",
            "net income revenue net sales consolidated statements of operations income",
            "total assets liabilities stockholders equity consolidated balance sheet",
            "earnings per share diluted annual selected financial data quarterly",
        ],
        "governance": [
            query,
            "directors executive officers corporate governance board of directors names",
            "executive team leadership CEO CFO COO president chairman members",
            "information about our executive officers directors compensation",
            "part III item 10 directors executive officers corporate governance",
        ],
        "risk": [
            query,
            "risk factors item 1A business risks uncertainties challenges",
            "risk factors that could affect our business results operations",
            "key risks regulatory competition cybersecurity market risk",
            "forward looking statements risks uncertainties factors",
        ],
        "strategy": [
            query,
            "business overview strategy products services markets segments",
            "growth strategy initiatives priorities investments acquisitions",
            "management discussion analysis overview business highlights",
            "item 1 business description products services competition",
        ],
        "legal": [
            query,
            "legal proceedings litigation settlements commitments contingencies",
            "item 3 legal proceedings lawsuits regulatory investigations",
            "commitments contingencies legal claims pending proceedings",
            "note commitments contingencies legal matters",
        ],
    }

    return sub_query_map.get(query_type, sub_query_map["financial"])

def retrieve_across_years(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    query: str,
    company: str,
    years: list[str],
    chunks_per_year: int = 2,
) -> list[tuple[Document, float]]:
    """
    Retrieve chunks for each year separately and combine the results.

    For trend questions like "net income from 2018 to 2022", a standard
    top-5 query fails because all 5 results tend to come from the single
    most similar document. This function queries each year independently
    with a metadata filter so every year gets representation in the context.

    Uses multiple sub-queries per year to improve recall — financial
    figures like "net income" may appear under different section names
    (Consolidated Statements of Operations, Quarterly Data, Selected
    Financial Data) so we try several phrasings and deduplicate.

    Args:
        collection:      ChromaDB Collection.
        model:           SentenceTransformer for query embedding.
        query:           User's question string.
        company:         Company name to filter by (e.g. "AMAZON").
        years:           List of year strings to retrieve from.
        chunks_per_year: How many chunks to fetch per year (default 2).

    Returns:
        Combined list of (Document, score) pairs ordered by year then score.
    """
    all_results  = []
    seen_ids     = set()
    query_type   = detect_query_type(query)
    sub_queries  = get_sub_queries(query, query_type)

    for year in years:
        where         = build_filter(company=company, year=year)
        year_results  = []

        for sub_q in sub_queries:
            query_vec = embed_query(sub_q, model)
            try:
                raw  = query_collection(
                    collection, query_vec,
                    top_k=chunks_per_year,
                    where=where,
                )
                docs = results_to_documents(raw)
                for doc, score in docs:
                    cid = doc.metadata.get("chunk_id", "")
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        year_results.append((doc, score))
            except Exception:
                continue

        # Keep only the top chunks_per_year best results for this year
        year_results.sort(key=lambda x: x[1], reverse=True)
        all_results.extend(year_results[:chunks_per_year])

    return all_results


# ---------------------------------------------------------------------------
# Step 8 — LangChain-style retriever wrapper
# ---------------------------------------------------------------------------

class FinanceRetriever:
    """
    Retriever that supports both simple queries and filtered queries.

    get_relevant_documents() — standard top-K search across all documents
    get_filtered_documents() — search filtered by company and/or year
    get_trend_documents()    — multi-year retrieval for trend questions
    """

    def __init__(
        self,
        collection: chromadb.Collection,
        model: SentenceTransformer,
        top_k: int = TOP_K,
    ) -> None:
        self.collection = collection
        self.model      = model
        self.top_k      = top_k

    def get_relevant_documents(
        self,
        query: str,
        company: str | None = None,
        year: str | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Embed the query using type-aware sub-queries and return top_k results.

        Detects the query type (financial, governance, risk, strategy, legal)
        and selects sub-queries that match the section headings used in 10-K
        filings for that topic. This ensures the retriever fetches from the
        right part of the document regardless of what is being asked.

        Args:
            query:   User's plain-text question.
            company: Optional company name filter.
            year:    Optional year filter.

        Returns:
            List of (Document, similarity_score) tuples, best match first.
        """
        where      = build_filter(company=company, year=year)
        seen_ids   = set()
        results    = []
        query_type = detect_query_type(query)
        sub_queries = get_sub_queries(query, query_type)

        print(f"  Query type detected: {query_type}")

        for sub_q in sub_queries:
            query_vec = embed_query(sub_q, self.model)
            try:
                raw  = query_collection(
                    self.collection, query_vec,
                    top_k=self.top_k,
                    where=where,
                )
                docs = results_to_documents(raw)
                for doc, score in docs:
                    cid = doc.metadata.get("chunk_id", "")
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        results.append((doc, score))
            except Exception:
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:self.top_k]

    def get_comparison_documents(
        self,
        query: str,
        companies: list[str],
        year: str | None = None,
        chunks_per_company: int = 3,
    ) -> list[tuple[Document, float]]:
        """
        Retrieve chunks for each company separately and combine.

        For comparison questions like "Compare Apple and Microsoft revenue
        in 2022", this fetches chunks_per_company chunks from each company
        independently using metadata filters, then combines them.

        Each company gets equal representation in the context so the LLM
        has data for both sides of the comparison.

        Args:
            query:              User's question.
            companies:          List of company names e.g. ["APPLE", "MICROSOFT"].
            year:               Optional year to filter all companies by.
            chunks_per_company: Chunks to fetch per company (default 3).

        Returns:
            Combined list of (Document, score) grouped by company.
        """
        all_results = []
        seen_ids    = set()
        query_type  = detect_query_type(query)
        sub_queries = get_sub_queries(query, query_type)

        for company in companies:
            where        = build_filter(company=company, year=year)
            comp_results = []

            for sub_q in sub_queries:
                query_vec = embed_query(sub_q, self.model)
                try:
                    raw  = query_collection(
                        self.collection, query_vec,
                        top_k=chunks_per_company,
                        where=where,
                    )
                    docs = results_to_documents(raw)
                    for doc, score in docs:
                        cid = doc.metadata.get("chunk_id", "")
                        if cid not in seen_ids:
                            seen_ids.add(cid)
                            comp_results.append((doc, score))
                except Exception:
                    continue

            # Keep top chunks_per_company for this company
            comp_results.sort(key=lambda x: x[1], reverse=True)
            all_results.extend(comp_results[:chunks_per_company])

        return all_results

    def get_trend_documents(
        self,
        query: str,
        company: str,
        years: list[str],
        chunks_per_year: int = 2,
    ) -> list[tuple[Document, float]]:
        """
        Retrieve chunks covering multiple years for trend analysis.

        Calls retrieve_across_years() which queries each year separately
        so every year gets representation in the returned context.

        Args:
            query:           User's question.
            company:         Company name to filter by.
            years:           List of years to cover e.g. ["2018","2019",...].
            chunks_per_year: Chunks to fetch per year.

        Returns:
            Combined list of (Document, score) ordered by year.
        """
        return retrieve_across_years(
            self.collection, self.model,
            query, company, years, chunks_per_year,
        )


# ---------------------------------------------------------------------------
# Print helper
# ---------------------------------------------------------------------------

def print_retrieval_results(
    query: str,
    docs_and_scores: list[tuple[Document, float]],
) -> None:
    print("\n" + "=" * 60)
    print(f"QUERY: {query}")
    print(f"Top {len(docs_and_scores)} results")
    print("=" * 60)

    for rank, (doc, score) in enumerate(docs_and_scores, start=1):
        print(f"\n--- Result {rank}  |  Similarity score: {score:.4f} ---")
        print("\n  Metadata:")
        for key, value in doc.metadata.items():
            print(f"    {key:<14}: {value}")
        print(f"\n  Chunk text:")
        for line in doc.page_content.splitlines():
            print(f"    {line}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).parent
    db_path      = project_root / "vectordb"

    collection = load_chroma_collection(db_path)
    model      = load_embedding_model()
    retriever  = FinanceRetriever(collection=collection, model=model, top_k=TOP_K)

    test_query = "What was Amazon's net income from 2018 to 2022?"

    print(f"\nRunning trend retrieval for: '{test_query}'")
    docs_and_scores = retriever.get_trend_documents(
        query=test_query,
        company="AMAZON",
        years=["2018", "2019", "2020", "2021", "2022"],
        chunks_per_year=2,
    )
    print_retrieval_results(test_query, docs_and_scores)
