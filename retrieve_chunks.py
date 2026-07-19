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
# Step 7 — Multi-year retrieval for trend queries
# ---------------------------------------------------------------------------

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

    # Sub-queries targeting different section names that contain net income.
    # Financial statements use varied headings across companies:
    # "Consolidated Statements of Income", "Consolidated Statements of
    # Operations", "Selected Financial Data", "Quarterly Data".
    # Multiple sub-queries improve recall across all these variations.
    sub_queries = [
        query,
        "net income consolidated statements of income operations earnings",
        "net income attributable total annual earnings per share diluted",
        "income tax provision net earnings consolidated financial statements",
    ]

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
        Embed the query and return top_k (Document, score) pairs.

        Optionally filter by company and/or year using ChromaDB metadata
        filters. If neither is provided, searches across all documents.

        Args:
            query:   User's plain-text question.
            company: Optional company name filter e.g. "AMAZON".
            year:    Optional year filter e.g. "2022".

        Returns:
            List of (Document, similarity_score) tuples, best match first.
        """
        query_vec   = embed_query(query, self.model)
        where       = build_filter(company=company, year=year)
        raw_results = query_collection(
            self.collection, query_vec,
            top_k=self.top_k,
            where=where,
        )
        return results_to_documents(raw_results)

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
