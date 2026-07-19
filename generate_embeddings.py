"""
Finance RAG - Phase 5: Generate Embeddings

Loads chunked Documents from Phase 4, generates a vector embedding for
every chunk using sentence-transformers/all-MiniLM-L6-v2, and stores the
result as a list of dicts — each containing the original Document plus its
embedding vector — saved to disk with pickle.

Why sentence-transformers/all-MiniLM-L6-v2?
--------------------------------------------
- Lightweight: 22M parameters, runs fast on CPU
- Output: 384-dimensional dense vector
- Trained on 1B+ sentence pairs — strong semantic similarity
- Standard choice for RAG prototypes before scaling to larger models

What is an embedding?
---------------------
An embedding is a list of 384 floats that represents the *meaning* of a
chunk of text. Two chunks about revenue will have vectors close together
in 384-dimensional space. Two chunks about different topics will be far
apart. This is what lets the retriever find relevant chunks without
exact keyword matching.

Input  : chunks/<CompanyName>/<pdf_name>_chunks.pkl
Output : embeddings/<CompanyName>/<pdf_name>_embeddings.pkl

Each entry in the output file:
    {
        "document": Document(page_content=..., metadata=...),
        "embedding": [0.23, -0.81, 0.44, ...]   # 384 floats
    }
"""

import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Batch size: how many chunks to embed in one forward pass.
# Larger = faster but uses more RAM. 64 is safe for most machines.
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Step 1 — Load chunks from pickle
# ---------------------------------------------------------------------------

def load_chunks(pkl_path: Path) -> list[Document]:
    """
    Load the list of chunk Documents saved by Phase 4.

    Each Document has:
      - page_content : the text of the chunk
      - metadata     : company_name, report_year, section, chunk_id, etc.

    Args:
        pkl_path: Path to the .pkl file from Phase 4.

    Returns:
        List of LangChain Document objects.

    Raises:
        FileNotFoundError: If the pickle file does not exist.
    """
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {pkl_path}\n"
            "Run chunk_documents.py first."
        )
    with open(pkl_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded {len(chunks)} chunks from {pkl_path}")
    return chunks


# ---------------------------------------------------------------------------
# Step 2 — Load the embedding model
# ---------------------------------------------------------------------------

def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """
    Download (first run) and load the Sentence Transformer model.

    On the first call this downloads ~90MB of model weights from
    HuggingFace and caches them locally. Subsequent runs load from cache.

    The model maps any text string → a fixed-size vector of 384 floats.
    The dimension is fixed by the model architecture — all-MiniLM-L6-v2
    always outputs 384 dimensions regardless of input length.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Loaded SentenceTransformer model ready for inference.
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


# ---------------------------------------------------------------------------
# Step 3 — Generate embeddings
# ---------------------------------------------------------------------------

def generate_embeddings(
    chunks: list[Document],
    model: SentenceTransformer,
    batch_size: int = BATCH_SIZE,
) -> list[list[float]]:
    """
    Generate one embedding vector per chunk.

    Extracts the page_content string from each Document and passes all
    texts to the model in batches. Batching is more efficient than
    embedding one chunk at a time because the model processes multiple
    texts in parallel.

    The model encodes text → 384-dimensional float vector.
    Normalisation is enabled (normalize_embeddings=True) so all vectors
    have unit length. This makes cosine similarity equivalent to dot
    product and slightly improves retrieval quality.

    Args:
        chunks:     List of chunk Documents whose page_content to embed.
        model:      Loaded SentenceTransformer model.
        batch_size: Number of chunks per forward pass.

    Returns:
        List of embeddings. embeddings[i] corresponds to chunks[i].
        Each embedding is a list of 384 floats.
    """
    texts = [chunk.page_content for chunk in chunks]

    print(f"\nGenerating embeddings for {len(texts)} chunks "
          f"(batch_size={batch_size})...")

    # encode() returns a numpy array of shape (num_chunks, 384)
    # convert_to_python=False keeps it as numpy for efficiency
    embeddings_np = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,   # unit-length vectors
        show_progress_bar=True,
    )

    # Convert to plain Python lists so pickle works without numpy dependency
    embeddings = [vec.tolist() for vec in embeddings_np]

    print(f"Done. Generated {len(embeddings)} embeddings.")
    print(f"Embedding dimension: {len(embeddings[0])}")

    return embeddings


# ---------------------------------------------------------------------------
# Step 4 — Pair documents with their embeddings
# ---------------------------------------------------------------------------

def pair_chunks_with_embeddings(
    chunks: list[Document],
    embeddings: list[list[float]],
) -> list[dict]:
    """
    Combine each chunk Document with its embedding into a single dict.

    Keeping them paired ensures the embedding and the Document (with its
    metadata) are never accidentally separated. Phase 6 (ChromaDB) will
    unpack these pairs to insert text, embedding, and metadata together
    into the vector store.

    Structure of each entry:
        {
            "document" : Document(page_content=..., metadata={...}),
            "embedding": [float, float, ...]   # 384 floats
        }

    Args:
        chunks:     List of chunk Documents.
        embeddings: List of embedding vectors in the same order.

    Returns:
        List of dicts, one per chunk.
    """
    assert len(chunks) == len(embeddings), (
        f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
    )

    paired = [
        {"document": chunk, "embedding": embedding}
        for chunk, embedding in zip(chunks, embeddings)
    ]
    return paired


# ---------------------------------------------------------------------------
# Step 5 — Save to disk
# ---------------------------------------------------------------------------

def save_embeddings(paired: list[dict], output_path: Path) -> None:
    """
    Persist the paired (Document + embedding) list to disk using pickle.

    Pickle preserves the exact Python objects — no information is lost.
    Phase 6 loads this file and inserts everything into ChromaDB.

    Args:
        paired:      List of {document, embedding} dicts.
        output_path: Destination .pkl file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(paired, f)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved to: {output_path}  ({size_mb:.2f} MB)")


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_embedding_summary(
    chunks: list[Document],
    embeddings: list[list[float]],
    sample_index: int = 0,
    vector_preview: int = 8,
) -> None:
    """
    Print a summary table and one sample embedding for verification.

    Shows:
      - Total number of embeddings generated
      - Embedding dimension (should always be 384 for this model)
      - Sample chunk metadata
      - First N floats of the sample embedding vector

    Args:
        chunks:        List of chunk Documents.
        embeddings:    List of embedding vectors.
        sample_index:  Which chunk/embedding to show as sample.
        vector_preview: How many floats of the vector to display.
    """
    sample_doc = chunks[sample_index]
    sample_vec = embeddings[sample_index]

    print("\n" + "=" * 60)
    print("EMBEDDING SUMMARY")
    print("=" * 60)
    print(f"  Total embeddings : {len(embeddings)}")
    print(f"  Embedding dim    : {len(sample_vec)}")
    print(f"  Model            : {EMBEDDING_MODEL_NAME}")

    print(f"\n--- Sample chunk (index {sample_index}) metadata ---")
    for key, value in sample_doc.metadata.items():
        print(f"  {key:<15}: {value}")

    print(f"\n--- Sample chunk page_content (first 200 chars) ---")
    print(sample_doc.page_content[:200], "...")

    print(f"\n--- Sample embedding (first {vector_preview} of "
          f"{len(sample_vec)} floats) ---")
    preview = [round(v, 6) for v in sample_vec[:vector_preview]]
    print(f"  {preview}  ...")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).parent
    chunks_dir   = project_root / "chunks"
    embeddings_dir = project_root / "embeddings"

    # Discover all chunk pickle files produced by Phase 4
    chunk_files = sorted(chunks_dir.rglob("*_chunks.pkl"))

    if not chunk_files:
        print(f"No chunk files found under {chunks_dir}")
        print("Run chunk_documents.py first.")
        raise SystemExit(1)

    print(f"Found {len(chunk_files)} chunk file(s).\n")

    # Load the embedding model once — reuse across all files
    model = load_embedding_model()

    all_paired: list[dict] = []

    print("=" * 60)
    for chunk_file in chunk_files:
        # Derive output path: chunks/3M/3M_2015_10K_chunks.pkl
        #                  → embeddings/3M/3M_2015_10K_embeddings.pkl
        relative   = chunk_file.relative_to(chunks_dir)
        output_name = relative.stem.replace("_chunks", "_embeddings") + ".pkl"
        output_path = embeddings_dir / relative.parent / output_name

        # Step 1: load chunks
        chunks = load_chunks(chunk_file)

        # Step 2: generate embeddings
        embeddings = generate_embeddings(chunks, model)

        # Step 3: pair documents with embeddings
        paired = pair_chunks_with_embeddings(chunks, embeddings)

        # Step 4: save
        save_embeddings(paired, output_path)

        all_paired.extend(paired)
        print("=" * 60)

    # Print summary using the first paired entry
    all_chunks     = [p["document"]  for p in all_paired]
    all_embeddings = [p["embedding"] for p in all_paired]
    print_embedding_summary(all_chunks, all_embeddings, sample_index=5)
