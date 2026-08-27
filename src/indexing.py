from ingestion.loader import load_text_file
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from vectorstore.qdrant_store import QdrantStore


DOCUMENT_PATH = "data/documents/company_policy.txt"


def main():

    print("=" * 60)
    print("RAG DOCUMENT INDEXING")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load document
    # --------------------------------------------------

    print("\n[1/4] Loading document...")

    text = load_text_file(DOCUMENT_PATH)

    print(
        f"Document loaded successfully."
    )

    print(
        f"Characters: {len(text)}"
    )

    print(
        f"Words: {len(text.split())}"
    )

    # --------------------------------------------------
    # 2. Create chunks
    # --------------------------------------------------

    print("\n[2/4] Creating chunks...")

    chunks = chunk_text(
        text,
        chunk_size=50,
        overlap=10
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    for i, chunk in enumerate(chunks, start=1):
        print(
            f"  Chunk {i}: {len(chunk.split())} words"
        )

    # --------------------------------------------------
    # 3. Generate embeddings
    # --------------------------------------------------

    print("\n[3/4] Generating embeddings...")

    embedder = Embedder()

    embeddings = embedder.embed_documents(chunks)

    print(
        f"Generated {len(embeddings)} embeddings."
    )

    print(
        f"Embedding dimension: {len(embeddings[0])}"
    )

    # --------------------------------------------------
    # 4. Store in Qdrant
    # --------------------------------------------------

    print("\n[4/4] Storing vectors in Qdrant...")

    store = QdrantStore()

    store.add_documents(
        chunks,
        embeddings
    )

    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()