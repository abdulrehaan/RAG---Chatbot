import sys
from pathlib import Path
from typing import List, Dict

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.embedder import Embedder
from src.vectorstore.qdrant_store import QdrantStore


class Retriever:
    """
    Converts a user query into an embedding
    and retrieves the most relevant document
    chunks from Qdrant.
    """

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

        print("Initializing retriever...")

        self.embedder = Embedder()
        self.vector_store = QdrantStore()

        print("Retriever initialized successfully.")

    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve the most relevant document chunks
        for a given user query.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        # Convert query into a 384-dimensional embedding
        query_embedding = self.embedder.embed_text(query)

        # Search Qdrant
        results = self.vector_store.search(
            query_embedding=query_embedding,
            limit=self.top_k
        )

        retrieved_chunks = []

        for result in results:
            retrieved_chunks.append(
                {
                    "score": result.score,
                    "text": result.payload["text"],
                    "source": result.payload["source"],
                    "chunk_id": result.payload["chunk_id"],
                }
            )

        return retrieved_chunks


if __name__ == "__main__":

    print("=" * 60)
    print("RAG RETRIEVAL TEST")
    print("=" * 60)

    retriever = Retriever(top_k=3)

    query = "What happens if I arrive late?"

    print(f"\nQuery:")
    print(query)

    results = retriever.retrieve(query)

    print("\nRetrieved documents:")
    print("-" * 60)

    for i, result in enumerate(results, start=1):

        print(f"\nRESULT {i}")
        print(f"Score: {result['score']:.4f}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Source: {result['source']}")

        print("\nText:")
        print(result["text"])