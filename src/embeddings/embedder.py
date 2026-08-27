from sentence_transformers import SentenceTransformer
from typing import List


MODEL_NAME = "BAAI/bge-small-en-v1.5"


class Embedder:
    """
    Converts text into numerical embeddings.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        print("Embedding model loaded successfully.")

    def embed_text(self, text: str) -> List[float]:
        """
        Convert a single piece of text into an embedding.
        """

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple texts into embeddings.
        """

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return embeddings.tolist()


if __name__ == "__main__":

    embedder = Embedder()

    text = "Employees may carry forward a maximum of 10 unused leaves."

    embedding = embedder.embed_text(text)

    print("=" * 60)
    print("EMBEDDING TEST")
    print("=" * 60)

    print(f"\nText:")
    print(text)

    print(f"\nEmbedding dimensions: {len(embedding)}")

    print("\nFirst 10 values:")
    print(embedding[:10])