from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid


COLLECTION_NAME = "company_policies"
VECTOR_SIZE = 384
QDRANT_PATH = "data/qdrant"


class QdrantStore:
    """
    Handles storing and searching document embeddings in Qdrant.
    """

    def __init__(self):
        # Persistent local Qdrant database
        self.client = QdrantClient(path=QDRANT_PATH)

        self._create_collection()

    def _create_collection(self):
        """
        Create the collection if it doesn't already exist.
        """

        collections = self.client.get_collections().collections

        collection_names = [
            collection.name
            for collection in collections
        ]

        if COLLECTION_NAME not in collection_names:

            self.client.create_collection(
                collection_name=COLLECTION_NAME,

                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )

            print(
                f"Created collection: {COLLECTION_NAME}"
            )

        else:
            print(
                f"Collection already exists: {COLLECTION_NAME}"
            )

    def add_documents(self, chunks, embeddings):
        """
        Store document chunks and their embeddings in Qdrant.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must be equal."
            )

        points = []

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):

            point = PointStruct(
                id=str(uuid.uuid4()),

                vector=embedding,

                payload={
                    "chunk_id": index,
                    "text": chunk,
                    "source": "company_policy.txt"
                }
            )

            points.append(point)

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        print(
            f"Stored {len(points)} documents in Qdrant."
        )

    def search(self, query_embedding, limit=3):
        """
        Search Qdrant for the most similar document chunks.
        """

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=limit
        )

        return results.points


if __name__ == "__main__":

    print("=" * 60)
    print("QDRANT VECTOR STORE TEST")
    print("=" * 60)

    store = QdrantStore()

    print("\nQdrant is working correctly.")

    print(
        "\nCollections:"
    )

    print(
        store.client.get_collections()
    )