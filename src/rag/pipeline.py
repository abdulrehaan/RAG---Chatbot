from pathlib import Path
import sys


# ============================================================
# PROJECT PATH
# ============================================================

# Add src directory to Python path
SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORTS
# ============================================================

from embeddings.embedder import Embedder
from generation.generator import Generator
from vectorstore.qdrant_store import QdrantStore


# ============================================================
# RAG PIPELINE
# ============================================================

class RAGPipeline:
    """
    Complete Retrieval-Augmented Generation pipeline.

    Flow:

        User Query
            ↓
        Query Embedding
            ↓
        Qdrant Similarity Search
            ↓
        Relevance Filtering
            ↓
        Context Construction
            ↓
        FLAN-T5 Generation
            ↓
        Final Answer
    """

    def __init__(
        self,
        top_k: int = 3,
        score_threshold: float = 0.65
    ):
        """
        Initialize the RAG pipeline.

        Args:
            top_k:
                Maximum number of documents to retrieve.

            score_threshold:
                Minimum similarity score required for
                a document to be used for generation.
        """

        print("=" * 60)
        print("INITIALIZING RAG PIPELINE")
        print("=" * 60)

        self.top_k = top_k
        self.score_threshold = score_threshold

        # --------------------------------------------------------
        # 1. Embedding model
        # --------------------------------------------------------

        print("\n[1/3] Loading embedding model...")

        self.embedder = Embedder()

        # --------------------------------------------------------
        # 2. Qdrant
        # --------------------------------------------------------

        print("\n[2/3] Connecting to Qdrant...")

        self.vector_store = QdrantStore()

        # --------------------------------------------------------
        # 3. Generation model
        # --------------------------------------------------------

        print("\n[3/3] Loading generation model...")

        self.generator = Generator()

        print("\nRAG pipeline initialized successfully.")

    # ========================================================
    # RETRIEVAL
    # ========================================================

    def retrieve(self, query: str):
        """
        Retrieve relevant documents from Qdrant.

        Args:
            query:
                User's question.

        Returns:
            List of Qdrant search results.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        # Convert query into embedding
        query_embedding = self.embedder.embed_text(query)

        # Search vector database
        results = self.vector_store.search(
            query_embedding=query_embedding,
            limit=self.top_k
        )

        return results

    # ========================================================
    # FILTERING
    # ========================================================

    def filter_results(self, results):
        """
        Remove documents below the similarity threshold.

        This prevents unrelated documents from being passed
        to the generation model.
        """

        filtered_results = []

        for result in results:

            if result.score >= self.score_threshold:
                filtered_results.append(result)

        return filtered_results

    # ========================================================
    # CONTEXT
    # ========================================================

    def build_context(self, results) -> str:
        """
        Combine retrieved document chunks into context.
        """

        context_parts = []

        for result in results:

            text = result.payload.get("text", "").strip()

            if text:
                context_parts.append(text)

        return "\n\n".join(context_parts)

    # ========================================================
    # PROMPT
    # ========================================================

    def build_prompt(
        self,
        context: str,
        query: str
    ) -> str:
        """
        Build a grounded prompt for FLAN-T5.
        """

        return f"""
Answer the question using ONLY the information
provided in the context.

Do not use outside knowledge.

If the answer is not explicitly stated in the
context, say:

"I could not find the answer in the provided documents."

Keep the answer short, clear, and factual.

Context:
{context}

Question:
{query}

Answer:
""".strip()

    # ========================================================
    # ANSWER
    # ========================================================

    def answer(self, query: str) -> str:
        """
        Complete RAG process:

        Query
        → Retrieval
        → Filtering
        → Context
        → Generation
        → Answer
        """

        if not query or not query.strip():
            return "Please enter a valid question."

        # ----------------------------------------------------
        # Step 1: Retrieve
        # ----------------------------------------------------

        results = self.retrieve(query)

        if not results:
            return "I could not find relevant information in the documents."

        # ----------------------------------------------------
        # Step 2: Filter
        # ----------------------------------------------------

        filtered_results = self.filter_results(results)

        print("\n" + "=" * 60)
        print("FILTERED RESULTS")
        print("=" * 60)

        print(
            f"\nUsing {len(filtered_results)} "
            f"of {len(results)} retrieved documents."
        )

        for index, result in enumerate(
            filtered_results,
            start=1
        ):

            print(
                f"\nResult {index}: "
                f"score={result.score:.4f}"
            )

        # ----------------------------------------------------
        # Step 3: Build context
        # ----------------------------------------------------

        context = self.build_context(filtered_results)

        if not context:

            return (
                "I could not find the answer "
                "in the provided documents."
            )

        # ----------------------------------------------------
        # Step 4: Build prompt
        # ----------------------------------------------------

        prompt = self.build_prompt(
            context=context,
            query=query
        )

        # ----------------------------------------------------
        # Step 5: Generate answer
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("GENERATING ANSWER")
        print("=" * 60)

        answer = self.generator.generate(prompt)

        # ----------------------------------------------------
        # Safety fallback
        # ----------------------------------------------------

        if not answer.strip():

            return (
                "I could not find the answer "
                "in the provided documents."
            )

        return answer.strip()

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):
        """
        Close the Qdrant client cleanly.
        """

        try:
            if hasattr(self.vector_store, "client"):
                self.vector_store.client.close()

            print("\nQdrant client closed.")

        except Exception as error:

            print(
                f"\nWarning while closing Qdrant: {error}"
            )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    pipeline = None

    try:

        pipeline = RAGPipeline(
            top_k=3,
            score_threshold=0.65
        )

        print("\n" + "=" * 60)
        print("RAG PIPELINE TEST")
        print("=" * 60)

        query = (
            "What happens if I arrive "
            "more than 15 minutes late?"
        )

        print("\nQuestion:")
        print(query)

        print("\nRetrieving relevant documents...")

        # ----------------------------------------------------
        # Retrieve documents
        # ----------------------------------------------------

        results = pipeline.retrieve(query)

        print(
            f"\nRetrieved {len(results)} documents."
        )

        # ----------------------------------------------------
        # Display retrieved documents
        # ----------------------------------------------------

        for index, result in enumerate(
            results,
            start=1
        ):

            print("\n" + "-" * 60)

            print(f"RESULT {index}")

            print(
                f"Score: {result.score:.4f}"
            )

            text = result.payload.get(
                "text",
                ""
            )

            print("\nText:")
            print(text)

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        answer = pipeline.answer(query)

        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)

        print("\n" + answer)

    finally:

        if pipeline is not None:
            pipeline.close()