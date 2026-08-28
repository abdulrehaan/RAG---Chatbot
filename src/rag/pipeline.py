from pathlib import Path
import sys


# ============================================================
# PATH CONFIGURATION
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from embeddings.embedder import Embedder
from generation.generator import Generator
from vectorstore.qdrant_store import QdrantStore


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_TOP_K = 3
SIMILARITY_THRESHOLD = 0.65


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
    Similarity Filtering
        ↓
    Context Construction
        ↓
    Prompt Creation
        ↓
    FLAN-T5 Generation
        ↓
    Final Answer
    """

    def __init__(
        self,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD
    ):

        print("=" * 60)
        print("INITIALIZING RAG PIPELINE")
        print("=" * 60)

        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        # ----------------------------------------------------
        # 1. Embedding Model
        # ----------------------------------------------------

        print("\n[1/3] Loading embedding model...")

        self.embedder = Embedder()

        # ----------------------------------------------------
        # 2. Qdrant
        # ----------------------------------------------------

        print("\n[2/3] Connecting to Qdrant...")

        self.vector_store = QdrantStore()

        # ----------------------------------------------------
        # 3. Generation Model
        # ----------------------------------------------------

        print("\n[3/3] Loading generation model...")

        self.generator = Generator()

        print("\nRAG pipeline initialized successfully.")

    # ========================================================
    # RETRIEVAL
    # ========================================================

    def retrieve(self, query: str):

        """
        Retrieve the most relevant document chunks.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        query_embedding = self.embedder.embed_text(query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            limit=self.top_k
        )

        return results

    # ========================================================
    # FILTER RESULTS
    # ========================================================

    def filter_results(self, results):

        """
        Remove weakly relevant results using
        cosine similarity threshold.
        """

        filtered_results = []

        for result in results:

            if result.score >= self.similarity_threshold:
                filtered_results.append(result)

        return filtered_results

    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    def build_context(self, results) -> str:

        """
        Convert retrieved results into clean context.
        """

        context_parts = []

        for result in results:

            text = result.payload.get("text", "")

            if text:
                context_parts.append(text.strip())

        return "\n\n".join(context_parts)

    # ========================================================
    # BUILD PROMPT
    # ========================================================

    def build_prompt(
        self,
        query: str,
        context: str
    ) -> str:

        """
        Create a focused prompt for FLAN-T5.
        """

        prompt = f"""
Answer the question using only the information in the context.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. Give a complete and direct answer.
4. Include all relevant details from the context.
5. If the context does not contain the answer, say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

        return prompt.strip()

    # ========================================================
    # ANSWER
    # ========================================================

    def answer(self, query: str) -> str:

        """
        Complete RAG process.
        """

        # ----------------------------------------------------
        # Step 1: Retrieve
        # ----------------------------------------------------

        results = self.retrieve(query)

        if not results:
            return "I could not find the answer in the provided documents."

        # ----------------------------------------------------
        # Step 2: Filter
        # ----------------------------------------------------

        filtered_results = self.filter_results(results)

        if not filtered_results:
            return "I could not find the answer in the provided documents."

        # ----------------------------------------------------
        # Step 3: Build Context
        # ----------------------------------------------------

        context = self.build_context(filtered_results)

        if not context:
            return "I could not find the answer in the provided documents."

        # ----------------------------------------------------
        # Step 4: Build Prompt
        # ----------------------------------------------------

        prompt = self.build_prompt(
            query=query,
            context=context
        )

        # ----------------------------------------------------
        # Step 5: Generate Answer
        # ----------------------------------------------------

        answer = self.generator.generate(
            prompt,
            max_new_tokens=80
        )

        if not answer.strip():
            return "I could not find the answer in the provided documents."

        return answer.strip()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    pipeline = RAGPipeline(
        top_k=3,
        similarity_threshold=0.65
    )

    print("\n" + "=" * 60)
    print("RAG PIPELINE TEST")
    print("=" * 60)

    query = "What happens if I arrive more than 15 minutes late?"

    print("\nQuestion:")
    print(query)

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    print("\nRetrieving relevant documents...")

    results = pipeline.retrieve(query)

    print(f"\nRetrieved {len(results)} documents.")

    for index, result in enumerate(results, start=1):

        print(f"\n{'-' * 60}")
        print(f"RESULT {index}")

        print(f"Score: {result.score:.4f}")

        text = result.payload.get("text", "")

        print("\nText:")
        print(text)

    # --------------------------------------------------------
    # FILTERING
    # --------------------------------------------------------

    filtered_results = pipeline.filter_results(results)

    print("\n" + "=" * 60)
    print("FILTERED RESULTS")
    print("=" * 60)

    print(
        f"\nUsing {len(filtered_results)} "
        f"of {len(results)} retrieved documents."
    )

    for index, result in enumerate(filtered_results, start=1):

        print(
            f"\nResult {index}: "
            f"score={result.score:.4f}"
        )

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("GENERATING ANSWER")
    print("=" * 60)

    answer = pipeline.answer(query)

    print("\nAnswer:")
    print("-" * 60)
    print(answer)