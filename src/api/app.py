import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
# PROJECT PATH
# ============================================================

# Add src directory to Python path
SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORT RAG PIPELINE
# ============================================================

from rag.pipeline import RAGPipeline


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Company Policy RAG Chatbot",
    description="A Retrieval-Augmented Generation chatbot for company policies.",
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    """
    Request body for the /chat endpoint.
    """

    question: str


# ============================================================
# RESPONSE MODEL
# ============================================================

class ChatResponse(BaseModel):
    """
    Response returned by the /chat endpoint.
    """

    answer: str


# ============================================================
# RAG PIPELINE
# ============================================================

pipeline = None


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():
    """
    Initialize the RAG pipeline when the API starts.
    """

    global pipeline

    print("=" * 60)
    print("STARTING COMPANY POLICY RAG API")
    print("=" * 60)

    pipeline = RAGPipeline(
        top_k=3,
        score_threshold=0.65
    )

    print("\nRAG API is ready.")


# ============================================================
# SHUTDOWN
# ============================================================

@app.on_event("shutdown")
def shutdown_event():
    """
    Close the RAG pipeline when the API shuts down.
    """

    global pipeline

    if pipeline is not None:

        pipeline.close()

        pipeline = None

        print("\nRAG pipeline closed.")


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    """
    Basic API health message.
    """

    return {
        "message": "Company Policy RAG Chatbot API is running."
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    """
    Check whether the API and RAG pipeline are ready.
    """

    return {
        "status": "healthy",
        "rag_pipeline": pipeline is not None
    }


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Ask a question to the Company Policy RAG chatbot.
    """

    if pipeline is None:

        return ChatResponse(
            answer="The RAG pipeline is not ready."
        )

    question = request.question.strip()

    if not question:

        return ChatResponse(
            answer="Please enter a question."
        )

    try:

        answer = pipeline.answer(question)

        return ChatResponse(
            answer=answer
        )

    except Exception as error:

        print(
            f"Error while processing question: {error}"
        )

        return ChatResponse(
            answer=(
                "Sorry, I was unable to process "
                "your question."
            )
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )