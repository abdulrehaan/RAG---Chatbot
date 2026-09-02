import sys
from pathlib import Path


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
# CHATBOT
# ============================================================

class Chatbot:
    """
    Command-line interface for the Company Policy RAG chatbot.

    The chatbot uses the RAG pipeline to:
        1. Retrieve relevant company policy documents
        2. Filter irrelevant documents
        3. Build context
        4. Generate an answer using FLAN-T5
    """

    def __init__(self):

        print("=" * 60)
        print("INITIALIZING COMPANY POLICY CHATBOT")
        print("=" * 60)

        self.pipeline = RAGPipeline(
            top_k=3,
            score_threshold=0.65
        )

        print("\nChatbot ready.")

    # ========================================================
    # DISPLAY
    # ========================================================

    def display_welcome(self):
        """
        Display the chatbot welcome message.
        """

        print("\n" + "=" * 60)
        print("             COMPANY POLICY CHATBOT")
        print("=" * 60)

        print(
            "\nAsk questions about company policies."
        )

        print(
            "Type 'exit', 'quit', or 'q' to close the chatbot."
        )

        print("-" * 60)

    # ========================================================
    # PROCESS QUESTION
    # ========================================================

    def ask(self, question: str) -> str:
        """
        Send a question to the RAG pipeline.
        """

        if not question.strip():
            return "Please enter a question."

        try:

            answer = self.pipeline.answer(
                question
            )

            return answer

        except Exception as error:

            print(
                f"\nError while processing question: {error}"
            )

            return (
                "Sorry, I was unable to process "
                "your question."
            )

    # ========================================================
    # CHAT LOOP
    # ========================================================

    def run(self):
        """
        Start the interactive chatbot loop.
        """

        self.display_welcome()

        while True:

            try:

                question = input("\nYou: ").strip()

            except (KeyboardInterrupt, EOFError):

                print("\n\nGoodbye!")
                break

            # ------------------------------------------------
            # Exit commands
            # ------------------------------------------------

            if question.lower() in {
                "exit",
                "quit",
                "q"
            }:

                print("\nGoodbye!")
                break

            # ------------------------------------------------
            # Empty question
            # ------------------------------------------------

            if not question:

                print(
                    "Bot: Please enter a question."
                )

                continue

            # ------------------------------------------------
            # Generate answer
            # ------------------------------------------------

            print("\nBot: ", end="")

            answer = self.ask(question)

            print(answer)

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):
        """
        Close the underlying RAG pipeline.
        """

        if self.pipeline is not None:

            self.pipeline.close()


# ============================================================
# MAIN
# ============================================================

def main():

    chatbot = None

    try:

        chatbot = Chatbot()

        chatbot.run()

    finally:

        if chatbot is not None:

            chatbot.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()