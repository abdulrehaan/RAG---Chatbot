from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--google--flan-t5-small"
    / "snapshots"
    / "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
)


# ============================================================
# GENERATOR
# ============================================================

class Generator:
    """
    Generates answers using FLAN-T5.

    The model is instructed to answer only from
    the supplied context.
    """

    def __init__(self, model_path: Path = MODEL_PATH):

        print("Loading generation model from:")
        print(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at: {model_path}"
            )

        # CPU because CUDA is not available
        self.device = torch.device("cpu")

        # ----------------------------------------------------
        # Load tokenizer
        # ----------------------------------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True
        )

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            str(model_path),
            local_files_only=True
        )

        self.model.to(self.device)

        # Evaluation mode
        self.model.eval()

        print("Generation model loaded successfully.")
        print(f"Device: {self.device}")

    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 80
    ) -> str:
        """
        Generate an answer from the supplied prompt.
        """

        # ----------------------------------------------------
        # Tokenize prompt
        # ----------------------------------------------------

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,

                # Limit answer length
                max_new_tokens=max_new_tokens,

                # Deterministic generation
                do_sample=False,

                # Beam search
                num_beams=4,

                # Prevent repetitive answers
                no_repeat_ngram_size=3,

                # Stop when appropriate
                early_stopping=True
            )

        # ----------------------------------------------------
        # Decode
        # ----------------------------------------------------

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return answer.strip()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TEXT GENERATION TEST")
    print("=" * 60)

    generator = Generator()

    prompt = """
Answer the question using only the information provided
in the context.

Do not use outside knowledge.
Do not mention information that is unrelated to the question.
Give a short and direct answer.

Context:
Employees are expected to maintain regular attendance.
Employees arriving more than 15 minutes late may be marked as late.
Three late arrivals in a month may require a discussion
with the reporting manager.

Question:
What happens if an employee arrives more than 15 minutes late?

Answer:
"""

    print("\nPrompt:")
    print(prompt)

    answer = generator.generate(prompt)

    print("\nGenerated Answer:")
    print("-" * 60)
    print(answer)