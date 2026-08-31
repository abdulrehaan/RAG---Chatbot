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
    Generates answers using the locally stored FLAN-T5 model.
    """

    def __init__(self, model_path: Path = MODEL_PATH):

        print("Loading generation model from:")
        print(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at: {model_path}"
            )

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        self.device = torch.device("cpu")

        # ----------------------------------------------------
        # Tokenizer
        # ----------------------------------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True
        )

        # ----------------------------------------------------
        # Model
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
    # GENERATE
    # ========================================================

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50
    ) -> str:
        """
        Generate a concise answer from the given prompt.
        """

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

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
        # Generate
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,

                # Keep answers short
                max_new_tokens=max_new_tokens,

                # Deterministic generation
                do_sample=False,

                # Small beam search
                num_beams=2,

                # Prevent unnecessary repetition
                repetition_penalty=1.1,

                # Stop when the model reaches EOS
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
Answer the question based only on the context.

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