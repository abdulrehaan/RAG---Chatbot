import re
from typing import List


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences while preserving the sentence content.
    """

    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 50,
    overlap: int = 10
) -> List[str]:
    """
    Create sentence-aware overlapping chunks.

    chunk_size:
        Approximate maximum number of words per chunk.

    overlap:
        Approximate number of words to carry into the next chunk.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    sentences = split_into_sentences(text)

    chunks = []
    current_sentences = []
    current_word_count = 0

    for sentence in sentences:

        sentence_word_count = len(sentence.split())

        if (
            current_sentences
            and current_word_count + sentence_word_count > chunk_size
        ):
            chunks.append(" ".join(current_sentences))

            # Keep the last few sentences as overlap
            overlap_sentences = []
            overlap_words = 0

            for previous_sentence in reversed(current_sentences):

                words = len(previous_sentence.split())

                if overlap_words + words > overlap:
                    break

                overlap_sentences.insert(0, previous_sentence)
                overlap_words += words

            current_sentences = overlap_sentences
            current_word_count = overlap_words

        current_sentences.append(sentence)
        current_word_count += sentence_word_count

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


if __name__ == "__main__":
    from loader import load_text_file

    file_path = "data/documents/company_policy.txt"

    text = load_text_file(file_path)

    chunks = chunk_text(
        text,
        chunk_size=50,
        overlap=10
    )

    print("=" * 60)
    print("SENTENCE-AWARE DOCUMENT CHUNKING")
    print("=" * 60)

    print(f"\nTotal chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n{'-' * 60}")
        print(f"CHUNK {i}")
        print(f"Words: {len(chunk.split())}")
        print("-" * 60)
        print(chunk)