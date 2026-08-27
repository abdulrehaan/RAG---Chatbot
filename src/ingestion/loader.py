from pathlib import Path


def load_text_file(file_path: str) -> str:
    """
    Load a UTF-8 text file and return its contents.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    return path.read_text(encoding="utf-8")


if __name__ == "__main__":
    file_path = "data/documents/company_policy.txt"

    text = load_text_file(file_path)

    print("=" * 60)
    print("DOCUMENT LOADED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nCharacters: {len(text)}")
    print(f"Words: {len(text.split())}")

    print("\nDocument content:")
    print("-" * 60)
    print(text)