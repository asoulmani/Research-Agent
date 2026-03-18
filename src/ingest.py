from pathlib import Path
from typing import Dict, List

from . import config
from pypdf import PdfReader


def read_txt_file(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n".join(parts)


def load_documents(doc_dir: Path | None = None) -> Dict[str, str]:
    """
    Load all supported documents from a directory into memory.

    Returns a mapping: filename -> full text content.
    """
    if doc_dir is None:
        doc_dir = config.DOCS_DIR

    if not doc_dir.exists():
        raise FileNotFoundError(f"Document directory does not exist: {doc_dir}")

    docs: Dict[str, str] = {}

    for path in sorted(doc_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue

        suffix = path.suffix.lower()

        try:
            if suffix == ".txt":
                text = read_txt_file(path)
            elif suffix == ".pdf":
                text = read_pdf_file(path)
            else:
                # Skip unsupported file types for now
                continue
        except Exception as e:
            print(f"[ERROR] Failed to read {path.name}: {e}")
            continue

        docs[path.name] = text

    return docs


def simple_chunk(text: str, max_chars: int = 800, overlap: int = 200) -> List[str]:
    """
    Very simple character-based chunking.
    Not token-aware yet, but good enough for V1.
    """
    chunks: List[str] = []
    start = 0
    n = len(text)

    if n == 0:
        return chunks

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0

        if end == n:
            break

    return chunks


def chunk_documents(
    docs: Dict[str, str],
    max_chars: int = 800,
    overlap: int = 200,
) -> Dict[str, List[str]]:
    """
    Turn each full document text into a list of chunks.
    """
    return {
        name: simple_chunk(text, max_chars=max_chars, overlap=overlap)
        for name, text in docs.items()
    }

    