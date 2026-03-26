"""
Ingest documents: read text and PDF files, convert to text, and chunk.
"""
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader

from . import chunk, config


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
        raise FileNotFoundError(f"[ERROR] Document directory does not exist: {doc_dir}")

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


def chunk_documents(
    docs: Dict[str, str],
    strategy: str = "token",
    max_chars: int = 800,
    overlap: int = 200,
    max_tokens: int = 500,
    overlap_tokens: int = 100,
    encoding_name: str = "cl100k_base",
) -> Dict[str, List[str]]:
    """
    Turn each full document text into a list of chunks.

    Strategies:
      - "simple": character-based chunking
      - "token" : token-aware chunking with paragraph-first packing
    """
    if strategy not in {"simple", "token"}:
        raise ValueError("[ERROR] Strategy must be one of: simple, token")

    out: Dict[str, List[str]] = {}
    for name, text in docs.items():
        if strategy == "simple":
            out[name] = chunk.simple_chunk(text, max_chars=max_chars, overlap=overlap)
        else:
            out[name] = chunk.token_chunks(
                text,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                encoding_name=encoding_name,
            )
    return out