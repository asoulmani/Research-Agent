"""
Chunking utilities: 
- Simple character-based strategy
- Token-aware strategy

Later:
- Semantic chunking
"""
from typing import List

import tiktoken


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


def token_chunks(
    text: str,
    max_tokens: int = 500,
    overlap_tokens: int = 100,
    encoding_name: str = "cl100k_base",
) -> List[str]:
    """
    Token-aware chunking with overlap.
    Even if it's not semantic chunking, it provides a predictable prompt size for the LLM.

    - Prefer paragraph boundaries (`\\n\\n`) and pack paragraphs up to `max_tokens`.
    - If a single paragraph exceeds `max_tokens`, split it into token windows.
    - Apply `overlap_tokens` between adjacent chunks for continuity.
    """
    if not text.strip():
        return []

    if overlap_tokens >= max_tokens:
        raise ValueError("[ERROR] overlap_tokens must be < max_tokens.")

    enc = tiktoken.get_encoding(encoding_name)

    def token_len(s: str) -> int:
        # Helper: get the number of tokens in a string.
        return len(enc.encode(s))

    def from_tokens(tokens: List[int]) -> str:
        # Helper: convert a list of tokens back to a string.
        return enc.decode(tokens).strip()

    # Split the text into paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    # First pass: paragraph-aware packing by token budget
    initial_chunks: List[str] = []
    current_parts: List[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = token_len(para)

        if para_tokens > max_tokens:
            # If a paragraph is longer than max_tokens, flush current chunk and split paragraph itself
            if current_parts:
                initial_chunks.append("\n\n".join(current_parts).strip())
                current_parts = []
                current_tokens = 0

            para_token_ids = enc.encode(para)
            # No overlap in first pass for oversized paragraphs.
            # Overlap is applied uniformly in the second pass.
            step = max_tokens
            for start in range(0, len(para_token_ids), step):
                window = para_token_ids[start : start + max_tokens]
                chunk = from_tokens(window)  # Go back to string format from tokens
                if chunk:
                    initial_chunks.append(chunk)
                if start + max_tokens >= len(para_token_ids):
                    break
            continue

        # Normal paragraph packing
        if current_parts and (current_tokens + para_tokens > max_tokens):
            initial_chunks.append("\n\n".join(current_parts).strip())
            current_parts = [para]
            current_tokens = para_tokens
        else:
            current_parts.append(para)
            current_tokens += para_tokens

    if current_parts:
        initial_chunks.append("\n\n".join(current_parts).strip())

    # Second pass: apply overlap between packed chunks for better continuity
    if overlap_tokens == 0 or len(initial_chunks) <= 1:
        return [c for c in initial_chunks if c]

    final_chunks: List[str] = []
    prev_tail: List[int] = []

    for i, chunk_text in enumerate(initial_chunks):
        chunk_tokens = enc.encode(chunk_text)
        if i == 0:
            final_chunks.append(chunk_text)
            prev_tail = chunk_tokens[-overlap_tokens:]
            continue

        merged_tokens = prev_tail + chunk_tokens
        if len(merged_tokens) > max_tokens:
            merged_tokens = merged_tokens[-max_tokens:]

        merged_text = from_tokens(merged_tokens)
        if merged_text:
            final_chunks.append(merged_text)

        prev_tail = chunk_tokens[-overlap_tokens:]

    return final_chunks