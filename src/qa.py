"""
QA: use LLM to answer questions based on the retrieved chunks.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from . import config, index


def _format_sources(hits: List[Dict[str, Any]], max_chars_per_source: int = 1600) -> str:
    """
    Convert retrieved chunks into a prompt-friendly "Sources" block with stable IDs.
    """
    parts: List[str] = []
    for i, hit in enumerate(hits, 1):
        meta = hit.get("metadata") or {}
        source_file = meta.get("source_file", "?")
        chunk_index = meta.get("chunk_index", "?")
        text = (hit.get("document") or "").strip()

        # Truncate to keep prompts smaller (V1: simple character limit).
        text = text[:max_chars_per_source]

        parts.append(
            f"[{i}] {source_file} (chunk {chunk_index}):\n{text}"
        )
    return "\n\n".join(parts)


def answer_question(
    question: str,
    top_k: int = 5,
    hits: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Retrieval-Augmented QA:
      1) retrieve top_k chunks from Chroma
      2) call an LLM with the retrieved chunks as sources
      3) return (answer_text, hits_used)
    """
    if hits is None:
        hits = index.query_index(question, n_results=top_k)
    if not hits:
        return "[INFO] I couldn't find relevant information in the documents.", []

    sources_block = _format_sources(hits)

    client = OpenAI()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant. Answer ONLY using the provided sources. "
                "If the sources do not contain the answer, say so. "
                "When you use information, cite it using the source IDs like [1], [2], etc."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Sources:\n{sources_block}\n\n"
                "Answer (include citations like [1] where relevant):"
            ),
        },
    ]

    resp = client.chat.completions.create(
        model=config.DEFAULT_CHAT_MODEL,
        messages=messages,
        temperature=0,
    )

    answer = resp.choices[0].message.content or ""
    return answer, hits

