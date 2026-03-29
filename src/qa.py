"""
QA: use LLM to answer questions based on the retrieved chunks.
"""
from __future__ import annotations

import json
import re
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
    policy: str = "strict",
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
        return "I couldn't find relevant information in the documents.", []

    sources_block = _format_sources(hits)

    client = OpenAI()
    abstain = "I couldn't find relevant information in the documents."
    policy = (policy or "strict").strip().lower()
    if policy not in {"strict", "definition"}:
        policy = "strict"

    is_yes_no = bool(
        re.match(
            r"^\s*(is|are|was|were|do|does|did|can|could|should|would|will|has|have|had)\b",
            question.strip().lower(),
        )
    )

    policy_block = (
        "Policy=STRICT: treat a yes/no as supported only if the sources explicitly justify that polarity. "
        "If not explicit, set supported=false.\n\n"
        if policy == "strict"
        else "Policy=DEFINITION: you may answer from quoted definitions/descriptions even if the source does not explicitly say yes/no. "
        "For yes/no questions, if polarity is not explicitly supported, do NOT guess; instead answer using the sources' definition/description and state that the sources do not explicitly say yes/no.\n\n"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant. Answer ONLY using the provided sources. "
                "Do NOT use any outside knowledge. "
                "A claim is supported only if it is explicitly stated in the sources or is a minimal paraphrase. "
                "You MUST identify verbatim supporting quotes from the sources, then answer ONLY from those quotes. "
                f"If you cannot find direct support, respond with supported=false and answer exactly: \"{abstain}\".\n\n"
                + policy_block
                + "Output MUST be valid JSON in this exact schema (no markdown fences, no extra text):\n"
                "{\n"
                '  "supported": true|false,\n'
                '  "answer": "string",\n'
                '  "citations": [1,2],\n'
                '  "supporting_spans": [{"source_id": 1, "quote": "verbatim quote"}]\n'
                "}\n\n"
                "Rules:\n"
                f"- If supported=false: answer MUST equal \"{abstain}\", citations MUST be [], supporting_spans MUST be [].\n"
                "- If supported=true: supporting_spans MUST contain at least one item.\n"
                "- Each quote must be copied verbatim from a source chunk.\n"
                "- citations must match the source_id values used in supporting_spans.\n"
                "- Keep answer to 1-2 concise sentences.\n"
                + (
                    "- Yes/No questions: apply the selected policy rules above.\n"
                    if is_yes_no
                    else ""
                )
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Sources:\n{sources_block}\n\n"
                "Return JSON only:"
            ),
        },
    ]

    for attempt in range(2):  # retry once to fix formatting/contract issues
        resp = client.chat.completions.create(
            model=config.DEFAULT_CHAT_MODEL,
            messages=messages,
            temperature=0,
        )

        raw = (resp.choices[0].message.content or "").strip()
        # Parse JSON even if the model prepends/appends whitespace.
        m = re.search(r"\{[\s\S]*\}\s*$", raw)
        json_text = m.group(0) if m else raw

        try:
            payload = json.loads(json_text)
        except Exception:
            payload = None

        if not isinstance(payload, dict):
            messages[0]["content"] = messages[0]["content"] + " STRICT REWRITE: Return only valid JSON (no extra text)."
            continue

        supported = bool(payload.get("supported"))
        answer = payload.get("answer")
        citations = payload.get("citations")
        spans = payload.get("supporting_spans")

        if not isinstance(answer, str):
            messages[0]["content"] = messages[0]["content"] + " STRICT REWRITE: 'answer' must be a string."
            continue
        if not isinstance(citations, list) or not all(isinstance(x, int) for x in citations):
            messages[0]["content"] = messages[0]["content"] + " STRICT REWRITE: 'citations' must be a list of integers."
            continue
        if not isinstance(spans, list):
            messages[0]["content"] = messages[0]["content"] + " STRICT REWRITE: 'supporting_spans' must be a list."
            continue

        if not supported:
            if answer.strip() != abstain or citations != [] or spans != []:
                messages[0]["content"] = (
                    messages[0]["content"]
                    + f" STRICT REWRITE: If supported=false, answer must be exactly \"{abstain}\" and citations/supporting_spans must be empty."
                )
                continue
            return abstain, []

        # supported == True
        if not spans:
            messages[0]["content"] = messages[0]["content"] + " STRICT REWRITE: supported=true requires supporting_spans."
            continue

        span_ids: List[int] = []
        ok_spans = True
        for sp in spans:
            if not isinstance(sp, dict):
                ok_spans = False
                break
            sid = sp.get("source_id")
            quote = sp.get("quote")
            if not isinstance(sid, int) or not isinstance(quote, str) or not quote.strip():
                ok_spans = False
                break
            if sid < 1 or sid > len(hits):
                ok_spans = False
                break
            span_ids.append(sid)
        if not ok_spans:
            messages[0]["content"] = messages[0]["content"] + " STRICT REWRITE: Each supporting span must be {source_id:int, quote:str} with valid source_id."
            continue

        if sorted(set(citations)) != sorted(set(span_ids)):
            messages[0]["content"] = messages[0]["content"] + " STRICT REWRITE: citations must match supporting_spans source_id values."
            continue

        # Heuristic: keep the display format you're already using (markdown + [1] citations).
        cites_md = "".join(f"[{i}]" for i in sorted(set(citations)))
        final_answer = f"{answer.strip()} {cites_md}".strip()
        return final_answer, hits

    return abstain, []
