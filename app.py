"""
Streamlit UI for ResearchAgent.

Run:
  streamlit run app.py
"""
from __future__ import annotations

from collections import Counter
import html
import re
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

from src import config, index, ingest, qa

# Load environment variables
load_dotenv()


def _count_docs_in_folder(doc_dir: Path) -> int:
    if not doc_dir.exists():
        return 0
    return len([p for p in doc_dir.iterdir() if p.is_file() and not p.name.startswith(".")])


def _build_or_rebuild_index() -> int:
    docs = ingest.load_documents()
    chunked = ingest.chunk_documents(docs, strategy="token")
    return index.index_documents(chunked)


def _source_badge(hit: Dict[str, Any], i: int) -> str:
    meta = hit.get("metadata") or {}
    src = meta.get("source_file", "?")
    chunk_index = meta.get("chunk_index", "?")
    dist = hit.get("distance")
    if dist is None:
        return f"[{i}] {src} (chunk {chunk_index})"
    return f"[{i}] {src} (chunk {chunk_index}) · distance={dist:.3f}"


def _extract_answer_phrases(answer: str, max_phrases: int = 10) -> List[str]:
    """
    Extract a small set of keywords from the answer to highlight in retrieved sources.
    We use keywords (not long phrases) so they actually match noisy PDF text.
    """
    text = re.sub(r"\[\d+\]", "", answer)  # remove citations like [1]
    text = re.sub(r"[^A-Za-z0-9\s\-\_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()

    stopwords = {
        "the",
        "and",
        "or",
        "to",
        "of",
        "in",
        "a",
        "an",
        "for",
        "on",
        "with",
        "as",
        "is",
        "are",
        "it",
        "that",
        "this",
        "at",
        "by",
        "from",
        "we",
        "you",
        "your",
        "be",
        "into",
        "than",
        "such",
        "not",
        "only",
        "when",
        "where",
        "what",
        "how",
        "why",
        "can",
        "could",
        "should",
        "would",
    }

    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-\_]{3,}", text)
    words = [w for w in words if w not in stopwords]
    counts = Counter(words)
    # Keep keywords stable + deterministic.
    keywords = [w for w, _ in counts.most_common(max_phrases)]
    return keywords


def _highlight_text(source_text: str, phrases: List[str]) -> str:
    """
    Return HTML-renderable text with answer-keywords highlighted.
    """
    safe_text = html.escape(source_text)
    for phrase in phrases:
        # Highlight whole-word matches to avoid over-highlighting common substrings.
        pattern = re.compile(rf"(?<!\\w)({re.escape(phrase)})(?!\\w)", flags=re.IGNORECASE)
        safe_text = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", safe_text)
    # Preserve line breaks for readability inside source chunks.
    return safe_text.replace("\n", "<br>")


def _looks_like_question(text: str) -> bool:
    """
    Minimal heuristic to avoid running retrieval/QA for non-questions.
    """
    if not text:
        return False
    t = text.strip()
    if "?" in t:
        return True
    t = t.lower()
    starters = (
        "what ",
        "how ",
        "why ",
        "when ",
        "where ",
        "who ",
        "which ",
        "can ",
        "could ",
        "should ",
        "would ",
        "do ",
        "does ",
        "did ",
        "is ",
        "are ",
        "am ",
        "were ",
        "will ",
        "would ",
        "please ",
    )
    return t.startswith(starters)


def _render_history() -> None:
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])
            sources = msg.get("sources") or []
            answer_phrases = msg.get("answer_phrases") or []
            if role == "assistant" and sources:
                st.markdown("**Sources**")
                for i, hit in enumerate(sources, start=1):
                    with st.expander(_source_badge(hit, i)):
                        source_text = hit.get("document") or ""
                        highlighted = _highlight_text(source_text, answer_phrases)
                        st.markdown(highlighted, unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="ResearchAgent", page_icon="📚", layout="wide")

    st.title("ResearchAgent")
    st.caption("Grounded Q&A over local papers with citations.")

    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = []

    with st.sidebar:
        st.subheader("Index")
        st.write(f"Docs folder: `{config.DOCS_DIR}`")
        st.write(f"Detected documents: `{_count_docs_in_folder(config.DOCS_DIR)}`")
        st.write(f"Distance threshold: `{config.RETRIEVAL_DISTANCE_THRESHOLD}`")

        st.subheader("Answering")
        grounding_mode = st.selectbox(
            "Grounding mode",
            options=["Strict (explicit support only)", "Definition-based (allow minimal inference)"],
            index=0,
            help=(
                "Strict: abstains unless the sources explicitly support the answer.\n"
                "Definition-based: still requires verbatim quotes, but allows short answers derived from quoted definitions/descriptions."
            ),
        )
        policy = "strict" if grounding_mode.startswith("Strict") else "definition"

        prev_policy = st.session_state.get("policy")
        st.session_state["policy"] = policy

        rebuild_clicked = st.button("Rebuild index", use_container_width=True, type="primary")
        if rebuild_clicked:
            with st.spinner("Loading documents, chunking, and indexing..."):
                try:
                    indexed_chunks = _build_or_rebuild_index()
                    st.success(f"Index ready ({indexed_chunks} chunks).")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    policy_label = "Strict" if st.session_state.get("policy") == "strict" else "Definition-based"
    st.info(f"Grounding mode: **{policy_label}**", icon="ℹ️")

    # If user switches modes, drop a short visible note in the chat.
    if prev_policy and prev_policy != st.session_state.get("policy"):
        note = f"*Grounding mode changed to **{policy_label}**.*"
        st.session_state.messages.append(
            {"role": "assistant", "content": note, "sources": [], "answer_phrases": []}
        )

    _render_history()

    question = st.chat_input("Ask a question about your papers")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not _looks_like_question(question):
        with st.chat_message("assistant"):
            st.markdown("Please ask a question (for example: *What does X mean?*).")
        st.session_state.messages.append(
            {"role": "assistant", "content": "Please ask a question (for example: *What does X mean?*).", "sources": [], "answer_phrases": []}
        )
        st.rerun()
        return

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and answering..."):
            hits = index.query_index(question, n_results=5)
            answer, used_hits = qa.answer_question(question, top_k=5, hits=hits, policy=policy)

        st.markdown(answer)
        answer_phrases = _extract_answer_phrases(answer)

        if used_hits:
            st.markdown("**Sources**")
            for i, hit in enumerate(used_hits, start=1):
                with st.expander(_source_badge(hit, i)):
                    source_text = hit.get("document") or ""
                    highlighted = _highlight_text(source_text, answer_phrases)
                    st.markdown(highlighted, unsafe_allow_html=True)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": used_hits,
            "answer_phrases": answer_phrases,
        }
    )


if __name__ == "__main__":
    main()

