"""
Vector index: embed chunks with OpenAI, store in Chroma, query by semantic similarity.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import chromadb
from chromadb.utils import embedding_functions

from . import config

COLLECTION_NAME = "research_docs"


def _require_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "[ERROR] OPENAI_API_KEY is not set. Add it to a .env file in the project root "
            "(see .env.example) or export it in your shell."
        )
    return key


def _embedding_function():
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=_require_api_key(),
        model_name=config.DEFAULT_EMBEDDING_MODEL,
    )


def _client():
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.INDEX_DIR))


def index_documents(chunked: Dict[str, List[str]]) -> int:
    """
    Embed all chunks and persist them in Chroma under INDEX_DIR.
    Rebuilds the collection from scratch each time (simple V1).

    Returns the number of chunks indexed.
    """
    _require_api_key()
    ef = _embedding_function()
    client = _client()

    # Delete collection if it exists to avoid duplicates
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"embedding_model": config.DEFAULT_EMBEDDING_MODEL},
    )

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for doc_name, chunks in chunked.items():
        for i, text in enumerate(chunks):
            if not text.strip():
                continue
            ids.append(f"{doc_name}::{i}")  # Unique identifier for each chunk
            documents.append(text)
            metadatas.append(
                {
                    "source_file": doc_name,
                    "chunk_index": int(i),
                }
            )

    if not ids:
        print("[WARN] No chunks to index.")
        return 0

    # Chroma add in batches to avoid huge single requests
    batch_size = 100
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"[INFO] Indexed {len(ids)} chunks into Chroma at {config.INDEX_DIR}")
    return len(ids)


def query_index(question: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k chunks most similar to the question.

    Each item: id, document, metadata, distance (if available).
    """
    _require_api_key()
    ef = _embedding_function()
    client = _client()

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )

    cnt = collection.count()
    if cnt == 0:
        return []

    k = min(n_results, cnt)  # Limit # of results to the # of chunks in the collection
    result = collection.query(
        query_texts=[question],
        n_results=k,
    )

    if not result["ids"] or not result["ids"][0]:
        return []

    out: List[Dict[str, Any]] = []
    ids_batch = result["ids"][0]  # 0 because we only query one question
    docs_batch = result["documents"][0] if result.get("documents") else []
    meta_batch = result["metadatas"][0] if result.get("metadatas") else []
    dist_batch = result["distances"][0] if result.get("distances") else [None] * len(ids_batch)

    for i, cid in enumerate(ids_batch):
        out.append(
            {
                "id": cid,
                "document": docs_batch[i] if i < len(docs_batch) else "",
                "metadata": meta_batch[i] if i < len(meta_batch) else {},
                "distance": dist_batch[i] if i < len(dist_batch) else None,
            }
        )
    return out