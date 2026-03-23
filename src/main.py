from dotenv import load_dotenv

from . import config
from . import ingest
from . import index


# Load environment variables
load_dotenv()


def main() -> None:
    print(f"[INFO] Project root (from config): {config.PROJECT_ROOT}")
    print(f"[INFO] Docs directory (from config): {config.DOCS_DIR}\n")

    # Load full text of all supported documents
    docs = ingest.load_documents()
    if not docs:
        print("[INFO] No supported documents found to ingest.")
        return

    print(f"[INFO] Loaded {len(docs)} documents.")

    # Chunk them into smaller pieces
    chunked = ingest.chunk_documents(docs)

    total_chunks = sum(len(chunks) for chunks in chunked.values())
    print(f"[INFO] Created {total_chunks} chunks across documents.\n")

    # Show a short preview per document
    for name, chunks in chunked.items():
        print(f"--- {name} ---")
        print(f"  Chunks: {len(chunks)}")
        if chunks:
            preview = chunks[0][:200].replace("\n", " ")
            print(f"  First chunk preview: {preview!r}")
        print()

    # Build vector index (requires OPENAI_API_KEY in .env or environment)
    try:
        indexed = index.index_documents(chunked)
        print(f"[INFO] Vector index ready ({indexed} vectors).\n")
    except RuntimeError as e:
        print(f"[SKIP] Indexing: {e}")
        return
    except Exception as e:
        print(f"[ERROR] Indexing failed: {e}")
        return

    # Quick retrieval test
    test_q = "What is self-attention?"
    hits = index.query_index(test_q, n_results=3)
    print(f"[INFO] Sample query: {test_q!r}")
    for i, hit in enumerate(hits, 1):
        meta = hit.get("metadata") or {}
        src = meta.get("source_file", "?")
        preview = (hit.get("document") or "")[:160].replace("\n", " ")
        print(f"  {i}. {src} | {preview!r}...")


if __name__ == "__main__":
    main()