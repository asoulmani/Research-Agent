from dotenv import load_dotenv

from . import config
from . import ingest
from . import index
from . import qa

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
    chunked = ingest.chunk_documents(docs, strategy="token")  # Choices: "simple" and "token"

    total_chunks = sum(len(chunks) for chunks in chunked.values())
    print(f"[INFO] Created {total_chunks} chunks across documents.\n")

    # Build vector index (requires OPENAI_API_KEY in .env)
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
    test_q = "What is my name? And how many sports do I play and cite them?"
    hits = index.query_index(test_q, n_results=3)
    print(f"[INFO] Sample query: {test_q!r}")
    for i, hit in enumerate(hits, start=1):
        meta = hit.get("metadata") or {}
        src = meta.get("source_file", "?")
        preview = (hit.get("document") or "")[:160].replace("\n", " ")
        print(f"  {i}. {src} | {preview!r}...")
        
    # LLM answer 
    try:
        answer, used_hits = qa.answer_question(test_q, top_k=3, hits=hits)
        print(f"\n[INFO] Answer:\n{answer}\n")

        if used_hits:
            print("[INFO] Sources used:")
            for j, hit in enumerate(used_hits, 1):
                meta = hit.get("metadata") or {}
                src = meta.get("source_file", "?")
                chunk_index = meta.get("chunk_index", "?")
                print(f"  [{j}] {src} (chunk {chunk_index})")

                # Debug: print the exact chunk text that the model saw (used for this citation).
                chunk_text = hit.get("document") or ""
                print("  [DEBUG] Chunk text:")
                print(chunk_text)
                print()
    except Exception as e:
        print(f"[SKIP] QA step: {e}")


if __name__ == "__main__":
    main()