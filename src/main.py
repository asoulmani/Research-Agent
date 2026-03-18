from . import config
from . import ingest


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


if __name__ == "__main__":
    main()