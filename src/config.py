from pathlib import Path


# Root of the project (one level above the src/ directory)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Documents directory
DATA_DIR: Path = PROJECT_ROOT / "data"
DOCS_DIR: Path = DATA_DIR / "docs"

# Index directory   
INDEX_DIR: Path = DATA_DIR / "index"

# Default embedding model
DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-small"

# Default chat model
DEFAULT_CHAT_MODEL: str = "gpt-4.1-mini"

# Retrieval / QA confidence
# Distance threshold for deciding whether retrieval is confident enough to answer.
# This should be tuned using eval/reports on the current corpus and embedding model.
RETRIEVAL_DISTANCE_THRESHOLD: float = 0.75