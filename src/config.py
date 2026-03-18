from pathlib import Path

# Root of the project (one level above the src/ directory)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Where raw documents live
DATA_DIR: Path = PROJECT_ROOT / "data"
DOCS_DIR: Path = DATA_DIR / "docs"

# Later we’ll add things like:
# INDEX_DIR = DATA_DIR / "index"
# DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
# DEFAULT_CHAT_MODEL = "gpt-4.1-mini"