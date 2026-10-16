"""Find Python files and chunk them. Kept free of heavy imports so tests can use it."""
import os
from typing import Dict, List

from .chunker import CodeChunker

SKIP_DIRS = {"venv", "env", "__pycache__", "node_modules", "qdrant_data", "qdrant_storage", "index"}


def skip_dirs() -> set:
    """Default skips plus any listed in INDEX_SKIP_DIRS, e.g. INDEX_SKIP_DIRS=tests,scripts."""
    extra = {d.strip() for d in os.environ.get("INDEX_SKIP_DIRS", "").split(",") if d.strip()}
    return SKIP_DIRS | extra


def iter_python_files(repo_path: str):
    skip = skip_dirs()
    for root, dirs, files in os.walk(repo_path):
        # prune in place so os.walk never descends into hidden or vendored dirs
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in skip)
        for name in sorted(files):
            if name.endswith(".py"):
                yield os.path.join(root, name)


def collect_chunks(repo_path: str) -> List[Dict]:
    chunker = CodeChunker()
    chunks: List[Dict] = []
    for full_path in iter_python_files(repo_path):
        rel = os.path.relpath(full_path, repo_path)
        chunks.extend(chunker.chunk_file(full_path, rel_path=rel))
    return chunks
