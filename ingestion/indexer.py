"""Build the three stores the retriever reads.

    python -m ingestion.indexer [repo_path]      # default: current directory

Writes:
  * Qdrant collection ``codebase_chunks``   dense MiniLM vectors + payload
  * index/chunks.json                       every chunk (BM25 is rebuilt from this)
  * index/dependency_graph.json             calls / called_by edges

The collection is recreated on every run so no stale points survive a
re-index, and the BM25 corpus is stored as JSON rather than a pickle, so
loading it never executes code.
"""
import json
import os
import sys
from typing import Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from langchain_community.embeddings import HuggingFaceEmbeddings

from .collect import collect_chunks
from .graph_builder import DependencyGraph

INDEX_DIR = os.environ.get("INDEX_DIR", "index")
COLLECTION = "codebase_chunks"
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384


def connect_qdrant() -> QdrantClient:
    """Qdrant server if reachable, otherwise embedded file mode in ./qdrant_data."""
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))
    try:
        client = QdrantClient(host=host, port=port, timeout=3.0)
        client.get_collections()
        return client
    except Exception:
        print(f"Qdrant not reachable at {host}:{port}; using embedded file mode (./qdrant_data).")
        return QdrantClient(path="qdrant_data")


class Indexer:
    def __init__(self, qdrant: QdrantClient = None, embeddings=None):
        self.qdrant = qdrant or connect_qdrant()
        self.embeddings = embeddings or HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    def index_repo(self, repo_path: str = ".", index_dir: str = INDEX_DIR) -> List[Dict]:
        chunks = collect_chunks(repo_path)
        if not chunks:
            print("No Python chunks found to index.")
            return []

        # 1. Dense vectors: one batched call instead of one call per chunk
        vectors = self.embeddings.embed_documents([c["content"] for c in chunks])
        if self.qdrant.collection_exists(COLLECTION):
            self.qdrant.delete_collection(COLLECTION)
        self.qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        points = [
            PointStruct(id=i, vector=vec, payload={k: v for k, v in c.items() if k != "calls"})
            for i, (c, vec) in enumerate(zip(chunks, vectors))
        ]
        self.qdrant.upsert(collection_name=COLLECTION, points=points)

        # 2. Chunk list for BM25 and lookups (JSON, not pickle)
        os.makedirs(index_dir, exist_ok=True)
        with open(os.path.join(index_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=1)

        # 3. Call graph
        graph = DependencyGraph()
        for c in chunks:
            graph.add_chunk(c)
        graph.save(os.path.join(index_dir, "dependency_graph.json"))

        files = len({c["file_path"] for c in chunks})
        print(f"Indexed {len(chunks)} chunks from {files} files into '{COLLECTION}' and {index_dir}/.")
        return chunks


if __name__ == "__main__":
    Indexer().index_repo(sys.argv[1] if len(sys.argv) > 1 else ".")
