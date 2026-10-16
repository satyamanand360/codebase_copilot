"""Hybrid retrieval: dense (Qdrant) + sparse (BM25), fused with Reciprocal Rank Fusion."""
import json
import os
from typing import Dict, List

from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Plus

from .fusion import rrf_fuse
from .tokenizer import code_tokens

INDEX_DIR = os.environ.get("INDEX_DIR", "index")
COLLECTION = "codebase_chunks"


class HybridSearcher:
    def __init__(self, qdrant: QdrantClient = None, embeddings=None, index_dir: str = INDEX_DIR):
        if qdrant is None:
            host = os.environ.get("QDRANT_HOST", "localhost")
            port = int(os.environ.get("QDRANT_PORT", "6333"))
            try:
                qdrant = QdrantClient(host=host, port=port, timeout=3.0)
                qdrant.get_collections()
            except Exception:
                qdrant = QdrantClient(path="qdrant_data")
        self.qdrant = qdrant
        self.embeddings = embeddings or HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.dense_available = True

        chunks_path = os.path.join(index_dir, "chunks.json")
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.chunks: List[Dict] = json.load(f)
        except FileNotFoundError:
            self.chunks = []
            print(f"Warning: {chunks_path} not found. Run `python -m ingestion.indexer` first.")

        self.chunk_map = {c["chunk_id"]: c for c in self.chunks}
        corpus = [code_tokens(c["content"]) for c in self.chunks]
        self.token_sets = [set(toks) for toks in corpus]
        # BM25Plus keeps IDF positive. BM25Okapi's IDF goes negative for words in
        # more than half the chunks, which scrambles rankings on small repos.
        self.bm25 = BM25Plus(corpus) if corpus else None

    # ------------------------------------------------------------ retrievers
    def dense_ids(self, query: str, top_k: int) -> List[str]:
        try:
            vector = self.embeddings.embed_query(query)
            hits = self.qdrant.query_points(collection_name=COLLECTION, query=vector, limit=top_k).points
            return [h.payload["chunk_id"] for h in hits]
        except Exception as e:
            if self.dense_available:
                print(f"Warning: dense search unavailable ({e}); continuing with BM25 only.")
            self.dense_available = False
            return []

    def sparse_ids(self, query: str, top_k: int) -> List[str]:
        if not self.bm25:
            return []
        tokens = code_tokens(query)
        query_set = set(tokens)
        scores = self.bm25.get_scores(tokens)
        # a chunk sharing no token with the query should not earn any RRF credit
        matching = [i for i in range(len(scores)) if self.token_sets[i] & query_set]
        ranked = sorted(matching, key=lambda i: scores[i], reverse=True)
        return [self.chunks[i]["chunk_id"] for i in ranked[:top_k]]

    # ----------------------------------------------------------------- search
    def search(self, query: str, top_k: int = 30, mode: str = "hybrid") -> List[Dict]:
        """mode: 'hybrid' (default), 'dense' or 'bm25' (the last two are for ablations)."""
        if not self.chunks:
            return []
        if mode == "dense":
            ids = self.dense_ids(query, top_k)
        elif mode == "bm25":
            ids = self.sparse_ids(query, top_k)
        else:
            ids = rrf_fuse([self.dense_ids(query, top_k), self.sparse_ids(query, top_k)], k=60, top_k=top_k)
        return [self.chunk_map[i] for i in ids if i in self.chunk_map]
