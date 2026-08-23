import os
import pickle
from typing import List, Dict
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings

class HybridSearcher:
    def __init__(self):
        # Fallback to in-memory for testing, similar to the indexer
        qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
        try:
            self.qdrant = QdrantClient(host=qdrant_host, port=6333, timeout=3.0)
            self.qdrant.get_collections()
        except Exception:
            self.qdrant = QdrantClient(path="qdrant_data")
            
        self.collection_name = "codebase_chunks"
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load BM25 index
        try:
            with open("bm25_index.pkl", "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.chunks = data["chunks"]
        except FileNotFoundError:
            self.bm25 = None
            self.chunks = []
            print("Warning: bm25_index.pkl not found. Hybrid search will fail.")

    def _rrf_score(self, rank: int, k=60):
        return 1.0 / (k + rank)

    def search(self, query: str, top_k: int = 30) -> List[Dict]:
        """Performs hybrid search using Dense vectors and Sparse BM25 + Reciprocal Rank Fusion."""
        if not self.bm25 or not self.chunks:
            return []

        # 1. Dense Search (Qdrant)
        vector = self.embeddings.embed_query(query)
        try:
            dense_results = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=top_k
            )
            
            dense_ranked = []
            for hit in dense_results.points:
                payload = hit.payload
                # We need a unique identifier to merge results; we'll use file_path + symbol_name
                chunk_id = f"{payload['file_path']}::{payload['symbol_name']}"
                dense_ranked.append(chunk_id)
        except Exception as e:
            print(f"Qdrant search failed: {e}")
            dense_ranked = []

        # 2. Sparse Search (BM25)
        tokenized_query = query.lower().split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k sparse indices
        sparse_top_indices = sorted(range(len(sparse_scores)), key=lambda i: sparse_scores[i], reverse=True)[:top_k]
        
        sparse_ranked = []
        for i in sparse_top_indices:
            chunk = self.chunks[i]
            chunk_id = f"{chunk['file_path']}::{chunk['symbol_name']}"
            sparse_ranked.append(chunk_id)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for rank, chunk_id in enumerate(dense_ranked):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + self._rrf_score(rank + 1)
            
        for rank, chunk_id in enumerate(sparse_ranked):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + self._rrf_score(rank + 1)

        # Sort by RRF score
        sorted_chunk_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Map back to full chunk dicts
        # Create a lookup map for chunks
        chunk_map = {f"{c['file_path']}::{c['symbol_name']}": c for c in self.chunks}
        
        results = []
        for chunk_id, score in sorted_chunk_ids:
            if chunk_id in chunk_map:
                results.append(chunk_map[chunk_id])
                
        return results
