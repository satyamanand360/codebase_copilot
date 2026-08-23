import os
import pickle
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from .chunker import CodeChunker
from .graph_builder import DependencyGraph

class Indexer:
    def __init__(self):
        # We assume Qdrant is running locally, or if not, use an in-memory instance for testing
        # Since docker compose failed in Phase 0, we'll gracefully fallback to in-memory for the demo.
        qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
        try:
            self.qdrant = QdrantClient(host=qdrant_host, port=6333, timeout=3.0)
            self.qdrant.get_collections()
        except Exception:
            print("Warning: Could not connect to Qdrant on localhost:6333. Using local file-based Qdrant.")
            self.qdrant = QdrantClient(path="qdrant_data")
            
        self.collection_name = "codebase_chunks"
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.bm25 = None
        self.chunks_data = []

        # Setup Qdrant collection
        try:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
        except Exception as e:
            # Collection might already exist
            pass

    def index_repo(self, repo_path: str):
        chunker = CodeChunker()
        graph = DependencyGraph()
        all_chunks = []

        # Walk directory
        for root, _, files in os.walk(repo_path):
            # Skip hidden dirs and envs
            if any(part.startswith('.') or part == 'venv' for part in root.split(os.sep)):
                continue
                
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    chunks = chunker.chunk_file(full_path)
                    for c in chunks:
                        graph.add_chunk(c)
                    all_chunks.extend(chunks)

        self.chunks_data = all_chunks
        
        if not all_chunks:
            print("No valid code chunks found to index.")
            return

        # 1. Vector Indexing (Qdrant)
        points = []
        for i, chunk in enumerate(all_chunks):
            vector = self.embeddings.embed_query(chunk["content"])
            points.append(
                PointStruct(
                    id=i,
                    vector=vector,
                    payload={
                        "file_path": chunk["file_path"],
                        "symbol_name": chunk["symbol_name"],
                        "type": chunk["type"],
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                        "content": chunk["content"]
                    }
                )
            )
        
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points
        )

        # 2. Sparse Indexing (BM25)
        tokenized_corpus = [chunk["content"].lower().split() for chunk in all_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        with open("bm25_index.pkl", "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": all_chunks}, f)

        # 3. Save Graph
        graph.save("dependency_graph.json")
        
        print(f"Successfully indexed {len(all_chunks)} chunks.")

if __name__ == "__main__":
    indexer = Indexer()
    # If run directly, run on current directory
    indexer.index_repo(".")
