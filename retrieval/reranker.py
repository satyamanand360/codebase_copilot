from typing import List, Dict
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # We use a lightweight cross-encoder model
        self.model = CrossEncoder(model_name, max_length=512)
        
    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """Reranks the retrieved chunks against the query."""
        if not chunks:
            return []
            
        # Prepare inputs: list of (query, chunk_content) pairs
        model_inputs = [[query, chunk["content"]] for chunk in chunks]
        
        # Get cross-encoder scores
        scores = self.model.predict(model_inputs)
        
        # Combine chunks with scores
        scored_chunks = list(zip(chunks, scores))
        
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top-k chunks
        return [chunk for chunk, score in scored_chunks[:top_k]]
