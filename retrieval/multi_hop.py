import json
import os
from typing import List, Dict

class MultiHopExpander:
    def __init__(self, graph_path="dependency_graph.json"):
        self.graph = {}
        if os.path.exists(graph_path):
            with open(graph_path, 'r', encoding='utf-8') as f:
                self.graph = json.load(f)
        else:
            print(f"Warning: Graph file {graph_path} not found.")
            
    def expand(self, chunks: List[Dict], all_chunks: List[Dict]) -> List[Dict]:
        """
        Expands the initial set of chunks by pulling in direct callers and callees
        from the dependency graph.
        """
        if not self.graph or not all_chunks:
            return chunks
            
        expanded_symbols = set([c['symbol_name'] for c in chunks])
        new_chunks = list(chunks)
        
        chunk_map = {c['symbol_name']: c for c in all_chunks}
        
        for chunk in chunks:
            symbol = chunk['symbol_name']
            if symbol in self.graph:
                # Add things this symbol calls
                for callee in self.graph[symbol].get("calls", []):
                    if callee in chunk_map and callee not in expanded_symbols:
                        new_chunks.append(chunk_map[callee])
                        expanded_symbols.add(callee)
                
                # Add things that call this symbol
                for caller in self.graph[symbol].get("called_by", []):
                    if caller in chunk_map and caller not in expanded_symbols:
                        new_chunks.append(chunk_map[caller])
                        expanded_symbols.add(caller)
                        
        return new_chunks
