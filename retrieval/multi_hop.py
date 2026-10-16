"""One-hop expansion along the call graph (callees and callers)."""
import json
import os
from collections import defaultdict
from typing import Dict, List

INDEX_DIR = os.environ.get("INDEX_DIR", "index")


class MultiHopExpander:
    def __init__(self, graph_path: str = None, max_neighbors: int = 4, max_defs_per_name: int = 2):
        """
        max_neighbors:      cap on chunks added per retrieved chunk (stops a widely
                            used helper from flooding the prompt with its callers)
        max_defs_per_name:  the graph is keyed by bare names, so a name defined in
                            more places than this (e.g. __init__, format) is too
                            ambiguous to follow and is skipped
        """
        graph_path = graph_path or os.path.join(INDEX_DIR, "dependency_graph.json")
        self.max_neighbors = max_neighbors
        self.max_defs_per_name = max_defs_per_name
        self.graph: Dict = {}
        if os.path.exists(graph_path):
            with open(graph_path, "r", encoding="utf-8") as f:
                self.graph = json.load(f)
        else:
            print(f"Warning: graph file {graph_path} not found.")

    def expand(self, chunks: List[Dict], all_chunks: List[Dict]) -> List[Dict]:
        if not self.graph or not all_chunks:
            return list(chunks)

        by_name = defaultdict(list)
        for c in all_chunks:
            by_name[c["symbol_name"]].append(c)

        seen = {c["chunk_id"] for c in chunks}
        result = list(chunks)
        for chunk in chunks:
            node = self.graph.get(chunk["symbol_name"], {})
            # callees first: they explain what this code does
            neighbours = list(node.get("calls", [])) + list(node.get("called_by", []))
            added = 0
            for name in neighbours:
                defs = by_name.get(name, [])
                if not defs or len(defs) > self.max_defs_per_name:
                    continue
                for d in defs:
                    if added >= self.max_neighbors:
                        break
                    if d["chunk_id"] not in seen:
                        seen.add(d["chunk_id"])
                        result.append(d)
                        added += 1
        return result
