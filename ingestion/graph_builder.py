import json
from collections import defaultdict

class DependencyGraph:
    def __init__(self):
        # symbol_name -> {"calls": set(), "called_by": set()}
        self.graph = defaultdict(lambda: {"calls": set(), "called_by": set()})
        
    def add_chunk(self, chunk):
        symbol = chunk['symbol_name']
        calls = chunk['calls']
        
        for call in calls:
            self.graph[symbol]["calls"].add(call)
            self.graph[call]["called_by"].add(symbol)
            
    def save(self, filepath="dependency_graph.json"):
        # Convert sets to lists for JSON serialization
        serializable = {
            k: {
                "calls": list(v["calls"]),
                "called_by": list(v["called_by"])
            }
            for k, v in self.graph.items()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2)
            
    def load(self, filepath="dependency_graph.json"):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.graph = defaultdict(lambda: {"calls": set(), "called_by": set()})
            for k, v in data.items():
                self.graph[k]["calls"] = set(v["calls"])
                self.graph[k]["called_by"] = set(v["called_by"])
