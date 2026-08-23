import os
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

# In tree-sitter >= 0.22, you initialize Language from the module directly
try:
    PY_LANGUAGE = Language(tspython.language())
except TypeError:
    PY_LANGUAGE = Language(tspython.language(), "python")

class CodeChunker:
    def __init__(self):
        self.parser = Parser()
        self.parser.language = PY_LANGUAGE
    
    def chunk_file(self, file_path: str):
        if not file_path.endswith('.py'):
            return [] # Currently only supporting Python for dummy testing
            
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
            
        tree = self.parser.parse(bytes(code, "utf8"))
        
        chunks = []
        
        # Simple AST traversal to find function and class definitions
        def traverse(node):
            if node.type in ['function_definition', 'class_definition']:
                # Extract the name of the function/class
                name_node = None
                for child in node.children:
                    if child.type == 'identifier':
                        name_node = child
                        break
                
                name = name_node.text.decode('utf8') if name_node else "unknown"
                
                # Extract basic function calls inside this node for dependency mapping
                calls = []
                def find_calls(subnode):
                    if subnode.type == 'call':
                        for c in subnode.children:
                            if c.type == 'identifier':
                                calls.append(c.text.decode('utf8'))
                            elif c.type == 'attribute':
                                # e.g. obj.method() -> get 'method'
                                for gc in c.children:
                                    if gc.type == 'identifier' and gc.text != c.text:
                                        calls.append(gc.text.decode('utf8'))
                    for child in subnode.children:
                        find_calls(child)
                        
                find_calls(node)
                
                chunks.append({
                    "file_path": file_path,
                    "symbol_name": name,
                    "type": node.type,
                    "start_line": node.start_point[0],
                    "end_line": node.end_point[0],
                    "content": code[node.start_byte:node.end_byte],
                    "calls": list(set(calls))
                })
                
            for child in node.children:
                traverse(child)
                
        traverse(tree.root_node)
        return chunks
