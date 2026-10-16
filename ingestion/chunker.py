"""Split Python files into retrievable chunks using Tree-sitter.

One chunk per function or method (full source, decorators included), one
"header" chunk per class (signature, docstring, class attributes and the list
of its methods, so method bodies are not duplicated), and one module chunk per
file holding everything at top level that is not a def or class (imports,
constants, the ``if __name__ == "__main__"`` block).

Each chunk records the bare names it calls, which the dependency graph uses.
Line numbers are 1-based so citations match what an editor shows.
"""
import os
from typing import Dict, List, Optional

from tree_sitter import Language, Parser
import tree_sitter_python as tspython

# tree-sitter >= 0.22 builds a Language straight from the grammar module
try:
    PY_LANGUAGE = Language(tspython.language())
except TypeError:  # older bindings
    PY_LANGUAGE = Language(tspython.language(), "python")

DEF_TYPES = ("function_definition", "class_definition")
WRAPPER_TYPES = DEF_TYPES + ("decorated_definition",)


def _text(source: bytes, start: int, end: int) -> str:
    # Tree-sitter offsets are byte offsets, so slice bytes, not str.
    return source[start:end].decode("utf8", errors="replace")


class CodeChunker:
    def __init__(self):
        self.parser = Parser()
        self.parser.language = PY_LANGUAGE

    # ------------------------------------------------------------------ public
    def chunk_file(self, file_path: str, rel_path: Optional[str] = None) -> List[Dict]:
        if not file_path.endswith(".py"):
            return []  # only Python is supported for now

        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read().encode("utf8")

        tree = self.parser.parse(source)
        path = (rel_path or file_path).replace("\\", "/")

        chunks: List[Dict] = []
        self._walk(tree.root_node, source, path, [], chunks)

        module_chunk = self._module_chunk(tree.root_node, source, path)
        if module_chunk:
            chunks.insert(0, module_chunk)
        return chunks

    # ----------------------------------------------------------------- helpers
    def _walk(self, node, source: bytes, path: str, scope: List[str], chunks: List[Dict]):
        for child in node.children:
            if child.type in DEF_TYPES:
                self._emit(child, source, path, scope, chunks)
            else:
                self._walk(child, source, path, scope, chunks)

    def _emit(self, node, source: bytes, path: str, scope: List[str], chunks: List[Dict]):
        name_node = node.child_by_field_name("name")
        name = name_node.text.decode("utf8") if name_node else "unknown"
        qualified = ".".join(scope + [name])

        outer = node
        if node.parent is not None and node.parent.type == "decorated_definition":
            outer = node.parent  # include decorators in the chunk

        start_line = outer.start_point[0] + 1
        if node.type == "class_definition":
            content, end_line = self._class_header(node, outer, source, start_line)
        else:
            content = _text(source, outer.start_byte, outer.end_byte)
            end_line = outer.end_point[0] + 1

        chunks.append({
            "chunk_id": f"{path}::{qualified}",
            "file_path": path,
            "symbol_name": name,
            "qualified_name": qualified,
            "type": node.type,
            "start_line": start_line,
            "end_line": end_line,
            "content": content,
            "calls": sorted(self._find_calls(node)),
        })

        # methods and nested functions become their own chunks
        body = node.child_by_field_name("body")
        if body is not None:
            self._walk(body, source, path, scope + [name], chunks)

    def _class_header(self, node, outer, source: bytes, start_line: int):
        body = node.child_by_field_name("body")
        first_def = None
        methods = []
        if body is not None:
            for child in body.children:
                if child.type in WRAPPER_TYPES:
                    first_def = first_def or child
                    inner = child
                    if child.type == "decorated_definition":
                        inner = child.child_by_field_name("definition") or child
                    name_node = inner.child_by_field_name("name")
                    if name_node is not None and inner.type == "function_definition":
                        methods.append(name_node.text.decode("utf8"))

        end_byte = first_def.start_byte if first_def is not None else outer.end_byte
        header = _text(source, outer.start_byte, end_byte).rstrip()
        end_line = start_line + header.count("\n")
        if methods:
            header += "\n    # methods: " + ", ".join(methods)
        return header, end_line

    def _find_calls(self, node) -> set:
        """Names called inside ``node``, not descending into nested defs."""
        return self._calls_in(node.children)

    def _calls_in(self, nodes) -> set:
        calls = set()
        stack = list(nodes)
        while stack:
            n = stack.pop()
            if n.type in WRAPPER_TYPES:
                continue  # nested defs/classes are separate chunks
            if n.type == "call":
                fn = n.child_by_field_name("function")
                if fn is not None and fn.type == "identifier":
                    calls.add(fn.text.decode("utf8"))
                elif fn is not None and fn.type == "attribute":
                    # obj.method() -> keep only "method", never "obj"/"self"
                    attr = fn.child_by_field_name("attribute")
                    if attr is not None:
                        calls.add(attr.text.decode("utf8"))
            stack.extend(n.children)
        return calls

    def _module_chunk(self, root, source: bytes, path: str) -> Optional[Dict]:
        parts = [c for c in root.children if c.type not in WRAPPER_TYPES]
        if not parts:
            return None
        content = "\n".join(_text(source, c.start_byte, c.end_byte) for c in parts).strip()
        if not content:
            return None
        calls = self._calls_in(parts)
        return {
            "chunk_id": f"{path}::<module>",
            "file_path": path,
            "symbol_name": f"<module:{path}>",
            "qualified_name": "<module>",
            "type": "module",
            "start_line": parts[0].start_point[0] + 1,
            "end_line": parts[-1].end_point[0] + 1,
            "content": content,
            "calls": sorted(calls),
        }
