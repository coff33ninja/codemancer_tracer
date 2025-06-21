import ast
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

def find_py_files(folder: Path) -> List[Path]:
    """Recursively finds all Python files in a given folder, skipping venv, .venv, __pycache__, and .vscode folders."""
    skip_dirs = {"venv", ".venv", "__pycache__", ".vscode"}
    py_files = []
    for path in folder.rglob("*.py"):
        if any(part in skip_dirs for part in path.parts):
            continue
        py_files.append(path)
    return py_files

def parse_ast(filepath: Path) -> ast.AST | None:
    """Parses a Python file into an AST."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return ast.parse(f.read(), filename=str(filepath))
    except Exception as e:
        print(f"[!] Failed to parse {filepath}: {e}")
        return None

def extract_info(tree: ast.AST) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """Extracts functions, imports, and calls from an AST, associating calls with their containing function."""
    functions = []
    imports = []
    calls: Dict[str, List[str]] = defaultdict(list)

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.current_function_signature = None

        def visit_FunctionDef(self, node: ast.FunctionDef):
            args = [arg.arg for arg in node.args.args]
            signature = f"{node.name}({', '.join(args)})"
            functions.append(signature)

            original_signature = self.current_function_signature
            self.current_function_signature = signature
            self.generic_visit(node)
            self.current_function_signature = original_signature

        def visit_Import(self, node: ast.Import):
            for alias in node.names:
                imports.append(alias.name)

        def visit_ImportFrom(self, node: ast.ImportFrom):
            if node.module:
                imports.append(node.module)

        def _get_full_call_name(self, node_func):
            if isinstance(node_func, ast.Name):
                return node_func.id
            elif isinstance(node_func, ast.Attribute):
                parts = []
                curr = node_func
                while isinstance(curr, ast.Attribute):
                    parts.append(curr.attr)
                    curr = curr.value
                if isinstance(curr, ast.Name):
                    parts.append(curr.id)
                    return ".".join(reversed(parts))
            return None

        def visit_Call(self, node: ast.Call):
            call_name = self._get_full_call_name(node.func)
            if call_name:
                scope = self.current_function_signature if self.current_function_signature else "__module__"
                calls[scope].append(call_name)
            self.generic_visit(node)

    Visitor().visit(tree)
    return functions, imports, calls
