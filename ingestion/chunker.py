"""
Deterministic AST-based chunking + graph construction for a Python codebase.

Design principle: this entire module is pure code, zero LLM calls. Chunk
boundaries follow function/class definitions (not blind token windows), and
the call/import graph is built from the same AST pass. This is the
"deterministic part stays deterministic" layer of the pipeline.

Output:
  - chunks.jsonl   : one JSON object per function/class/module-level chunk
  - graph.json      : adjacency data (calls, imports, defined_in, contains)
"""

import ast
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict


@dataclass
class Chunk:
    chunk_id: str
    kind: str  # "function" | "class" | "method" | "module_docstring"
    name: str
    qualified_name: str  # e.g. fastapi.routing.APIRoute.__init__
    file_path: str
    start_line: int
    end_line: int
    source: str
    docstring: str = ""
    signature: str = ""
    parent_class: str = ""


@dataclass
class GraphData:
    nodes: dict = field(default_factory=dict)   # qualified_name -> {kind, file_path}
    calls: list = field(default_factory=list)    # (caller_qname, callee_name)
    imports: list = field(default_factory=list)  # (file_path, imported_name)
    contains: list = field(default_factory=list) # (class_qname, method_qname)


def _make_id(qualified_name: str) -> str:
    return hashlib.sha1(qualified_name.encode()).hexdigest()[:12]


def _get_signature(node) -> str:
    args = []
    for a in node.args.args:
        args.append(a.arg)
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(args)})"


def _collect_calls(node) -> list:
    """Walk a function/method body and collect names of functions it calls."""
    calls = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            func = n.func
            if isinstance(func, ast.Name):
                calls.append(func.id)
            elif isinstance(func, ast.Attribute):
                calls.append(func.attr)
    return calls


def chunk_file(file_path: Path, repo_root: Path) -> tuple[list[Chunk], GraphData]:
    rel_path = str(file_path.relative_to(repo_root))
    source_text = file_path.read_text(encoding="utf-8", errors="ignore")

    try:
        tree = ast.parse(source_text, filename=rel_path)
    except SyntaxError:
        return [], GraphData()

    lines = source_text.splitlines()
    module_qname = rel_path.replace("/", ".").removesuffix(".py")

    chunks: list[Chunk] = []
    graph = GraphData()

    # module-level docstring as its own chunk (useful for "what does this file do")
    mod_doc = ast.get_docstring(tree)
    if mod_doc:
        chunks.append(Chunk(
            chunk_id=_make_id(module_qname + ".__module__"),
            kind="module_docstring",
            name=Path(rel_path).name,
            qualified_name=module_qname,
            file_path=rel_path,
            start_line=1,
            end_line=1,
            source=mod_doc,
            docstring=mod_doc,
        ))
    graph.nodes[module_qname] = {"kind": "module", "file_path": rel_path}
    _seen_qnames: set[str] = set()

    # imports at module level -> graph edges
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                graph.imports.append((rel_path, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            graph.imports.append((rel_path, node.module))

    def handle_function(node, parent_class: str = ""):
        base_qname = f"{module_qname}.{parent_class + '.' if parent_class else ''}{node.name}"
        # Same-name redefinitions are legal Python and common in the wild:
        # @property getter/setter pairs, @x.setter/@x.deleter, @overload
        # variants. Each is a distinct chunk with distinct source, but they'd
        # otherwise collide on qualified_name -> same chunk_id -> UNIQUE
        # constraint failure at storage time. Disambiguate with the line
        # number, which is always unique within a file.
        qname = base_qname if base_qname not in _seen_qnames else f"{base_qname}@L{node.lineno}"
        _seen_qnames.add(qname)
        cid = _make_id(qname)
        src = "\n".join(lines[node.lineno - 1:node.end_lineno])
        doc = ast.get_docstring(node) or ""
        kind = "method" if parent_class else "function"

        chunks.append(Chunk(
            chunk_id=cid,
            kind=kind,
            name=node.name,
            qualified_name=qname,
            file_path=rel_path,
            start_line=node.lineno,
            end_line=node.end_lineno,
            source=src,
            docstring=doc,
            signature=_get_signature(node),
            parent_class=parent_class,
        ))
        graph.nodes[qname] = {"kind": kind, "file_path": rel_path}

        if parent_class:
            parent_qname = f"{module_qname}.{parent_class}"
            graph.contains.append((parent_qname, qname))

        for callee in _collect_calls(node):
            graph.calls.append((qname, callee))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            handle_function(node)
        elif isinstance(node, ast.ClassDef):
            base_class_qname = f"{module_qname}.{node.name}"
            class_qname = base_class_qname if base_class_qname not in _seen_qnames else f"{base_class_qname}@L{node.lineno}"
            _seen_qnames.add(class_qname)
            src = "\n".join(lines[node.lineno - 1:node.end_lineno])
            doc = ast.get_docstring(node) or ""
            bases = [b.id for b in node.bases if isinstance(b, ast.Name)]

            chunks.append(Chunk(
                chunk_id=_make_id(class_qname),
                kind="class",
                name=node.name,
                qualified_name=class_qname,
                file_path=rel_path,
                start_line=node.lineno,
                end_line=node.end_lineno,
                source=src[:2000],  # class bodies can be huge; cap the raw source
                docstring=doc,
                signature=f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}",
            ))
            graph.nodes[class_qname] = {"kind": "class", "file_path": rel_path}

            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    handle_function(sub, parent_class=node.name)

    return chunks, graph


def ingest_repo(repo_root: str, output_dir: str, subdir: str = "fastapi"):
    repo_root_path = Path(repo_root)
    target_dir = repo_root_path / subdir
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_chunks: list[Chunk] = []
    merged_graph = GraphData()

    py_files = sorted(target_dir.rglob("*.py"))
    py_files = [f for f in py_files if "__pycache__" not in str(f)]

    for f in py_files:
        chunks, graph = chunk_file(f, repo_root_path)
        all_chunks.extend(chunks)
        merged_graph.nodes.update(graph.nodes)
        merged_graph.calls.extend(graph.calls)
        merged_graph.imports.extend(graph.imports)
        merged_graph.contains.extend(graph.contains)

    with open(out / "chunks.jsonl", "w") as fh:
        for c in all_chunks:
            fh.write(json.dumps(asdict(c)) + "\n")

    with open(out / "graph.json", "w") as fh:
        json.dump(asdict(merged_graph), fh, indent=2)

    print(f"Files processed: {len(py_files)}")
    print(f"Chunks extracted: {len(all_chunks)}")
    print(f"  functions/methods: {sum(1 for c in all_chunks if c.kind in ('function','method'))}")
    print(f"  classes: {sum(1 for c in all_chunks if c.kind == 'class')}")
    print(f"Graph nodes: {len(merged_graph.nodes)}")
    print(f"Call edges: {len(merged_graph.calls)}")
    print(f"Import edges: {len(merged_graph.imports)}")
    print(f"Contains edges: {len(merged_graph.contains)}")

    return all_chunks, merged_graph


if __name__ == "__main__":
    ingest_repo(
        repo_root="/home/claude/codebase-rag/repo_src",
        output_dir="/home/claude/codebase-rag/data",
        subdir="fastapi",
    )