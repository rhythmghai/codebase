"""
Given a git URL, clone it, auto-detect the actual source directory (skip
tests/docs/examples/vendored-code noise), and run the full deterministic
ingestion pipeline (chunk -> embed -> store) into a repo-specific workspace.

This is what makes the system "point it at any repo" instead of "hardcoded
to whatever I manually cloned once" -- the /ingest API endpoint calls this.
"""

import re
import shutil
import subprocess
import pickle
from pathlib import Path

from ingestion.chunker import ingest_repo
from ingestion.embedder import get_embedder
from storage.db import build_store


_IGNORE_DIR_NAMES = {
    "tests", "test", "docs", "doc", "examples", "example", "scripts",
    "benchmarks", "node_modules", ".git", "venv", ".venv", "env",
    "__pycache__", "build", "dist", ".github", "vendor",
}


def _slugify(repo_url: str) -> str:
    name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    org = repo_url.rstrip("/").split("/")[-2] if "/" in repo_url.rstrip("/") else ""
    slug = f"{org}_{name}" if org else name
    return re.sub(r"[^a-zA-Z0-9_-]", "-", slug).lower()


def clone_repo(repo_url: str, workspace_root: Path) -> Path:
    slug = _slugify(repo_url)
    dest = workspace_root / slug / "src"
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(dest)],
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")
    return dest


def detect_source_dir(cloned_path: Path, max_depth: int = 2) -> str:
    """
    Heuristic: find the directory (relative to the clone root, up to
    max_depth) containing the most .py files, ignoring tests/docs/vendored
    noise. Returns "." if the repo has no clear nested source package
    (source files sit directly at the repo root).
    """
    candidates: dict[str, int] = {}

    for py_file in cloned_path.rglob("*.py"):
        rel = py_file.relative_to(cloned_path)
        parts = rel.parts
        if any(p in _IGNORE_DIR_NAMES or p.startswith(".") for p in parts):
            continue
        if len(parts) == 1:
            candidates["."] = candidates.get(".", 0) + 1
            continue
        top = parts[0]
        candidates[top] = candidates.get(top, 0) + 1

    if not candidates:
        return "."

    best = max(candidates, key=candidates.get)
    return best


def ingest_from_url(repo_url: str, workspace_root: str = "/tmp/codebase-rag-workspaces") -> dict:
    """
    Full pipeline: clone -> detect source dir -> chunk -> embed -> store.
    Returns paths + stats needed to point a CodebaseRAGPipeline at the result.
    """
    workspace_root_path = Path(workspace_root)
    slug = _slugify(repo_url)
    repo_dir = workspace_root_path / slug

    cloned_path = clone_repo(repo_url, workspace_root_path)
    source_subdir = detect_source_dir(cloned_path)

    data_dir = repo_dir / "data"
    chunks, graph = ingest_repo(str(cloned_path), str(data_dir), subdir=source_subdir)

    if not chunks:
        raise RuntimeError(
            f"No Python source found under detected subdir '{source_subdir}'. "
            f"This repo may not be a Python project, or its source layout wasn't detected correctly."
        )

    embedder = get_embedder("sentence-transformer")
    texts = [f"{c.qualified_name}\n{c.signature}\n{c.docstring}\n{c.source}" for c in chunks]
    vectors = embedder.encode(texts)

    db_path = data_dir / "store.db"
    build_store(str(db_path), [c.__dict__ for c in chunks], vectors)

    embedder_path = data_dir / "embedder.pkl"
    with open(embedder_path, "wb") as f:
        pickle.dump(embedder, f)

    return {
        "repo_url": repo_url,
        "slug": slug,
        "source_subdir": source_subdir,
        "num_chunks": len(chunks),
        "num_files": len({c.file_path for c in chunks}),
        "db_path": str(db_path),
        "graph_path": str(data_dir / "graph.json"),
        "embedder_path": str(embedder_path),
    }
