"""
Shared fixtures for the test suite.

Deliberately avoids network access and heavy ML models (sentence-
transformers/torch, a real Gemini call, a real Neo4j instance) so the suite
runs fast and offline in CI: TfidfEmbedder (a real, lightweight embedder
already shipped in ingestion/embedder.py, not test-only fake code) stands
in for the neural embedder, LexicalReranker and RuleBasedLLM are the
project's own dependency-free defaults, and Neo4j is simply left
unconfigured so CodebaseRAGPipeline's existing fallback-to-local-JSON-graph
path is exercised for free.

Qdrant (storage/vector_store.py) is a real dependency here, not mocked --
its embedded mode needs no network/server, so it's cheap to use directly.
QDRANT_PATH is pinned to a session-scoped temp directory below, set before
any fixture can trigger the module-level singleton client's first
construction, so the test suite never touches the real project's
data/qdrant directory. Collections are per-repo_id, and every fixture that
populates one calls rebuild() (drop + recreate), so tests sharing the same
underlying Qdrant storage location never see stale data from each other.
"""

import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Must be set before storage/vector_store.py's singleton client is first
# constructed by any fixture/test -- see the module docstring above.
os.environ.setdefault("QDRANT_PATH", tempfile.mkdtemp(prefix="coderag-test-qdrant-"))

from ingestion.embedder import TfidfEmbedder
from storage.db import build_store
from storage.vector_store import QdrantVectorStore

# Shared by every fixture/test that needs a stable repo_id -- must match
# between whatever populates a Qdrant collection and whatever pipeline
# queries it.
TEST_REPO_ID = "test_repo"

# A tiny synthetic corpus with real call/contains relationships, including
# one case (foo -> helper, and Widget -> Widget.run) specifically shaped to
# exercise graph_expand_local's neighbor resolution across seed vs.
# full-corpus qname scope -- the exact bug class documented in the README's
# bug-fix history (a neighbor genuinely outside the seed set must still be
# resolvable).
SAMPLE_CHUNKS = [
    {
        "chunk_id": "c1", "kind": "function", "name": "foo",
        "qualified_name": "pkgtest.mod.foo", "file_path": "pkgtest/mod.py",
        "start_line": 1, "end_line": 5,
        "source": "def foo():\n    return helper()",
        "docstring": "Calls helper to do the real work.",
        "signature": "def foo():", "parent_class": None,
    },
    {
        "chunk_id": "c2", "kind": "function", "name": "helper",
        "qualified_name": "pkgtest.mod.helper", "file_path": "pkgtest/mod.py",
        "start_line": 7, "end_line": 9,
        "source": "def helper():\n    return 42",
        "docstring": "Returns the answer to everything.",
        "signature": "def helper():", "parent_class": None,
    },
    {
        "chunk_id": "c3", "kind": "class", "name": "Widget",
        "qualified_name": "pkgtest.mod.Widget", "file_path": "pkgtest/mod.py",
        "start_line": 11, "end_line": 20,
        "source": "class Widget:\n    def run(self):\n        return helper()",
        "docstring": "A widget class that runs things.",
        "signature": "class Widget:", "parent_class": None,
    },
    {
        "chunk_id": "c4", "kind": "method", "name": "run",
        "qualified_name": "pkgtest.mod.Widget.run", "file_path": "pkgtest/mod.py",
        "start_line": 13, "end_line": 15,
        "source": "    def run(self):\n        return helper()",
        "docstring": "Runs the widget by calling helper.",
        "signature": "def run(self):", "parent_class": "Widget",
    },
    {
        "chunk_id": "c5", "kind": "function", "name": "unrelated",
        "qualified_name": "pkgtest.other.unrelated", "file_path": "pkgtest/other.py",
        "start_line": 1, "end_line": 3,
        "source": "def unrelated():\n    return None",
        "docstring": "Not connected to anything else in the graph.",
        "signature": "def unrelated():", "parent_class": None,
    },
]

SAMPLE_GRAPH = {
    "nodes": [c["qualified_name"] for c in SAMPLE_CHUNKS],
    "calls": [
        ["pkgtest.mod.foo", "helper"],
        ["pkgtest.mod.Widget.run", "helper"],
    ],
    "imports": [],
    "contains": [
        ["pkgtest.mod.Widget", "pkgtest.mod.Widget.run"],
    ],
}


def _chunk_text(c: dict) -> str:
    return f"{c['qualified_name']}\n{c['signature']}\n{c['docstring']}\n{c['source']}"


@pytest.fixture
def sample_chunks():
    return [dict(c) for c in SAMPLE_CHUNKS]


@pytest.fixture
def data_dir(tmp_path, sample_chunks):
    """
    A fully populated data directory (store.db + graph.json + embedder.pkl)
    built from SAMPLE_CHUNKS, in the same shape ingestion/run_ingestion.py
    produces for a real repo -- lets tests exercise the real storage/
    retrieval/pipeline code paths against known, small, hand-verifiable
    data instead of mocking them out.
    """
    d = tmp_path / "data"
    d.mkdir()

    texts = [_chunk_text(c) for c in sample_chunks]
    embedder = TfidfEmbedder(dim=4)
    embedder.fit(texts)
    vectors = embedder.encode(texts)

    build_store(str(d / "store.db"), sample_chunks)
    QdrantVectorStore().rebuild(TEST_REPO_ID, [c["chunk_id"] for c in sample_chunks], vectors)

    with open(d / "graph.json", "w") as f:
        json.dump(SAMPLE_GRAPH, f)

    with open(d / "embedder.pkl", "wb") as f:
        pickle.dump(embedder, f)

    return d
