"""Unit tests for storage/db.py, storage/vector_store.py, and storage/jobs.py."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.db import (
    build_store, get_chunks_by_ids, get_all_qname_to_id, get_ids_by_qnames,
    bm25_search,
)
from storage.vector_store import QdrantVectorStore, _point_id
from storage import jobs as job_store


def test_get_chunks_by_ids_roundtrip(data_dir):
    chunks = get_chunks_by_ids(str(data_dir / "store.db"), ["c1", "c3"])
    assert set(chunks.keys()) == {"c1", "c3"}
    assert chunks["c1"]["qualified_name"] == "pkgtest.mod.foo"
    assert chunks["c3"]["parent_class"] is None


def test_get_ids_by_qnames_reverse_lookup(data_dir):
    result = get_ids_by_qnames(str(data_dir / "store.db"), ["pkgtest.mod.Widget.run", "does.not.exist"])
    assert result == {"pkgtest.mod.Widget.run": "c4"}


def test_get_ids_by_qnames_empty_input_returns_empty_dict(data_dir):
    assert get_ids_by_qnames(str(data_dir / "store.db"), []) == {}


def test_get_all_qname_to_id_covers_full_corpus(data_dir):
    """graph_expand_local's regression fix depends on this resolving
    against every chunk, not just a seed-scoped subset -- assert that
    contract directly."""
    mapping = get_all_qname_to_id(str(data_dir / "store.db"))
    assert mapping == {
        "pkgtest.mod.foo": "c1",
        "pkgtest.mod.helper": "c2",
        "pkgtest.mod.Widget": "c3",
        "pkgtest.mod.Widget.run": "c4",
        "pkgtest.other.unrelated": "c5",
    }


def test_bm25_search_ignores_stopwords(data_dir):
    """A natural-language query full of stopwords ("how", "does", "the")
    must not zero out the result set -- FTS5 ANDs terms by default, so this
    only works because bm25_search strips stopwords and ORs the rest
    (see storage/db.py's _STOPWORDS docstring)."""
    results = bm25_search(str(data_dir / "store.db"), "how does the helper work", top_k=5)
    assert any(cid == "c2" for cid, _ in results)


def test_bm25_search_all_stopwords_returns_empty(data_dir):
    assert bm25_search(str(data_dir / "store.db"), "the a is", top_k=5) == []


def test_chunk_id_collision_regression(tmp_path):
    """
    Regression test for the bug documented in the README's bug-fix history:
    two genuinely different chunks (e.g. a @property getter/setter pair)
    can share a qualified_name. chunk_id must be a composite key that keeps
    them distinct -- if it were derived from qualified_name alone, the
    second insert would collide with the first (chunk_id is the primary
    key) and either raise or silently overwrite.
    """
    db_path = tmp_path / "collide.db"
    chunks = [
        {
            "chunk_id": "mod.Foo.value:getter:10", "kind": "method", "name": "value",
            "qualified_name": "mod.Foo.value", "file_path": "mod.py",
            "start_line": 10, "end_line": 12, "source": "@property\ndef value(self):\n    return self._v",
            "docstring": "", "signature": "def value(self):", "parent_class": "Foo",
        },
        {
            "chunk_id": "mod.Foo.value:setter:14", "kind": "method", "name": "value",
            "qualified_name": "mod.Foo.value", "file_path": "mod.py",
            "start_line": 14, "end_line": 16, "source": "@value.setter\ndef value(self, v):\n    self._v = v",
            "docstring": "", "signature": "def value(self, v):", "parent_class": "Foo",
        },
    ]
    build_store(str(db_path), chunks)  # must not raise UNIQUE constraint failed

    stored = get_chunks_by_ids(str(db_path), ["mod.Foo.value:getter:10", "mod.Foo.value:setter:14"])
    assert len(stored) == 2
    assert stored["mod.Foo.value:getter:10"]["source"] != stored["mod.Foo.value:setter:14"]["source"]


def test_vector_store_point_id_distinct_for_different_chunk_ids():
    """Same bug class, in the vector store: Qdrant point IDs are derived
    from chunk_id (since chunk_id itself isn't a valid Qdrant point id
    format). Two distinct chunk_ids -- even ones that would have collided
    under the old qualified_name-only hashing scheme -- must map to two
    distinct point ids, or one chunk's vector would silently overwrite the
    other's in the index."""
    id_a = _point_id("mod.Foo.value:getter:10")
    id_b = _point_id("mod.Foo.value:setter:14")
    assert id_a != id_b
    # Deterministic: re-deriving from the same chunk_id always gives the
    # same point id, so re-ingesting a chunk overwrites its old vector
    # cleanly instead of accumulating a duplicate point.
    assert _point_id("mod.Foo.value:getter:10") == id_a


def test_vector_store_rebuild_replaces_previous_contents():
    """rebuild() must fully replace a repo's collection, not append to it
    -- otherwise re-ingesting a repo whose chunk set shrank would leave
    stale vectors for chunks that no longer exist."""
    store = QdrantVectorStore()
    repo_id = "vector_store_rebuild_test"

    store.rebuild(repo_id, ["a", "b", "c"], np.eye(3, dtype=np.float32))
    first = store.search(repo_id, np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=10)
    assert {cid for cid, _ in first} == {"a", "b", "c"}

    store.rebuild(repo_id, ["x"], np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    second = store.search(repo_id, np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=10)
    assert {cid for cid, _ in second} == {"x"}


def test_job_store_persists_across_connections(tmp_path):
    """The whole point of storage/jobs.py: job state must survive a fresh
    connection (i.e. a process restart), not just live in memory."""
    db_path = str(tmp_path / "jobs.db")
    job_store.create_job(db_path, "job-1", "https://github.com/x/y")

    job = job_store.get_job(db_path, "job-1")
    assert job["status"] == "running"
    assert job["repo_url"] == "https://github.com/x/y"

    job_store.update_job(db_path, "job-1", status="done", repo_id="x_y", num_chunks=42)

    reloaded = job_store.get_job(db_path, "job-1")
    assert reloaded["status"] == "done"
    assert reloaded["repo_id"] == "x_y"
    assert reloaded["num_chunks"] == 42


def test_job_store_unknown_job_returns_none(tmp_path):
    db_path = str(tmp_path / "jobs.db")
    assert job_store.get_job(db_path, "does-not-exist") is None
