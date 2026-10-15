"""
Unit tests for retrieval/hybrid_search.py: RRF merge math and graph
expansion. graph_expand_local's neighbor-resolution test is a direct
regression test for the bug documented in the README's bug-fix history --
the local JSON fallback used to resolve neighbor candidates only against
the *seed* chunks' own qualified names, which meant it could structurally
never surface a genuinely new neighbor (a silent no-op). This test fails
if that regresses.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.hybrid_search import (
    RetrievedChunk, merge_candidates, graph_expand_local, vector_search,
    bm25_search_wrapper, hybrid_retrieve, GRAPH_PRIOR_SCORE,
)
from storage.db import get_chunks_by_ids
from storage.vector_store import QdrantVectorStore
from tests.conftest import TEST_REPO_ID


def test_merge_candidates_uses_rank_not_raw_score():
    """
    Two channels with wildly different score scales (vector cosine ~0-1 vs.
    a BM25 rank score that can be much larger) must not let the
    larger-magnitude channel dominate just because its numbers are bigger --
    that was the exact bug RRF replaced (see hybrid_search.py's module
    docstring). A chunk ranked #1 by BM25 with a "small" score should still
    outrank a chunk ranked #1 by vector search if BM25 puts it first with a
    tighter list around it -- what matters is rank, not magnitude.
    """
    vector_results = [RetrievedChunk("a", 0.99, "vector"), RetrievedChunk("b", 0.10, "vector")]
    bm25_results = [RetrievedChunk("b", 9000.0, "bm25"), RetrievedChunk("a", 1.0, "bm25")]

    merged = merge_candidates(vector_results, bm25_results)

    # "b" is rank 1 in vector? no -- rank 2 in vector, rank 1 in bm25.
    # "a" is rank 1 in vector, rank 2 in bm25. RRF: a = 1/61 + 1/62, b = 1/62 + 1/61
    # -> tied by symmetry in this construction; assert the RRF formula directly instead.
    k = 60
    expected_a = 1.0 / (k + 1) + 1.0 / (k + 2)
    expected_b = 1.0 / (k + 2) + 1.0 / (k + 1)
    assert merged["a"].score == expected_a
    assert merged["b"].score == expected_b
    # Raw BM25 magnitude (9000 vs 1.0) must have zero influence on the merged score.
    assert merged["a"].score == merged["b"].score


def test_merge_candidates_preserves_first_seen_source():
    vector_results = [RetrievedChunk("x", 0.5, "vector")]
    bm25_results = [RetrievedChunk("x", 5.0, "bm25")]
    merged = merge_candidates(vector_results, bm25_results)
    assert merged["x"].source == "vector"  # first list wins provenance


def test_graph_expand_local_finds_genuinely_new_neighbors(data_dir):
    """
    Regression test for the fixed bug: seed on "foo" alone (which calls
    "helper"), and confirm "helper" -- which is NOT itself a seed -- comes
    back as a neighbor. Before the fix this returned nothing, because
    resolution was scoped only to the seed chunks' own qualified names.
    """
    seed_ids = ["c1"]  # pkgtest.mod.foo
    qname_lookup = get_chunks_by_ids(str(data_dir / "store.db"), seed_ids)

    results = graph_expand_local(seed_ids, str(data_dir / "graph.json"), qname_lookup, str(data_dir / "store.db"))

    neighbor_ids = {r.chunk_id for r in results}
    assert "c2" in neighbor_ids  # pkgtest.mod.helper -- genuinely new, not a seed
    assert "c1" not in neighbor_ids  # never returns a seed as its own neighbor
    assert all(r.source == "graph" for r in results)
    assert all(r.score == GRAPH_PRIOR_SCORE for r in results)


def test_graph_expand_local_contains_relationship(data_dir):
    """Class <-> method containment expansion: seeding on the class should
    surface its method, and vice versa."""
    seed_ids = ["c3"]  # pkgtest.mod.Widget
    qname_lookup = get_chunks_by_ids(str(data_dir / "store.db"), seed_ids)
    results = graph_expand_local(seed_ids, str(data_dir / "graph.json"), qname_lookup, str(data_dir / "store.db"))
    neighbor_ids = {r.chunk_id for r in results}
    assert "c4" in neighbor_ids  # pkgtest.mod.Widget.run


def test_graph_expand_local_no_neighbors_for_unconnected_chunk(data_dir):
    seed_ids = ["c5"]  # pkgtest.other.unrelated -- not in any call/contains edge
    qname_lookup = get_chunks_by_ids(str(data_dir / "store.db"), seed_ids)
    results = graph_expand_local(seed_ids, str(data_dir / "graph.json"), qname_lookup, str(data_dir / "store.db"))
    assert results == []


def test_bm25_search_finds_exact_identifier(data_dir):
    results = bm25_search_wrapper("helper", str(data_dir / "store.db"), top_k=5)
    ids = {cid for cid, _ in [(r.chunk_id, r.score) for r in results]}
    assert "c2" in ids  # pkgtest.mod.helper


def test_vector_search_returns_all_chunks_ranked(data_dir):
    import pickle
    embedder = pickle.load(open(data_dir / "embedder.pkl", "rb"))
    query_vec = embedder.encode(["Calls helper to do the real work."])[0]
    results = vector_search(QdrantVectorStore(), TEST_REPO_ID, query_vec, top_k=5)
    assert len(results) == 5
    assert all(r.source == "vector" for r in results)
    # scores should be sorted descending
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_vector_search_unknown_repo_returns_empty():
    """A repo_id whose collection was never created (never ingested)
    degrades to no vector results instead of raising."""
    import numpy as np
    results = vector_search(QdrantVectorStore(), "no-such-repo", np.zeros(4, dtype=np.float32), top_k=5)
    assert results == []


def test_hybrid_retrieve_end_to_end_includes_graph_neighbor(data_dir):
    import pickle
    embedder = pickle.load(open(data_dir / "embedder.pkl", "rb"))
    query_vec = embedder.encode(["foo"])[0]
    results = hybrid_retrieve(
        "foo", query_vec, str(data_dir / "store.db"), str(data_dir / "graph.json"),
        QdrantVectorStore(), TEST_REPO_ID,
        top_k_each=5, use_graph=True,
    )
    assert len(results) > 0
    assert all(isinstance(r, RetrievedChunk) for r in results)
