"""Unit tests for retrieval/reranker.py's LexicalReranker (the default,
dependency-free backend -- CrossEncoderReranker needs sentence-transformers
model weights and isn't exercised in the default offline test run)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.hybrid_search import RetrievedChunk
from retrieval.reranker import LexicalReranker, get_reranker


def test_lexical_reranker_field_weighting(data_dir):
    """A candidate whose qualified_name matches the query should rank above
    one that only shares an incidental body-text word, since FIELD_WEIGHTS
    weighs qualified_name (3.0) well above source (0.5)."""
    candidates = [
        RetrievedChunk("c2", 0.5, "vector"),  # pkgtest.mod.helper -- qualified_name match
        RetrievedChunk("c1", 0.5, "vector"),  # pkgtest.mod.foo -- only body mentions "helper"
    ]
    reranker = LexicalReranker()
    ranked = reranker.rerank("helper", candidates, str(data_dir / "store.db"), top_k=8)

    ranked_ids = [r.chunk_id for r in ranked]
    assert ranked_ids[0] == "c2"
    assert ranked[0].rerank_score > ranked[1].rerank_score


def test_lexical_reranker_respects_top_k(data_dir):
    candidates = [RetrievedChunk(cid, 0.1, "vector") for cid in ["c1", "c2", "c3", "c4", "c5"]]
    reranker = LexicalReranker()
    ranked = reranker.rerank("helper", candidates, str(data_dir / "store.db"), top_k=2)
    assert len(ranked) == 2


def test_lexical_reranker_skips_missing_chunks(data_dir):
    candidates = [RetrievedChunk("c2", 0.1, "vector"), RetrievedChunk("does-not-exist", 0.1, "vector")]
    reranker = LexicalReranker()
    ranked = reranker.rerank("helper", candidates, str(data_dir / "store.db"), top_k=8)
    assert [r.chunk_id for r in ranked] == ["c2"]


def test_get_reranker_lexical_backend():
    assert isinstance(get_reranker("lexical"), LexicalReranker)


def test_get_reranker_unknown_backend_raises():
    import pytest
    with pytest.raises(ValueError):
        get_reranker("not-a-real-backend")
