"""
Tests for orchestration/pipeline.py -- in particular the two things the
original review flagged as missing: (1) any retry/fallback behavior when
the LLM backend fails, and (2) any reflection loop when self_check finds
an answer ungrounded. Both are exercised directly against the pipeline's
node methods rather than relying on an emergent failure from RuleBasedLLM
(which, by construction, always echoes context verbatim and would almost
always look "grounded" -- not a reliable way to trigger the ungrounded
path deterministically).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.pipeline import CodebaseRAGPipeline, MAX_REFLECT_ATTEMPTS
from retrieval.reranker import RankedChunk
from tests.conftest import TEST_REPO_ID


@pytest.fixture
def pipeline(data_dir):
    return CodebaseRAGPipeline(
        db_path=str(data_dir / "store.db"),
        graph_path=str(data_dir / "graph.json"),
        embedder_path=str(data_dir / "embedder.pkl"),
        repo_id=TEST_REPO_ID,  # must match data_dir fixture's Qdrant collection
        use_neo4j=False,  # keep the test fast/offline; graph fallback is covered in test_hybrid_search.py
    )


def _ranked_chunk(chunk_id: str, qualified_name: str, file_path: str) -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id, rerank_score=1.0, retrieval_score=1.0, source="vector",
        chunk={"chunk_id": chunk_id, "qualified_name": qualified_name, "file_path": file_path},
    )


def test_pipeline_run_end_to_end_returns_expected_shape(pipeline):
    result = pipeline.run("What does foo call?")
    assert "answer" in result and result["answer"]
    assert "ranked" in result
    assert "grounded" in result
    assert isinstance(result["grounded"], bool)


def test_rewrite_falls_back_to_rule_based_on_llm_failure(pipeline, monkeypatch):
    def boom(query):
        raise RuntimeError("simulated transient LLM failure")

    monkeypatch.setattr(pipeline.llm, "rewrite_query", boom)
    state = pipeline._node_rewrite({"query": "how does foo work"})
    # Falls back to RuleBasedLLM's deterministic rewrite rather than raising.
    assert "rewrite" in state
    assert state["rewrite"]["semantic"] == "how does foo work"


def test_generate_falls_back_to_rule_based_on_llm_failure(pipeline, monkeypatch):
    def boom(query, context, strict=False):
        raise RuntimeError("simulated transient LLM failure")

    monkeypatch.setattr(pipeline.llm, "generate_answer", boom)
    state = pipeline._node_generate({"query": "q", "context": "some context"})
    assert "[RuleBasedLLM stand-in" in state["answer"]


def test_self_check_grounded_when_answer_references_retrieved_chunk(pipeline):
    state = {
        "answer": "The function foo calls helper() to compute the result.",
        "ranked": [_ranked_chunk("c1", "pkgtest.mod.foo", "pkgtest/mod.py")],
    }
    result = pipeline._node_self_check(state)
    assert result["grounded"] is True
    assert result["ungrounded_warning"] == ""


def test_self_check_ungrounded_when_answer_is_generic(pipeline):
    state = {
        "answer": "I'm not sure, but it probably does something with data.",
        "ranked": [_ranked_chunk("c1", "pkgtest.mod.foo", "pkgtest/mod.py")],
    }
    result = pipeline._node_self_check(state)
    assert result["grounded"] is False
    assert result["ungrounded_warning"] != ""


def test_route_after_self_check_ends_when_grounded(pipeline):
    assert pipeline._route_after_self_check({"grounded": True, "reflect_attempts": 0}) == "end"


def test_route_after_self_check_reflects_once_when_ungrounded(pipeline):
    assert pipeline._route_after_self_check({"grounded": False, "reflect_attempts": 0}) == "reflect"


def test_route_after_self_check_bounded_does_not_loop_forever(pipeline):
    """Even if the answer is still ungrounded after the retry, routing must
    terminate rather than looping indefinitely -- this is what makes it a
    bounded reflection step, not an open-ended agent loop."""
    assert pipeline._route_after_self_check({"grounded": False, "reflect_attempts": MAX_REFLECT_ATTEMPTS}) == "end"


def test_reflect_node_calls_generate_with_strict_flag_and_increments_attempts(pipeline, monkeypatch):
    captured = {}

    def fake_generate(query, context, strict=False):
        captured["strict"] = strict
        return "a stricter answer"

    monkeypatch.setattr(pipeline.llm, "generate_answer", fake_generate)
    state = pipeline._node_reflect({"query": "q", "context": "ctx", "reflect_attempts": 0})

    assert captured["strict"] is True
    assert state["reflect_attempts"] == 1
    assert state["answer"] == "a stricter answer"
