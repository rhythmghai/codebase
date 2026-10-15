"""
Orchestration layer. Deliberately a fixed-edge graph, not an agent deciding
what to do next at each step -- the task decomposes cleanly into a DAG
(rewrite -> retrieve -> rerank -> assemble -> generate -> self-check), so
there's no reason to pay for an LLM to re-derive that structure on every
query. Every node's input/output is a typed field on one shared state dict.

  rewrite_query
        |
  hybrid_retrieve  (vector + BM25 + graph, all deterministic code)
        |
     rerank
        |
  assemble_context
        |
     generate <------.
        |             |
  self_check ---------'   (ungrounded and not yet retried -> reflect -> regenerate once)
        |
       END

Error handling: rewrite_query and generate_answer are the only two nodes
that call out to a real LLM (everything else is deterministic local code),
so they're the only nodes that can fail on something outside this process's
control (timeout, rate limit, transient API error). Both are wrapped so a
failure there degrades to the deterministic RuleBasedLLM rather than
failing the whole request -- see _node_rewrite/_node_generate below.
"""

import sys
import time
import pickle
import logging
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END

from retrieval.hybrid_search import hybrid_retrieve
from retrieval.reranker import get_reranker
from orchestration.llm_client import get_llm, RuleBasedLLM
from storage.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)

# Reflection is capped at one retry -- enough to give a stricter prompt a
# real second chance without doubling worst-case LLM cost/latency on every
# single query, most of which are already grounded on the first pass.
MAX_REFLECT_ATTEMPTS = 1


class PipelineState(TypedDict, total=False):
    query: str
    rewrite: dict
    candidates: list
    ranked: list
    context: str
    answer: str
    grounded: bool
    ungrounded_warning: str
    reflect_attempts: int


class CodebaseRAGPipeline:
    def __init__(self, db_path: str, graph_path: str, embedder_path: str, repo_id: str,
                 reranker_backend: str = "lexical", llm_backend: str | None = None,
                 use_neo4j: bool = True):
        self.db_path = db_path
        self.graph_path = graph_path
        self.embedder = pickle.load(open(embedder_path, "rb"))
        self.reranker = get_reranker(reranker_backend)
        self.llm = get_llm(llm_backend)
        # Deterministic fallback used when self.llm raises after retries are
        # exhausted (see _node_rewrite/_node_generate) -- keeps a transient
        # upstream LLM outage a degraded-but-answered request instead of a
        # hard 500.
        self._fallback_llm = RuleBasedLLM()
        self.repo_id = repo_id

        # Vector search (Qdrant) is not optional the way Neo4j is: its
        # embedded mode needs zero external setup (no account, no server),
        # so there's no "not configured" case to fall back from -- every
        # pipeline gets one. repo_id doubles as the Qdrant collection name
        # (storage/vector_store.py), which is why it's now a required
        # constructor argument instead of Optional.
        self.vector_store = QdrantVectorStore()

        # Neo4j is optional per-instance, not a hard dependency: if it's not
        # configured or the connection fails, retrieval silently falls back
        # to the local JSON graph (see hybrid_search.graph_expand's
        # dispatcher) rather than crashing the whole pipeline. A query
        # pipeline shouldn't go down because an enhancement layer is
        # unavailable.
        self.neo4j_store = None
        if use_neo4j:
            try:
                from storage.neo4j_client import get_neo4j_store
                self.neo4j_store = get_neo4j_store()
            except ImportError:
                # The `neo4j` package itself isn't installed -- a mundane,
                # expected case in an environment that never installed the
                # full requirements.txt (e.g. this sandbox), not a failure
                # worth a WARNING-level traceback.
                logger.info("neo4j package not installed; repo_id=%s will use local JSON graph", repo_id)
            except Exception:
                logger.warning("Neo4j store unavailable for repo_id=%s; using local JSON graph", repo_id, exc_info=True)

        self.graph = self._build_graph()
        logger.info(
            "Pipeline initialized repo_id=%s reranker=%s neo4j=%s",
            repo_id, reranker_backend, self.neo4j_store is not None,
        )

    # ---- nodes ----

    def _node_rewrite(self, state: PipelineState) -> PipelineState:
        t0 = time.monotonic()
        try:
            rewrite = self.llm.rewrite_query(state["query"])
        except Exception:
            logger.exception("rewrite_query failed on primary LLM backend; falling back to RuleBasedLLM")
            rewrite = self._fallback_llm.rewrite_query(state["query"])
        logger.debug("node=rewrite duration_ms=%.1f", (time.monotonic() - t0) * 1000)
        return {**state, "rewrite": rewrite}

    def _node_retrieve(self, state: PipelineState) -> PipelineState:
        t0 = time.monotonic()
        rewrite = state["rewrite"]
        semantic_query = rewrite.get("semantic", state["query"])
        lexical_query = rewrite.get("lexical", state["query"])

        query_vec = self.embedder.encode([semantic_query])[0]
        candidates = hybrid_retrieve(
            lexical_query, query_vec, self.db_path, self.graph_path,
            self.vector_store, self.repo_id,
            top_k_each=15, use_graph=True,
            neo4j_store=self.neo4j_store,
        )
        logger.debug("node=retrieve duration_ms=%.1f candidates=%d", (time.monotonic() - t0) * 1000, len(candidates))
        return {**state, "candidates": candidates}

    def _node_rerank(self, state: PipelineState) -> PipelineState:
        t0 = time.monotonic()
        ranked = self.reranker.rerank(state["query"], state["candidates"], self.db_path, top_k=8)
        logger.debug("node=rerank duration_ms=%.1f ranked=%d", (time.monotonic() - t0) * 1000, len(ranked))
        return {**state, "ranked": ranked}

    def _node_assemble(self, state: PipelineState) -> PipelineState:
        parts = []
        for r in state["ranked"]:
            c = r.chunk
            parts.append(
                f"### {c['qualified_name']}  ({c['file_path']}:{c['start_line']}-{c['end_line']})\n"
                f"Signature: {c['signature']}\n"
                f"{('Docstring: ' + c['docstring']) if c['docstring'] else ''}\n"
                f"```python\n{c['source'][:600]}\n```\n"
            )
        return {**state, "context": "\n".join(parts)}

    def _generate(self, state: PipelineState, strict: bool) -> str:
        try:
            return self.llm.generate_answer(state["query"], state["context"], strict=strict)
        except Exception:
            logger.exception("generate_answer failed on primary LLM backend; falling back to RuleBasedLLM")
            return self._fallback_llm.generate_answer(state["query"], state["context"], strict=strict)

    def _node_generate(self, state: PipelineState) -> PipelineState:
        t0 = time.monotonic()
        answer = self._generate(state, strict=False)
        logger.debug("node=generate duration_ms=%.1f", (time.monotonic() - t0) * 1000)
        return {**state, "answer": answer}

    def _node_reflect(self, state: PipelineState) -> PipelineState:
        # Only reached when self_check found the first answer ungrounded.
        # Re-generates once with an explicit instruction to only state
        # claims backed by the retrieved context -- a real (if bounded)
        # self-correction step, not just a warning label on an answer
        # nobody tried to fix.
        attempts = state.get("reflect_attempts", 0) + 1
        t0 = time.monotonic()
        answer = self._generate(state, strict=True)
        logger.info("node=reflect attempt=%d duration_ms=%.1f", attempts, (time.monotonic() - t0) * 1000)
        return {**state, "answer": answer, "reflect_attempts": attempts}

    def _node_self_check(self, state: PipelineState) -> PipelineState:
        # cheap deterministic grounding check: does the answer reference at
        # least one qualified name / file path that was actually retrieved?
        # (a real deployment could add an LLM-judge call here; kept
        # deterministic to avoid a second LLM call on every request)
        referenced_names = {r.chunk["qualified_name"].split(".")[-1] for r in state["ranked"]}
        referenced_files = {r.chunk["file_path"] for r in state["ranked"]}
        answer_lower = state["answer"].lower()
        grounded = any(n.lower() in answer_lower for n in referenced_names) or \
                   any(f.lower() in answer_lower for f in referenced_files)
        warning = "" if grounded else "Answer may not be fully grounded in retrieved context -- verify against source."
        if not grounded:
            logger.info("self_check: answer not grounded (reflect_attempts=%d)", state.get("reflect_attempts", 0))
        return {**state, "grounded": grounded, "ungrounded_warning": warning}

    def _route_after_self_check(self, state: PipelineState) -> str:
        if state["grounded"] or state.get("reflect_attempts", 0) >= MAX_REFLECT_ATTEMPTS:
            return "end"
        return "reflect"

    def _build_graph(self):
        g = StateGraph(PipelineState)
        g.add_node("rewrite", self._node_rewrite)
        g.add_node("retrieve", self._node_retrieve)
        g.add_node("rerank", self._node_rerank)
        g.add_node("assemble", self._node_assemble)
        g.add_node("generate", self._node_generate)
        g.add_node("self_check", self._node_self_check)
        g.add_node("reflect", self._node_reflect)

        g.set_entry_point("rewrite")
        g.add_edge("rewrite", "retrieve")
        g.add_edge("retrieve", "rerank")
        g.add_edge("rerank", "assemble")
        g.add_edge("assemble", "generate")
        g.add_edge("generate", "self_check")
        g.add_conditional_edges("self_check", self._route_after_self_check, {"reflect": "reflect", "end": END})
        g.add_edge("reflect", "self_check")

        return g.compile()

    def run(self, query: str) -> PipelineState:
        return self.graph.invoke({"query": query})


if __name__ == "__main__":
    pipeline = CodebaseRAGPipeline(
        db_path="/home/claude/codebase-rag/data/store.db",
        graph_path="/home/claude/codebase-rag/data/graph.json",
        embedder_path="/home/claude/codebase-rag/data/embedder.pkl",
        repo_id="fastapi_test",
    )
    result = pipeline.run("How does FastAPI match an incoming request to a route?")
    print("REWRITE:", result["rewrite"])
    print("\nTOP CHUNKS:")
    for r in result["ranked"][:5]:
        print(" -", r.chunk["qualified_name"])
    print("\nGROUNDED:", result["grounded"], result["ungrounded_warning"])
    print("\nANSWER:\n", result["answer"][:1000])