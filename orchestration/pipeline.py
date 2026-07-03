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
     generate
        |
  self_check        (is the answer actually grounded in retrieved context?)
        |
       END
"""

import sys
import pickle
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END

from retrieval.hybrid_search import hybrid_retrieve, RetrievedChunk
from retrieval.reranker import get_reranker, RankedChunk
from orchestration.llm_client import get_llm


class PipelineState(TypedDict, total=False):
    query: str
    rewrite: dict
    candidates: list
    ranked: list
    context: str
    answer: str
    grounded: bool
    ungrounded_warning: str


class CodebaseRAGPipeline:
    def __init__(self, db_path: str, graph_path: str, embedder_path: str,
                 reranker_backend: str = "cross-encoder", llm_backend: str | None = None):
        self.db_path = db_path
        self.graph_path = graph_path
        self.embedder = pickle.load(open(embedder_path, "rb"))
        self.reranker = get_reranker(reranker_backend)
        self.llm = get_llm(llm_backend)
        self.graph = self._build_graph()

    # ---- nodes ----

    def _node_rewrite(self, state: PipelineState) -> PipelineState:
        rewrite = self.llm.rewrite_query(state["query"])
        return {**state, "rewrite": rewrite}

    def _node_retrieve(self, state: PipelineState) -> PipelineState:
        rewrite = state["rewrite"]
        semantic_query = rewrite.get("semantic", state["query"])
        lexical_query = rewrite.get("lexical", state["query"])

        query_vec = self.embedder.encode([semantic_query])[0]
        candidates = hybrid_retrieve(
            lexical_query, query_vec, self.db_path, self.graph_path,
            top_k_each=15, use_graph=True,
        )
        return {**state, "candidates": candidates}

    def _node_rerank(self, state: PipelineState) -> PipelineState:
        ranked = self.reranker.rerank(state["query"], state["candidates"], self.db_path, top_k=8)
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

    def _node_generate(self, state: PipelineState) -> PipelineState:
        answer = self.llm.generate_answer(state["query"], state["context"])
        return {**state, "answer": answer}

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
        return {**state, "grounded": grounded, "ungrounded_warning": warning}

    def _build_graph(self):
        g = StateGraph(PipelineState)
        g.add_node("rewrite", self._node_rewrite)
        g.add_node("retrieve", self._node_retrieve)
        g.add_node("rerank", self._node_rerank)
        g.add_node("assemble", self._node_assemble)
        g.add_node("generate", self._node_generate)
        g.add_node("self_check", self._node_self_check)

        g.set_entry_point("rewrite")
        g.add_edge("rewrite", "retrieve")
        g.add_edge("retrieve", "rerank")
        g.add_edge("rerank", "assemble")
        g.add_edge("assemble", "generate")
        g.add_edge("generate", "self_check")
        g.add_edge("self_check", END)

        return g.compile()

    def run(self, query: str) -> PipelineState:
        return self.graph.invoke({"query": query})


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    pipeline = CodebaseRAGPipeline(
        db_path=str(project_root / "data" / "store.db"),
        graph_path=str(project_root / "data" / "graph.json"),
        embedder_path=str(project_root / "data" / "embedder.pkl"),
    )
    result = pipeline.run("How does FastAPI match an incoming request to a route?")
    print("REWRITE:", result["rewrite"])
    print("\nTOP CHUNKS:")
    for r in result["ranked"][:5]:
        print(" -", r.chunk["qualified_name"])
    print("\nGROUNDED:", result["grounded"], result["ungrounded_warning"])
    print("\nANSWER:\n", result["answer"][:1000])
