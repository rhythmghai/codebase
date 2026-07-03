"""
Hybrid retrieval: dense vector search + BM25 full-text + graph neighbor
expansion, run in parallel conceptually (sequential here for simplicity,
each is independent of the others) and merged before reranking.

This is the layer that answers "why hybrid instead of just vector search":
- vector search catches semantic/paraphrased queries ("how do routes get registered")
- BM25 catches exact identifiers/error strings a dev actually searched for
  ("APIRoute", "HTTPException", "response_model")
- graph expansion adds structural context vector/BM25 can't see: what calls
  this, what does this call, what class is this a method of
"""

import sys
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.db import bm25_search, load_all_embeddings, get_chunks_by_ids
def main():
    import inspect
    from retrieval.hybrid_search import merge_candidates
    assert "rrf_scores" in inspect.getsource(merge_candidates), "merge_candidates is not the RRF version — check your save!"

@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    source: str  # "vector" | "bm25" | "graph"


def vector_search(query_vec: np.ndarray, db_path: str, top_k: int = 15) -> list[RetrievedChunk]:
    ids, vecs = load_all_embeddings(db_path)
    sims = vecs @ query_vec  # vectors are pre-normalized -> dot product = cosine sim
    top_idx = np.argsort(-sims)[:top_k]
    return [RetrievedChunk(ids[i], float(sims[i]), "vector") for i in top_idx]


def bm25_search_wrapper(query: str, db_path: str, top_k: int = 15) -> list[RetrievedChunk]:
    results = bm25_search(db_path, query, top_k=top_k)
    return [RetrievedChunk(cid, score, "bm25") for cid, score in results]


def graph_expand(chunk_ids: list[str], graph_path: str, qname_lookup: dict, max_neighbors: int = 3) -> list[RetrievedChunk]:
    """
    For each retrieved chunk, pull immediate graph neighbors (calls, contains,
    called-by) and add them as low-weight candidates. This is what lets the
    system answer "what else is relevant structurally" that pure text/vector
    similarity would never surface.
    """
    graph = json.load(open(graph_path))
    calls = graph["calls"]
    contains = graph["contains"]

    id_to_qname = {cid: q["qualified_name"] for cid, q in qname_lookup.items()}
    qname_to_id = {v: k for k, v in id_to_qname.items()}

    neighbor_qnames = set()
    for cid in chunk_ids:
        qname = id_to_qname.get(cid)
        if not qname:
            continue
        # things this chunk calls
        for caller, callee in calls:
            if caller == qname:
                # callee is a short name; try to resolve within same module
                module = qname.rsplit(".", 1)[0]
                candidate = f"{module}.{callee}"
                if candidate in qname_to_id:
                    neighbor_qnames.add(candidate)
        # things that call this chunk
        short_name = qname.split(".")[-1]
        for caller, callee in calls:
            if callee == short_name and caller in qname_to_id.values():
                neighbor_qnames.add(caller)
        # class <-> method relationships
        for cls_q, method_q in contains:
            if qname in (cls_q, method_q):
                neighbor_qnames.add(cls_q)
                neighbor_qnames.add(method_q)

    neighbor_qnames.discard(None)
    results = []
    for nq in list(neighbor_qnames)[: max_neighbors * len(chunk_ids)]:
        nid = qname_to_id.get(nq)
        if nid and nid not in chunk_ids:
            results.append(RetrievedChunk(nid, 0.001, "graph"))  # fixed low prior weight
    return results


def merge_candidates(*result_lists: list[RetrievedChunk], k: int = 60) -> dict[str, RetrievedChunk]:
    """
    Merge by chunk_id using Reciprocal Rank Fusion (RRF) instead of raw
    score comparison. Vector cosine scores and BM25 rank scores live on
    incomparable scales, so comparing raw scores lets whichever channel
    happens to produce larger numbers dominate the merge. RRF uses each
    candidate's rank within its own list instead of its raw score.
    """
    rrf_scores: dict[str, float] = {}
    provenance: dict[str, RetrievedChunk] = {}

    for results in result_lists:
        for rank, r in enumerate(results, start=1):
            rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0.0) + 1.0 / (k + rank)
            if r.chunk_id not in provenance:
                provenance[r.chunk_id] = r

    merged: dict[str, RetrievedChunk] = {}
    for chunk_id, score in rrf_scores.items():
        original = provenance[chunk_id]
        merged[chunk_id] = RetrievedChunk(chunk_id=chunk_id, score=score, source=original.source)
    return merged


def hybrid_retrieve(query: str, query_vec: np.ndarray, db_path: str, graph_path: str,
                     top_k_each: int = 15, use_graph: bool = True) -> list[RetrievedChunk]:
    vec_results = vector_search(query_vec, db_path, top_k=top_k_each)
    bm25_results = bm25_search_wrapper(query, db_path, top_k=top_k_each)

    merged = merge_candidates(vec_results, bm25_results)

    if use_graph:
        seed_ids = list(merged.keys())[:8]  # only expand from strongest hits, keep it scoped
        chunk_lookup = get_chunks_by_ids(db_path, seed_ids)
        graph_results = graph_expand(seed_ids, graph_path, chunk_lookup)
        merged = merge_candidates(list(merged.values()), graph_results)

    ordered = sorted(merged.values(), key=lambda r: -r.score)
    return ordered[:20]  # cap the pool -- retrieval score now actually gates what reaches rerank
