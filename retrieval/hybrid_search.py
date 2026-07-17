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

from storage.db import bm25_search, load_all_embeddings, get_chunks_by_ids, get_ids_by_qnames, get_all_qname_to_id


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


def graph_expand_local(chunk_ids: list[str], graph_path: str, qname_lookup: dict, db_path: str,
                        max_neighbors: int = 3) -> list[RetrievedChunk]:
    """
    JSON-file + in-process traversal. This is the fallback path, used when
    Neo4j isn't configured -- keeps the system fully runnable offline / in
    local dev without a graph database dependency.

    BUG THAT WAS HERE, fixed: this function used to build qname_to_id only
    from qname_lookup (the seed chunks' own qnames), which meant any
    candidate resolved from `calls`/`contains` could only ever match if it
    happened to already be one of the seeds -- and the final `nid not in
    chunk_ids` check would then exclude it anyway. Net effect: this path
    could structurally never return a genuinely new neighbor; it was a
    silent no-op in production. Fixed by resolving against the full-corpus
    qname_to_id (from the database) instead of the seed-scoped one.
    """
    graph = json.load(open(graph_path))
    calls = graph["calls"]
    contains = graph["contains"]

    id_to_qname = {cid: q["qualified_name"] for cid, q in qname_lookup.items()}
    qname_to_id = get_all_qname_to_id(db_path)  # full corpus, not just seeds

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
            if callee == short_name and caller in qname_to_id:
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
            results.append(RetrievedChunk(nid, 0.001, "graph"))  # low prior weight, scaled below the RRF floor
    return results


def graph_expand_neo4j(chunk_ids: list[str], neo4j_store, repo_id: str, db_path: str,
                        qname_lookup: dict, max_neighbors: int = 3) -> list[RetrievedChunk]:
    """
    Real graph-database traversal via Cypher (see storage/neo4j_client.py),
    replacing the hand-rolled JSON dict walk. Call resolution already
    happened once at ingestion time (Neo4jGraphStore.load_graph), so this
    is just a 1-hop MATCH query -- no per-query string resolution needed.
    """
    seed_qnames = [qname_lookup[cid]["qualified_name"] for cid in chunk_ids if cid in qname_lookup]
    if not seed_qnames:
        return []

    neighbor_qnames = neo4j_store.get_neighbors(repo_id, seed_qnames, max_neighbors_per_seed=max_neighbors)
    qname_to_id = get_ids_by_qnames(db_path, neighbor_qnames)

    results = []
    for nq in neighbor_qnames:
        nid = qname_to_id.get(nq)
        if nid and nid not in chunk_ids:
            results.append(RetrievedChunk(nid, 0.001, "graph"))
    return results


def graph_expand(chunk_ids: list[str], graph_path: str, qname_lookup: dict, db_path: str, max_neighbors: int = 3,
                  neo4j_store=None, repo_id: str | None = None) -> list[RetrievedChunk]:
    """
    Dispatcher: use real Neo4j graph traversal when a store + repo_id are
    provided, otherwise fall back to the local JSON file. This is what lets
    the system run identically whether or not a graph database is
    configured -- useful for local dev, and an honest degradation path
    rather than a hard dependency.
    """
    if neo4j_store is not None and repo_id is not None:
        try:
            return graph_expand_neo4j(chunk_ids, neo4j_store, repo_id, db_path, qname_lookup, max_neighbors)
        except Exception:
            pass  # fall through to local JSON on any Neo4j error (network, auth, etc.)
    return graph_expand_local(chunk_ids, graph_path, qname_lookup, db_path, max_neighbors)


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
                     top_k_each: int = 15, use_graph: bool = True,
                     neo4j_store=None, repo_id: str | None = None) -> list[RetrievedChunk]:
    vec_results = vector_search(query_vec, db_path, top_k=top_k_each)
    bm25_results = bm25_search_wrapper(query, db_path, top_k=top_k_each)

    merged = merge_candidates(vec_results, bm25_results)

    if use_graph:
        seed_ids = list(merged.keys())[:8]  # only expand from strongest hits, keep it scoped
        chunk_lookup = get_chunks_by_ids(db_path, seed_ids)
        graph_results = graph_expand(
            seed_ids, graph_path, chunk_lookup,
            neo4j_store=neo4j_store, repo_id=repo_id, db_path=db_path,
        )
        merged = merge_candidates(list(merged.values()), graph_results)

    ordered = sorted(merged.values(), key=lambda r: -r.score)
    return ordered[:20]  # cap the pool -- retrieval score gates what reaches rerank, not an unbounded set