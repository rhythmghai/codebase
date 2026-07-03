"""
Eval harness: measures retrieval quality against the hand-labeled set,
runs ablations across pipeline configurations, and reports the numbers
that actually justify each architectural choice (hybrid vs vector-only,
rerank vs no-rerank, graph vs no-graph).

This is deliberately retrieval-focused, not generation-focused: per the
project's guiding principle, evals on retrieval solve most of the problem
before any prompt/model tuning is worth doing.

Metrics:
  - Recall@k       : fraction of queries where >=1 relevant chunk is in top-k
  - Precision@k    : mean fraction of top-k that are relevant
  - MRR            : mean reciprocal rank of the first relevant chunk
"""

import sys
import json
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.hybrid_search import vector_search, bm25_search_wrapper, merge_candidates, graph_expand
from retrieval.reranker import get_reranker
from storage.db import get_chunks_by_ids


_PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = str(_PROJECT_ROOT / "data" / "store.db")
GRAPH_PATH = str(_PROJECT_ROOT / "data" / "graph.json")
EMBEDDER_PATH = str(_PROJECT_ROOT / "data" / "embedder.pkl")
LABELED_PATH = str(_PROJECT_ROOT / "eval" / "labeled_queries.json")


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    top_k = set(ranked_ids[:k])
    return 1.0 if top_k & relevant else 0.0


def precision_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in relevant)
    return hits / len(top_k)


def mrr(ranked_ids: list[str], relevant: set[str]) -> float:
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def qnames_to_ids(qnames: list[str], all_chunks: list[dict]) -> set[str]:
    lookup = {c["qualified_name"]: c["chunk_id"] for c in all_chunks}
    return {lookup[q] for q in qnames if q in lookup}


def run_config(query: str, query_vec, use_bm25: bool, use_graph: bool, use_rerank: bool,
                reranker, k: int = 8) -> list[str]:
    result_lists = [vector_search(query_vec, DB_PATH, top_k=15)]
    if use_bm25:
        result_lists.append(bm25_search_wrapper(query, DB_PATH, top_k=15))

    merged = merge_candidates(*result_lists)

    if use_graph:
        seed_ids = list(merged.keys())[:8]
        chunk_lookup = get_chunks_by_ids(DB_PATH, seed_ids)
        graph_results = graph_expand(seed_ids, GRAPH_PATH, chunk_lookup)
        merged = merge_candidates(list(merged.values()), graph_results)

    ordered = sorted(merged.values(), key=lambda r: -r.score)[:20]

    if use_rerank:
        ranked = reranker.rerank(query, ordered, DB_PATH, top_k=k)
        return [r.chunk_id for r in ranked]
    else:
        return [r.chunk_id for r in ordered[:k]]


def evaluate_config(labeled: list[dict], all_chunks: list[dict], embedder, reranker,
                     use_bm25: bool, use_graph: bool, use_rerank: bool, k: int = 8) -> dict:
    recalls, precisions, mrrs = [], [], []
    for item in labeled:
        query = item["query"]
        relevant_ids = qnames_to_ids(item["relevant"], all_chunks)
        if not relevant_ids:
            continue
        query_vec = embedder.encode([query])[0]
        ranked_ids = run_config(query, query_vec, use_bm25, use_graph, use_rerank, reranker, k=k)

        recalls.append(recall_at_k(ranked_ids, relevant_ids, k))
        precisions.append(precision_at_k(ranked_ids, relevant_ids, k))
        mrrs.append(mrr(ranked_ids, relevant_ids))

    n = len(recalls)
    return {
        "recall@8": sum(recalls) / n,
        "precision@8": sum(precisions) / n,
        "mrr": sum(mrrs) / n,
        "n_queries": n,
    }


def main():
    labeled = json.load(open(LABELED_PATH))
    all_chunks = [json.loads(l) for l in open(_PROJECT_ROOT / "data" / "chunks.jsonl")]
    embedder = pickle.load(open(EMBEDDER_PATH, "rb"))
    reranker = get_reranker("lexical")

    configs = [
        ("vector-only",                 dict(use_bm25=False, use_graph=False, use_rerank=False)),
        ("vector + BM25 (hybrid)",       dict(use_bm25=True,  use_graph=False, use_rerank=False)),
        ("hybrid + rerank",              dict(use_bm25=True,  use_graph=False, use_rerank=True)),
        ("hybrid + rerank + graph",      dict(use_bm25=True,  use_graph=True,  use_rerank=True)),
    ]

    print(f"{'Config':<28} {'Recall@8':>10} {'Precision@8':>13} {'MRR':>8}  (n={len(labeled)})")
    print("-" * 65)
    results = {}
    for name, kwargs in configs:
        metrics = evaluate_config(labeled, all_chunks, embedder, reranker, **kwargs)
        results[name] = metrics
        print(f"{name:<28} {metrics['recall@8']:>10.3f} {metrics['precision@8']:>13.3f} {metrics['mrr']:>8.3f}")

    with open(_PROJECT_ROOT / "eval" / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to eval/results.json")


if __name__ == "__main__":
    main()
