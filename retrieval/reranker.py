"""
Reranking stage: takes the merged hybrid candidate set (up to ~30 chunks)
and re-scores each against the raw query with a finer-grained signal than
the retrieval stage used. This is the classic recall/precision tradeoff:
retrieval is cheap and broad, rerank is more expensive and applied only to
a small top-k.

RealReranker is a real cross-encoder (sentence-transformers CrossEncoder,
e.g. ms-marco-MiniLM-L-6-v2) -- use this outside the sandbox / in deployment.

LexicalReranker is the sandbox-friendly stand-in: token-overlap + field-weighted
scoring (qualified_name and signature matches weighted higher than body text,
since an exact identifier match is a much stronger relevance signal than an
incidental word match somewhere in a function body). It's deliberately simple
and deterministic so the ablation ("with rerank vs without") still measures
something real, even without a neural model.
"""

import sys
import re
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.db import get_chunks_by_ids
from retrieval.hybrid_search import RetrievedChunk


@dataclass
class RankedChunk:
    chunk_id: str
    rerank_score: float
    retrieval_score: float
    source: str
    chunk: dict


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{1,}", text.lower()))


class LexicalReranker:
    FIELD_WEIGHTS = {"qualified_name": 3.0, "signature": 2.0, "docstring": 1.5, "source": 0.5}

    def rerank(self, query: str, candidates: list[RetrievedChunk], db_path: str, top_k: int = 8) -> list[RankedChunk]:
        query_tokens = _tokenize(query)
        chunk_ids = [c.chunk_id for c in candidates]
        chunk_lookup = get_chunks_by_ids(db_path, chunk_ids)

        ranked = []
        for cand in candidates:
            chunk = chunk_lookup.get(cand.chunk_id)
            if not chunk:
                continue
            score = 0.0
            for field, weight in self.FIELD_WEIGHTS.items():
                field_tokens = _tokenize(chunk.get(field) or "")
                overlap = len(query_tokens & field_tokens)
                score += weight * overlap
            # normalize by query length so longer queries don't trivially win
            score = score / max(len(query_tokens), 1)
            ranked.append(RankedChunk(cand.chunk_id, score, cand.score, cand.source, chunk))

        ranked.sort(key=lambda r: -r.rerank_score)
        return ranked[:top_k]


class CrossEncoderReranker:
    """Real neural reranker — use outside the sandbox."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[RetrievedChunk], db_path: str, top_k: int = 8) -> list[RankedChunk]:
        chunk_ids = [c.chunk_id for c in candidates]
        chunk_lookup = get_chunks_by_ids(db_path, chunk_ids)
        pairs, valid_cands = [], []
        for cand in candidates:
            chunk = chunk_lookup.get(cand.chunk_id)
            if not chunk:
                continue
            doc_text = f"{chunk['signature']}\n{chunk['docstring']}\n{chunk['source'][:500]}"
            pairs.append((query, doc_text))
            valid_cands.append((cand, chunk))

        scores = self.model.predict(pairs)
        ranked = [
            RankedChunk(c.chunk_id, float(s), c.score, c.source, chunk)
            for (c, chunk), s in zip(valid_cands, scores)
        ]
        ranked.sort(key=lambda r: -r.rerank_score)
        return ranked[:top_k]


def get_reranker(backend: str = "lexical"):
    if backend == "lexical":
        return LexicalReranker()
    if backend == "cross-encoder":
        return CrossEncoderReranker()
    raise ValueError(f"Unknown reranker backend: {backend}")
