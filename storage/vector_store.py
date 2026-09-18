"""
Qdrant-backed vector store, replacing the brute-force numpy scan that
vector_search() used to run against every embedding stored as a raw BLOB
in SQLite -- every query pulled the *entire* corpus's vectors into memory
and computed a dot product against all of them, O(n) per query, no index
at all. Fine at ~450 chunks, not a real answer at scale.

Why Qdrant, and why this shape: its embedded ("local") mode persists
straight to disk with zero server process, zero account, zero credentials
-- the same operational shape as SQLite. The same client code upgrades to
a real hosted/self-hosted Qdrant instance later by setting QDRANT_URL (+
QDRANT_API_KEY), without touching a single call site -- the same
optional-upgrade shape storage/neo4j_client.py already uses for the graph
layer.

Multi-tenancy: one shared client, one Qdrant *collection* per repo_id.
This is the direct counterpart to how Neo4j scopes multi-tenancy -- Neo4j
has exactly one shared database (AuraDB's free tier grants only one) and
filters every query by a repo_id *property*; Qdrant has no such
constraint, so each repo gets its own named *collection* instead of a
shared one filtered at query time. Same underlying problem, matched to
whichever primitive each backend actually offers cheaply.

Why exactly one shared client, never one per pipeline instance: Qdrant's
embedded mode is strictly single-process/single-client per storage path --
a second QdrantClient() opened on the same local path raises RuntimeError
immediately ("Storage folder ... is already accessed by another instance
of Qdrant client"), verified directly against this exact client version.
Memoizing one client for the whole process and using collections (not
separate paths) for isolation between repos means a repo can be
re-ingested (drop + recreate its collection) while other repos' pipelines
stay open, with no lock conflict ever possible.
"""

import os
import sys
import uuid
import logging
import threading
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.config import DATA_DIR

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None
_client_lock = threading.Lock()
_ensured_collections: set[str] = set()
_ensure_lock = threading.Lock()

_UPSERT_BATCH_SIZE = 256


def _get_client() -> QdrantClient:
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            url = os.environ.get("QDRANT_URL")
            if url:
                _client = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))
                logger.info("Qdrant: connected to hosted instance at %s", url)
            else:
                path = os.environ.get("QDRANT_PATH") or str(DATA_DIR / "qdrant")
                _client = QdrantClient(path=path)
                logger.info("Qdrant: using embedded local store at %s", path)
    return _client


def _point_id(chunk_id: str) -> str:
    """Qdrant point IDs must be an unsigned int or a UUID -- chunk_id is a
    12-char content hash, not a valid UUID, so derive a stable UUID from it
    instead. Deterministic: the same chunk_id always maps to the same
    point id, so re-upserting it overwrites cleanly rather than
    accumulating a duplicate point."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _ensure_collection(client: QdrantClient, collection_name: str, dim: int) -> None:
    if collection_name in _ensured_collections:
        return
    with _ensure_lock:
        if collection_name in _ensured_collections:
            return
        existing = {c.name for c in client.get_collections().collections}
        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            )
            logger.info("Qdrant: created collection '%s' (dim=%d)", collection_name, dim)
        _ensured_collections.add(collection_name)


class QdrantVectorStore:
    def __init__(self):
        self.client = _get_client()

    def upsert(self, repo_id: str, chunk_ids: list[str], vectors: np.ndarray) -> None:
        """Pure upsert: adds/overwrites the given points, leaves any other
        existing points in the collection untouched. Most callers want
        rebuild() instead (see below)."""
        _ensure_collection(self.client, repo_id, dim=vectors.shape[1])
        points = [
            qmodels.PointStruct(
                id=_point_id(cid),
                vector=vec.astype(np.float32).tolist(),
                payload={"chunk_id": cid},
            )
            for cid, vec in zip(chunk_ids, vectors)
        ]
        for i in range(0, len(points), _UPSERT_BATCH_SIZE):
            self.client.upsert(collection_name=repo_id, points=points[i:i + _UPSERT_BATCH_SIZE])
        logger.info("Qdrant: upserted %d vectors into collection '%s'", len(points), repo_id)

    def rebuild(self, repo_id: str, chunk_ids: list[str], vectors: np.ndarray) -> None:
        """Full delete-then-reinsert, mirroring storage/db.py's build_store
        -- drops the repo's collection (if it exists) and recreates it
        fresh, rather than leaving stale points around from a previous
        ingestion whose chunk_ids have since changed. This is what every
        ingestion path calls; upsert() is the lower-level primitive it's
        built on."""
        try:
            self.client.delete_collection(collection_name=repo_id)
        except Exception:
            pass  # collection didn't exist yet -- first-ever ingest of this repo
        _ensured_collections.discard(repo_id)
        self.upsert(repo_id, chunk_ids, vectors)

    def search(self, repo_id: str, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Returns [(chunk_id, cosine_score), ...], highest score first."""
        try:
            hits = self.client.query_points(
                collection_name=repo_id,
                query=query_vec.astype(np.float32).tolist(),
                limit=top_k,
            ).points
        except Exception:
            # Collection doesn't exist yet (this repo was never ingested
            # through the vector-store-aware path) or a transient Qdrant
            # error -- degrade to "no vector results" rather than failing
            # the whole query; BM25 + graph expansion still run.
            logger.warning("Qdrant search failed for collection='%s'", repo_id, exc_info=True)
            return []
        return [(hit.payload["chunk_id"], float(hit.score)) for hit in hits]
