"""
Local dev storage: SQLite + FTS5, mirroring the Postgres+pgvector schema
in schema_postgres.sql. Vectors are stored as raw bytes (numpy float32) and
compared in Python since SQLite has no native vector index — fine at this
repo's scale (~450 chunks); Postgres+ivfflat is the real answer at scale.

Retrieval code (retrieval/hybrid_search.py) only depends on the three
functions below, so swapping this file for a real Postgres client later
doesn't touch anything upstream.
"""

import sqlite3
import json
import numpy as np
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id       TEXT PRIMARY KEY,
    kind           TEXT NOT NULL,
    name           TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    file_path      TEXT NOT NULL,
    start_line     INTEGER,
    end_line       INTEGER,
    source         TEXT,
    docstring      TEXT,
    signature      TEXT,
    parent_class   TEXT,
    embedding      BLOB
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    name,
    docstring,
    signature,
    source
);
"""

# Natural-language queries carry stopwords ("how", "does", "the") that will
# never appear in code identifiers/docstrings, and FTS5's default MATCH
# semantics AND all terms together -- so a single non-matching stopword
# zeroes out the whole result set. Strip stopwords and OR the remaining
# terms so BM25 ranks by relevance instead of requiring every term to hit.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "how", "does", "do", "did",
    "and", "or", "to", "of", "in", "on", "for", "it", "this", "that", "what",
    "when", "where", "why", "which", "with", "from", "by", "as", "at", "be",
}


def get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def load_chunks(chunks_jsonl: str) -> list[dict]:
    with open(chunks_jsonl) as f:
        return [json.loads(line) for line in f]


def build_store(db_path: str, chunks: list[dict], embeddings: np.ndarray):
    conn = get_conn(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks")
    cur.execute("DELETE FROM chunks_fts")

    for c, vec in zip(chunks, embeddings):
        cur.execute(
            """INSERT INTO chunks
               (chunk_id, kind, name, qualified_name, file_path, start_line,
                end_line, source, docstring, signature, parent_class, embedding)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                c["chunk_id"], c["kind"], c["name"], c["qualified_name"],
                c["file_path"], c["start_line"], c["end_line"], c["source"],
                c["docstring"], c["signature"], c["parent_class"],
                vec.astype(np.float32).tobytes(),
            ),
        )
        cur.execute(
            """INSERT INTO chunks_fts (chunk_id, name, docstring, signature, source)
               VALUES (?,?,?,?,?)""",
            (c["chunk_id"], c["name"], c["docstring"], c["signature"], c["source"]),
        )

    conn.commit()
    conn.close()


def load_all_embeddings(db_path: str) -> tuple[list[str], np.ndarray]:
    """Load all (chunk_id, embedding) pairs into memory for cosine search."""
    conn = get_conn(db_path)
    rows = conn.execute("SELECT chunk_id, embedding FROM chunks").fetchall()
    conn.close()
    ids = [r["chunk_id"] for r in rows]
    vecs = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    return ids, vecs


def get_chunks_by_ids(db_path: str, chunk_ids: list[str]) -> dict[str, dict]:
    conn = get_conn(db_path)
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids
    ).fetchall()
    conn.close()
    return {r["chunk_id"]: dict(r) for r in rows}


def bm25_search(db_path: str, query: str, top_k: int = 20) -> list[tuple[str, float]]:
    """Returns [(chunk_id, bm25_rank_score), ...] — lower fts5 rank = more relevant."""
    conn = get_conn(db_path)
    terms = [t for t in query.lower().split() if (t.isalnum() or "_" in t) and t not in _STOPWORDS]
    if not terms:
        return []
    match_expr = " OR ".join(terms)
    try:
        rows = conn.execute(
            """SELECT chunk_id, bm25(chunks_fts) as score FROM chunks_fts
               WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?""",
            (match_expr, top_k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    # fts5 bm25() returns negative-is-better; flip sign so higher = better, matching cosine convention
    return [(r["chunk_id"], -r["score"]) for r in rows]
