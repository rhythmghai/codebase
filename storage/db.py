"""
Local dev storage: SQLite + FTS5 for chunk metadata and lexical (BM25)
search. Embeddings no longer live here -- they're stored and searched via
a real vector index (storage/vector_store.py, Qdrant) instead of the raw
BLOB + brute-force numpy scan this file used to do. This file now owns
exactly two things: chunk metadata (for citations/context assembly) and
full-text search.

Retrieval code (retrieval/hybrid_search.py) only depends on the functions
below, so swapping this file for a real Postgres client later doesn't
touch anything upstream.
"""

import sqlite3
import threading
import json


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
    parent_class   TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    name,
    docstring,
    signature,
    source
);

-- qualified_name is the lookup key for get_all_qname_to_id/get_ids_by_qnames,
-- which graph expansion calls on every query -- without this index those are
-- full table scans. Free at this repo's scale (~450 rows), real at real scale.
CREATE INDEX IF NOT EXISTS idx_chunks_qualified_name ON chunks(qualified_name);
"""

# get_conn() used to run `executescript(SCHEMA)` (a DDL statement, which
# implicitly commits and takes a brief write lock) on every single call --
# i.e. on every retrieval helper, every query. Tracking which db_path has
# already been initialized in this process turns that into a one-time cost.
_initialized_dbs: set[str] = set()
_init_lock = threading.Lock()

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
    # timeout=30: SQLite's busy-timeout, so a writer holding the DB briefly
    # (e.g. build_store's delete-then-reinsert during ingestion) makes
    # concurrent readers wait up to 30s instead of raising
    # "database is locked" immediately.
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    with _init_lock:
        if db_path not in _initialized_dbs:
            conn.executescript(SCHEMA)
            # WAL lets readers proceed concurrently with a writer instead of
            # blocking on the single rollback-journal lock every write used
            # to take under the default journal mode.
            conn.execute("PRAGMA journal_mode=WAL")
            _initialized_dbs.add(db_path)
    return conn


def load_chunks(chunks_jsonl: str) -> list[dict]:
    with open(chunks_jsonl) as f:
        return [json.loads(line) for line in f]


def build_store(db_path: str, chunks: list[dict]):
    """Chunk metadata + FTS index only. Vector storage is a separate call
    to storage/vector_store.py's QdrantVectorStore.rebuild() -- ingestion
    callers do both, not just this one."""
    conn = get_conn(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks")
    cur.execute("DELETE FROM chunks_fts")

    for c in chunks:
        cur.execute(
            """INSERT INTO chunks
               (chunk_id, kind, name, qualified_name, file_path, start_line,
                end_line, source, docstring, signature, parent_class)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                c["chunk_id"], c["kind"], c["name"], c["qualified_name"],
                c["file_path"], c["start_line"], c["end_line"], c["source"],
                c["docstring"], c["signature"], c["parent_class"],
            ),
        )
        cur.execute(
            """INSERT INTO chunks_fts (chunk_id, name, docstring, signature, source)
               VALUES (?,?,?,?,?)""",
            (c["chunk_id"], c["name"], c["docstring"], c["signature"], c["source"]),
        )

    conn.commit()
    conn.close()


def get_chunks_by_ids(db_path: str, chunk_ids: list[str]) -> dict[str, dict]:
    conn = get_conn(db_path)
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids
    ).fetchall()
    conn.close()
    return {r["chunk_id"]: dict(r) for r in rows}


def get_all_qname_to_id(db_path: str) -> dict[str, str]:
    """Full-corpus qualified_name -> chunk_id mapping (not scoped to any
    particular seed set). Needed by graph_expand_local: resolving a
    genuinely new neighbor to a real chunk_id requires knowing about chunks
    beyond whatever small seed batch triggered the expansion."""
    conn = get_conn(db_path)
    rows = conn.execute("SELECT qualified_name, chunk_id FROM chunks").fetchall()
    conn.close()
    return {r["qualified_name"]: r["chunk_id"] for r in rows}


def get_ids_by_qnames(db_path: str, qualified_names: list[str]) -> dict[str, str]:
    """Reverse lookup: qualified_name -> chunk_id. Needed because Neo4j
    stores/returns qualified_names, but retrieval elsewhere works in terms
    of chunk_id (the SQLite/Postgres primary key)."""
    if not qualified_names:
        return {}
    conn = get_conn(db_path)
    placeholders = ",".join("?" * len(qualified_names))
    rows = conn.execute(
        f"SELECT qualified_name, chunk_id FROM chunks WHERE qualified_name IN ({placeholders})",
        qualified_names,
    ).fetchall()
    conn.close()
    return {r["qualified_name"]: r["chunk_id"] for r in rows}


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