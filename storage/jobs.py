"""
Ingestion job status, persisted to SQLite instead of an in-memory dict.

Why this exists: a plain in-process dict (the previous approach) loses
every in-flight and completed job the instant the process restarts -- a
client polling GET /ingest/status/{job_id} after a redeploy gets a bare 404
with no way to tell "your ingest actually finished" from "the server has
no idea what you're talking about". Persisting to the same DATA_DIR as the
rest of the service's data means job history survives a restart as long as
DATA_DIR is on persistent storage (see common/config.py's DATA_DIR docstring
for the deployment caveat that still applies).
"""

import sqlite3
import threading
import time

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_jobs (
    job_id        TEXT PRIMARY KEY,
    status        TEXT NOT NULL,
    repo_url      TEXT NOT NULL,
    repo_id       TEXT,
    num_chunks    INTEGER,
    num_files     INTEGER,
    source_subdir TEXT,
    error         TEXT,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);
"""

_lock = threading.Lock()
_initialized: set[str] = set()


def _get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    if db_path not in _initialized:
        conn.executescript(_SCHEMA)
        _initialized.add(db_path)
    return conn


def create_job(db_path: str, job_id: str, repo_url: str) -> None:
    now = time.time()
    with _lock:
        conn = _get_conn(db_path)
        try:
            conn.execute(
                "INSERT INTO ingest_jobs (job_id, status, repo_url, created_at, updated_at) "
                "VALUES (?, 'running', ?, ?, ?)",
                (job_id, repo_url, now, now),
            )
            conn.commit()
        finally:
            conn.close()


def update_job(db_path: str, job_id: str, **fields) -> None:
    if not fields:
        return
    fields["updated_at"] = time.time()
    cols = ", ".join(f"{k} = ?" for k in fields)
    with _lock:
        conn = _get_conn(db_path)
        try:
            conn.execute(f"UPDATE ingest_jobs SET {cols} WHERE job_id = ?", (*fields.values(), job_id))
            conn.commit()
        finally:
            conn.close()


def get_job(db_path: str, job_id: str) -> dict | None:
    with _lock:
        conn = _get_conn(db_path)
        try:
            row = conn.execute("SELECT * FROM ingest_jobs WHERE job_id = ?", (job_id,)).fetchone()
        finally:
            conn.close()
    return dict(row) if row else None
