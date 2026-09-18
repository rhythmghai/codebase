"""
Central place for env-driven settings. Reading these here once, with a
documented default and meaning, replaces the pattern of scattering
os.environ.get() calls (with duplicated/undocumented defaults) across
api/main.py, storage/neo4j_client.py, orchestration/llm_client.py, etc.
"""

import os
from pathlib import Path


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


PROJECT_ROOT = Path(__file__).parent.parent

# Where SQLite/graph/embedder files for the default repo, plus the
# persistent ingest-jobs table, live. Override to point at a mounted
# persistent volume in production -- the default (repo-local ./data) does
# NOT survive a redeploy on most PaaS targets (Railway included) unless a
# volume is attached at this path.
DATA_DIR = Path(os.environ.get("DATA_DIR", str(PROJECT_ROOT / "data")))

# Scratch space for cloning + indexing repos submitted via POST /ingest.
# Same persistence caveat as DATA_DIR -- if this isn't on a persistent
# volume, every live-ingested repo (anything other than the one baked into
# the image via ingestion/run_ingestion.py) is lost on restart.
WORKSPACE_ROOT = os.environ.get("WORKSPACE_ROOT", "/tmp/codebase-rag-workspaces")

# Comma-separated API keys accepted via the X-API-Key header. Empty means
# auth is DISABLED -- every request is treated as an anonymous caller. This
# is only acceptable for local development; a warning is logged at startup
# if the service comes up this way.
API_KEYS = set(_split_csv(os.environ.get("API_KEYS")))

# Comma-separated list of allowed CORS origins. Empty/unset falls back to
# "*" (wide open) with a startup warning, same reasoning as API_KEYS above.
ALLOWED_ORIGINS = _split_csv(os.environ.get("ALLOWED_ORIGINS")) or ["*"]

QUERY_RATE_LIMIT_PER_MIN = int(os.environ.get("QUERY_RATE_LIMIT_PER_MIN", "15"))
INGEST_RATE_LIMIT_PER_MIN = int(os.environ.get("INGEST_RATE_LIMIT_PER_MIN", "3"))

# LRU cap on how many repos' pipelines (each holding a full in-memory
# embedding matrix) are kept live at once. Ingesting past this evicts the
# least-recently-queried repo -- callers can re-ingest it to bring it back.
MAX_ACTIVE_REPOS = int(os.environ.get("MAX_ACTIVE_REPOS", "5"))

# Wall-clock budget for a single LLM call (rewrite or generate) before it's
# treated as failed and the pipeline falls back to the deterministic
# RuleBasedLLM rather than hanging the request.
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "20"))

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
