"""
API layer. This is deliberately the most "plain engineering" file in the
project -- Pydantic schemas, auth, a per-client token-bucket rate limiter,
structured error handling, multi-repo state -- since that's what an SDE
interview actually probes, more than the retrieval internals.
"""

import time
import uuid
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config import (
    API_KEYS, ALLOWED_ORIGINS, DATA_DIR, WORKSPACE_ROOT,
    QUERY_RATE_LIMIT_PER_MIN, INGEST_RATE_LIMIT_PER_MIN, MAX_ACTIVE_REPOS, LOG_LEVEL,
)
from common.logging_config import configure_logging, request_id_var
from orchestration.pipeline import CodebaseRAGPipeline
from ingestion.ingest_from_url import ingest_from_url, slugify_repo_url
from storage import jobs as job_store

configure_logging(LOG_LEVEL)
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    repo_id: str | None = Field(
        default=None,
        description="Which ingested repo to query. Omit to use the most recently ingested repo.",
    )


class IngestRequest(BaseModel):
    repo_url: str = Field(..., min_length=8, max_length=300)


class IngestStartedResponse(BaseModel):
    job_id: str
    status: str  # "started"


class IngestStatusResponse(BaseModel):
    job_id: str
    status: str  # "running" | "done" | "error"
    repo_url: str
    repo_id: str | None = None
    num_chunks: int | None = None
    num_files: int | None = None
    source_subdir: str | None = None
    error: str | None = None
    stale: bool = False  # status=="running" but old enough a restart likely lost the background task


class RetrievedChunkResponse(BaseModel):
    qualified_name: str
    file_path: str
    start_line: int
    end_line: int
    source: str  # "vector" | "bm25" | "graph"


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    warning: str = ""
    sources: list[RetrievedChunkResponse]
    repo_id: str = ""
    active_repo: str = ""


class RepoInfo(BaseModel):
    repo_id: str
    repo_url: str


class RepoListResponse(BaseModel):
    repos: list[RepoInfo]
    default_repo_id: str | None = None


class TokenBucketLimiter:
    """Simple token-bucket rate limiter for a single caller's budget."""

    def __init__(self, rate_per_minute: int):
        self.capacity = rate_per_minute
        self.tokens = rate_per_minute
        self.refill_rate = rate_per_minute / 60.0  # tokens per second
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()

    def allow(self) -> bool:
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


class PerClientTokenBucketLimiter:
    """
    One TokenBucketLimiter per client identity (API key, or "anonymous"
    when auth is disabled), instead of one shared process-wide bucket. A
    single shared bucket meant one abusive/buggy caller could exhaust the
    entire budget for every other caller simultaneously -- since there was
    no per-caller identity to distinguish them, that was true of every
    client hitting the service, not just a hypothetical bad actor.

    Bounded via LRU eviction (MAX_TRACKED_CLIENTS) so an unbounded number
    of distinct identities (e.g. IPs, if this is ever keyed by IP) can't
    grow this dict forever.
    """

    MAX_TRACKED_CLIENTS = 10_000

    def __init__(self, rate_per_minute: int):
        self.rate_per_minute = rate_per_minute
        self._buckets: OrderedDict[str, TokenBucketLimiter] = OrderedDict()
        self._lock = threading.Lock()

    def allow(self, client_id: str) -> bool:
        with self._lock:
            bucket = self._buckets.get(client_id)
            if bucket is None:
                if len(self._buckets) >= self.MAX_TRACKED_CLIENTS:
                    self._buckets.popitem(last=False)
                bucket = TokenBucketLimiter(self.rate_per_minute)
                self._buckets[client_id] = bucket
            else:
                self._buckets.move_to_end(client_id)
        return bucket.allow()


def get_client_id(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> str:
    """
    FastAPI dependency: validates X-API-Key when auth is configured, and
    returns a stable per-caller identity used both for that decision and as
    the rate-limiter bucket key. If API_KEYS is empty, auth is disabled
    (local dev only -- see common/config.py) and every caller is treated as
    a single shared "anonymous" identity, same as the old process-wide
    limiter behavior.
    """
    if not API_KEYS:
        return "anonymous"
    if x_api_key is None or x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")
    return x_api_key


class _RepoEntry:
    __slots__ = ("pipeline", "repo_url")

    def __init__(self, pipeline: CodebaseRAGPipeline, repo_url: str):
        self.pipeline = pipeline
        self.repo_url = repo_url


class PipelineStore:
    """
    LRU cache of repo_id -> (pipeline, repo_url), replacing a single global
    `state["pipeline"]`. Previously, ingesting a new repo mutated that one
    global in place -- every other in-flight or future query from every
    other user would silently start hitting a different codebase mid-
    session. Now each ingested repo gets its own addressable pipeline;
    /query can target one explicitly via repo_id, or default to the most
    recently ingested one for backward compatibility with callers that
    don't pass repo_id.

    Bounded by MAX_ACTIVE_REPOS since each pipeline holds its repo's full
    embedding matrix in memory (retrieval/hybrid_search.py's vector_search
    loads all embeddings per query) -- unbounded growth here is unbounded
    memory growth. Evicting a repo doesn't delete its on-disk data; it just
    has to be re-ingested to come back into the live cache.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self._entries: OrderedDict[str, _RepoEntry] = OrderedDict()
        self._lock = threading.Lock()
        self.default_repo_id: str | None = None

    def get(self, repo_id: str | None) -> tuple[str | None, CodebaseRAGPipeline | None]:
        with self._lock:
            rid = repo_id or self.default_repo_id
            if rid is None:
                return None, None
            entry = self._entries.get(rid)
            if entry is None:
                return rid, None
            self._entries.move_to_end(rid)
            return rid, entry.pipeline

    def put(self, repo_id: str, pipeline: CodebaseRAGPipeline, repo_url: str, make_default: bool = True) -> None:
        with self._lock:
            self._entries[repo_id] = _RepoEntry(pipeline, repo_url)
            self._entries.move_to_end(repo_id)
            if make_default:
                self.default_repo_id = repo_id
            while len(self._entries) > self.max_size:
                evicted_id, _ = self._entries.popitem(last=False)
                logger.info("Evicted repo_id=%s from in-memory pipeline cache (LRU, max_size=%d)", evicted_id, self.max_size)

    def list_repos(self) -> list[tuple[str, str]]:
        with self._lock:
            return [(rid, e.repo_url) for rid, e in self._entries.items()]

    def repo_url_for(self, repo_id: str | None) -> str:
        if repo_id is None:
            return ""
        with self._lock:
            entry = self._entries.get(repo_id)
            return entry.repo_url if entry else ""


_ALLOWED_INGEST_HOSTS = {"github.com", "gitlab.com"}


def _validate_repo_url(repo_url: str) -> None:
    """
    Proper URL parsing instead of a bare str.startswith() prefix check --
    startswith would accept anything with the right literal prefix without
    confirming the host is actually github.com/gitlab.com or that there's a
    real owner/repo path to clone, and gives no structured way to extend
    the allowlist later.
    """
    parsed = urlparse(repo_url)
    if parsed.scheme != "https" or parsed.netloc not in _ALLOWED_INGEST_HOSTS:
        raise HTTPException(
            status_code=400,
            detail="Only public https://github.com/... or https://gitlab.com/... URLs are supported",
        )
    if len(parsed.path.strip("/").split("/")) < 2:
        raise HTTPException(status_code=400, detail="repo_url must include an owner/repo path")


app = FastAPI(title="Codebase RAG API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

if not API_KEYS:
    logger.warning("API_KEYS is not set -- authentication is DISABLED. Set API_KEYS (comma-separated) before deploying publicly.")
if ALLOWED_ORIGINS == ["*"]:
    logger.warning("ALLOWED_ORIGINS is not set -- CORS is wide open (*). Set ALLOWED_ORIGINS before deploying publicly.")


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Assigns a request ID used both in the response header (for client-side
    correlation when reporting an issue) and in every log line emitted while
    handling this request (via common/logging_config.py's contextvar), so a
    single "Pipeline error" report can be traced back through every module
    it touched."""
    req_id = str(uuid.uuid4())
    token = request_id_var.set(req_id)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = req_id
    return response


query_limiter = PerClientTokenBucketLimiter(QUERY_RATE_LIMIT_PER_MIN)
ingest_limiter = PerClientTokenBucketLimiter(INGEST_RATE_LIMIT_PER_MIN)  # cloning/embedding is expensive -- stricter limit

DATA_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DB_PATH = str(DATA_DIR / "jobs.db")

pipeline_store = PipelineStore(max_size=MAX_ACTIVE_REPOS)

# Dedupes concurrent /ingest calls for the same repo -- without this, two
# requests for the same repo_url race on ingest_from_url.clone_repo's
# shutil.rmtree + reclone of the same workspace slug directory.
_in_flight_slugs: set[str] = set()
_in_flight_lock = threading.Lock()

_STALE_JOB_AFTER_SECONDS = 30 * 60


def _load_default_pipeline() -> None:
    try:
        pipeline = CodebaseRAGPipeline(
            db_path=str(DATA_DIR / "store.db"),
            graph_path=str(DATA_DIR / "graph.json"),
            embedder_path=str(DATA_DIR / "embedder.pkl"),
            repo_id="fastapi_test",
        )
    except FileNotFoundError:
        logger.warning("No default dataset found under %s -- call /ingest before /query will work", DATA_DIR)
        return
    pipeline_store.put("fastapi_test", pipeline, repo_url="fastapi/fastapi (default)")


_load_default_pipeline()


@app.get("/health")
def health():
    repos = pipeline_store.list_repos()
    return {
        "status": "ok",
        "ready": len(repos) > 0,
        "default_repo_id": pipeline_store.default_repo_id,
        "active_repos": len(repos),
    }


@app.get("/repos", response_model=RepoListResponse)
def list_repos(client_id: str = Depends(get_client_id)):
    repos = [RepoInfo(repo_id=rid, repo_url=url) for rid, url in pipeline_store.list_repos()]
    return RepoListResponse(repos=repos, default_repo_id=pipeline_store.default_repo_id)


def _run_ingest_job(job_id: str, repo_url: str, slug: str):
    """
    Runs in a background thread. Cloning + embedding can take anywhere from
    seconds to minutes depending on repo size and available CPU/RAM -- doing
    this inside the request/response cycle is fragile on constrained infra
    (a slow container can hit a reverse-proxy timeout or get OOM-killed
    mid-request, orphaning the caller with no response ever coming back).
    Returning a job_id immediately and polling status instead means a slow
    or failed ingest degrades to "still running" / a clear error, not a
    silently hung request.
    """
    try:
        result = ingest_from_url(repo_url, workspace_root=WORKSPACE_ROOT)
        new_pipeline = CodebaseRAGPipeline(
            db_path=result["db_path"],
            graph_path=result["graph_path"],
            embedder_path=result["embedder_path"],
            repo_id=result["slug"],
        )
        pipeline_store.put(result["slug"], new_pipeline, repo_url=repo_url)
        job_store.update_job(
            JOBS_DB_PATH, job_id,
            status="done", repo_id=result["slug"],
            num_chunks=result["num_chunks"], num_files=result["num_files"],
            source_subdir=result["source_subdir"],
        )
        logger.info("Ingest job %s done: repo_url=%s repo_id=%s chunks=%d", job_id, repo_url, result["slug"], result["num_chunks"])
    except Exception as e:
        logger.exception("Ingest job %s failed: repo_url=%s", job_id, repo_url)
        job_store.update_job(JOBS_DB_PATH, job_id, status="error", error=str(e))
    finally:
        with _in_flight_lock:
            _in_flight_slugs.discard(slug)


@app.post("/ingest", response_model=IngestStartedResponse)
def ingest(req: IngestRequest, background_tasks: BackgroundTasks, client_id: str = Depends(get_client_id)):
    if not ingest_limiter.allow(client_id):
        raise HTTPException(status_code=429, detail="Ingestion rate limit exceeded, try again shortly")

    _validate_repo_url(req.repo_url)

    slug = slugify_repo_url(req.repo_url)
    with _in_flight_lock:
        if slug in _in_flight_slugs:
            raise HTTPException(
                status_code=409,
                detail=f"An ingest for this repo is already in progress (slug={slug}); poll its job or wait for it to finish",
            )
        _in_flight_slugs.add(slug)

    job_id = str(uuid.uuid4())
    job_store.create_job(JOBS_DB_PATH, job_id, req.repo_url)
    logger.info("Ingest job %s started: repo_url=%s slug=%s client=%s", job_id, req.repo_url, slug, client_id)

    background_tasks.add_task(_run_ingest_job, job_id, req.repo_url, slug)
    return IngestStartedResponse(job_id=job_id, status="started")


@app.get("/ingest/status/{job_id}", response_model=IngestStatusResponse)
def ingest_status(job_id: str, client_id: str = Depends(get_client_id)):
    job = job_store.get_job(JOBS_DB_PATH, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    stale = job["status"] == "running" and (time.time() - job["created_at"]) > _STALE_JOB_AFTER_SECONDS
    return IngestStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        repo_url=job["repo_url"],
        repo_id=job.get("repo_id"),
        num_chunks=job.get("num_chunks"),
        num_files=job.get("num_files"),
        source_subdir=job.get("source_subdir"),
        error=job.get("error"),
        stale=stale,
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, client_id: str = Depends(get_client_id)):
    if not query_limiter.allow(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded, try again shortly")

    repo_id, pipeline = pipeline_store.get(req.repo_id)
    if pipeline is None:
        if req.repo_id is not None:
            raise HTTPException(status_code=404, detail=f"Unknown or not-yet-ingested repo_id: {req.repo_id!r}")
        raise HTTPException(status_code=400, detail="No repo has been ingested yet -- call /ingest first")

    try:
        result = pipeline.run(req.question)
    except Exception:
        # Log the full exception server-side (with request-id correlation);
        # the client only ever gets a generic message -- returning
        # str(e) directly used to leak internals (file paths, library
        # tracebacks) to any caller.
        logger.exception("Pipeline error handling query (repo_id=%s)", repo_id)
        raise HTTPException(status_code=500, detail="Internal error processing query. Please try again.")

    sources = [
        RetrievedChunkResponse(
            qualified_name=r.chunk["qualified_name"],
            file_path=r.chunk["file_path"],
            start_line=r.chunk["start_line"],
            end_line=r.chunk["end_line"],
            source=r.source,
        )
        for r in result["ranked"]
    ]

    return QueryResponse(
        answer=result["answer"],
        grounded=result["grounded"],
        warning=result.get("ungrounded_warning", ""),
        sources=sources,
        repo_id=repo_id or "",
        active_repo=pipeline_store.repo_url_for(repo_id),
    )
