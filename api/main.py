"""
API layer. This is deliberately the most "plain engineering" file in the
project -- Pydantic schemas, a token-bucket rate limiter, structured error
handling -- since that's what an SDE interview actually probes, more than
the retrieval internals.
"""

import time
import threading
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.pipeline import CodebaseRAGPipeline
from ingestion.ingest_from_url import ingest_from_url


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class IngestRequest(BaseModel):
    repo_url: str = Field(..., min_length=8, max_length=300)


class IngestResponse(BaseModel):
    status: str
    repo_url: str
    num_chunks: int
    num_files: int
    source_subdir: str


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
    active_repo: str = ""


class TokenBucketLimiter:
    """Simple token-bucket rate limiter -- process-wide, same pattern used
    in CareerRadar's scraper rate limiting, applied here to LLM calls."""

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


app = FastAPI(title="Codebase RAG API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = TokenBucketLimiter(rate_per_minute=15)
ingest_limiter = TokenBucketLimiter(rate_per_minute=3)  # cloning/embedding is expensive -- stricter limit

DATA_DIR = Path(__file__).parent.parent / "data"

# App state: which repo is currently active. Starts pointed at whatever was
# ingested by ingestion/run_ingestion.py (FastAPI's source, by default).
# A single global pipeline is a deliberate simplification for a demo --
# swapping it isn't safe under concurrent requests from different users.
# Production would key storage by repo_id / session instead of mutating
# shared global state. Said explicitly here rather than hidden.
state = {
    "pipeline": None,
    "active_repo": "fastapi/fastapi (default)",
}
state_lock = threading.Lock()


def _load_default_pipeline():
    try:
        return CodebaseRAGPipeline(
            db_path=str(DATA_DIR / "store.db"),
            graph_path=str(DATA_DIR / "graph.json"),
            embedder_path=str(DATA_DIR / "embedder.pkl"),
        )
    except FileNotFoundError:
        return None  # no default data ingested yet -- /ingest is required first


state["pipeline"] = _load_default_pipeline()


@app.get("/health")
def health():
    return {"status": "ok", "active_repo": state["active_repo"], "ready": state["pipeline"] is not None}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    if not ingest_limiter.allow():
        raise HTTPException(status_code=429, detail="Ingestion rate limit exceeded, try again shortly")
    if not req.repo_url.startswith(("https://github.com/", "https://gitlab.com/")):
        raise HTTPException(status_code=400, detail="Only public GitHub/GitLab HTTPS URLs are supported")

    try:
        result = ingest_from_url(req.repo_url)
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    new_pipeline = CodebaseRAGPipeline(
        db_path=result["db_path"],
        graph_path=result["graph_path"],
        embedder_path=result["embedder_path"],
    )

    with state_lock:
        state["pipeline"] = new_pipeline
        state["active_repo"] = req.repo_url

    return IngestResponse(
        status="indexed",
        repo_url=req.repo_url,
        num_chunks=result["num_chunks"],
        num_files=result["num_files"],
        source_subdir=result["source_subdir"],
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not limiter.allow():
        raise HTTPException(status_code=429, detail="Rate limit exceeded, try again shortly")

    pipeline = state["pipeline"]
    if pipeline is None:
        raise HTTPException(status_code=400, detail="No repo has been ingested yet -- call /ingest first")

    try:
        result = pipeline.run(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

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
        active_repo=state["active_repo"],
    )