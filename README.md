# CodeRAG — Hybrid GraphRAG Q&A over Any Codebase

Point it at a GitHub repo URL, wait for it to index, then ask natural-language
questions about that codebase and get grounded answers with file/line
citations. Combines vector search, BM25, and graph traversal (Neo4j/Cypher),
reranked, orchestrated through a **fixed-edge LangGraph pipeline** rather
than an autonomous agent loop.

Built to demonstrate production-RAG engineering discipline, not just "call
an LLM with retrieved context" — most of what's below is the record of
actually testing that claim and fixing what testing found broken.

## What's actually in here

- **Hybrid retrieval** — dense vector search (sentence-transformers embeddings,
  Qdrant HNSW-indexed ANN search — not a brute-force scan) + BM25 full-text +
  graph neighbor expansion (calls / class-containment, extracted
  deterministically from the AST, not LLM-inferred)
- **Real graph database, not a JSON adjacency list** — Neo4j AuraDB, queried
  via Cypher at retrieval time. Ingestion resolves callee references once
  and pushes nodes/edges in; falls back gracefully to a local JSON graph if
  Neo4j isn't configured or unreachable
- **Query rewriting** before it hits the index — a single scoped LLM call,
  never an open-ended agent loop
- **Live ingestion** — `/ingest` clones a repo, auto-detects its source
  layout (handles both flat and `src/`-layout packages), and indexes it as
  an async background job (returns a job_id immediately, polled via
  `/ingest/status/{job_id}`) rather than blocking the request — verified
  against two structurally different real repos (FastAPI, `requests`)
- **Two hand-labeled eval sets** (30 single-hop + 12 multi-hop/structural
  queries) with ablation testing across every architectural stage — not
  just "it works," but "here's the measured effect of each piece, and here's
  what happened when the measurement itself was wrong"

## Architecture

```
  ingestion (POST /ingest, async job + polling)
    git clone -> auto-detect source dir -> AST chunker -> {chunks, call/contains graph}
                                              |                |                |
                                        embed (sentence-   index into      push to Neo4j
                                        transformers)       Qdrant         (Cypher)
                                              |
                                        chunk metadata + BM25 (SQLite)

  query time (POST /query, LangGraph, fixed edges)
    rewrite -> hybrid_retrieve (Qdrant vector search + BM25 + Neo4j graph traversal) -> rerank
             -> assemble_context -> generate -> self_check -> (reflect once if ungrounded) -> answer
```

A pastel-themed web UI (`ui/index.html`) sits on top: paste a repo URL,
index it, ask questions — no build step, polls ingestion status live.

## Eval results

Embeddings: sentence-transformers (all-MiniLM-L6-v2). Two reranker options
implemented and **directly benchmarked against each other**, not just one
assumed to be better:

| Set (n)                | Config                | Recall@8 | MRR   |
|-------------------------|------------------------|---------:|------:|
| Single-hop (30)          | vector-only            | 0.933    | 0.651 |
| Single-hop (30)          | hybrid                 | 0.933    | 0.688 |
| Single-hop (30)          | hybrid + rerank (neural) | 0.967  | 0.735 |
| Single-hop (30)          | hybrid + rerank (lexical)| 0.967  | 0.748 |
| Multi-hop (12)           | vector-only            | 1.000    | 0.535 |
| Multi-hop (12)           | hybrid + rerank (neural) | 0.667  | 0.272 |
| Multi-hop (12)           | hybrid + rerank (lexical)| 0.667  | 0.369 |

**Finding: a lexical, field-weighted reranker consistently beats a general-purpose
neural cross-encoder (`ms-marco-MiniLM-L-6-v2`) on code search, and the gap
widens on harder multi-hop queries.** The cross-encoder is trained on
natural-language web-passage ranking; it has no exposure to code syntax or
identifiers, and it measurably prioritizes prose similarity over the exact
identifier/signature overlap that actually indicates relevance in code. This
is the reranker the system defaults to — chosen from a benchmark, not an
assumption. The neural cross-encoder remains implemented and swappable
(`get_reranker("cross-encoder")`) for anyone who wants to test a
code-fine-tuned alternative.

**Known limitation, stated rather than hidden:** the multi-hop set is only
12 queries — small enough that a single query flipping status moves Recall@8
by ~8 points. The reranker-choice finding replicated independently across
both eval sets and is treated as solid; graph's specific numeric contribution
within the multi-hop set is not treated as stable at this sample size and
would need a larger set (25-30+) before drawing firm conclusions there.

**Note on the table above:** these numbers were measured before the
vector-search backend moved from a brute-force numpy scan to Qdrant
(see "Production hardening"). Both compute exact cosine similarity at
this corpus size (HNSW is near-exact, not lossy, at a few hundred vectors),
so the ranking math is unchanged — but the table hasn't been re-run against
the Qdrant-backed path, so treat these as representative, not re-verified
post-migration.

## The bug-fix history behind these numbers

Five distinct bugs, each found through eval regression or cross-repo/cross-eval-set
testing rather than code inspection alone — this is the actual argument for
why the eval harness and multi-corpus testing exist, not a footnote.

1. **Hybrid initially underperformed vector-only.** Raw vector-cosine and
   BM25-rank scores were compared directly in the merge step — incomparable
   scales, so whichever channel produced larger numbers dominated regardless
   of relevance. Fixed with Reciprocal Rank Fusion (rank-based merging).
2. **The fix didn't show up on the first re-measurement.** The eval harness
   had its own duplicated retrieval-merge logic that destroyed the
   per-channel rank information RRF needs. Removed the duplication.
3. **Graph expansion looked neutral-to-negative even after the RRF fix.**
   Traced to an unbounded candidate pool reaching the reranker. Capped it
   to the top 20 by retrieval score before reranking.
4. **`chunk_id` collisions on a second, different repo.** FastAPI never
   triggered it; `requests` did immediately (`UNIQUE constraint failed`).
   Root cause: chunk IDs were hashed from qualified names alone, and
   `@property` getter/setter pairs, `@x.setter`, and conditional `__init__`
   redefinitions all share a qualified name across genuinely different
   function bodies. Fixed with a composite key
   (`file_path + qualified_name + content_hash + start_line`) — collision-proof
   by construction, verified against a synthetic `@overload` case that
   previously would have broken it.
5. **Verifying the Neo4j migration surfaced a real, pre-existing bug in the
   local JSON fallback path.** `graph_expand_local` resolved neighbor
   candidates against a qname-to-id mapping scoped only to the seed chunks
   themselves — meaning it could structurally never surface a genuinely new
   neighbor. It had been a silent no-op the entire time (confirmed
   empirically: 0 neighbors returned on realistic production seeds before
   the fix, up to 19 after). Fixed by resolving against the full corpus
   instead of the seed-scoped subset. This also means an earlier "+0.005 MRR
   graph contribution" claim measured through this broken path was retracted
   once the bug was found, not kept as a result.

Building the second (multi-hop) eval set itself also surfaced a sixth,
smaller lesson: 3 of its first 4 "failing" queries turned out to be
ground-truth labeling errors, not retrieval failures — the system's actual
top result was correct, the label was wrong. Verifying eval ground truth is
its own discipline, not a one-time setup step.

## API

All endpoints except `GET /health` require an `X-API-Key` header once
`API_KEYS` is set (see Setup below) — unauthenticated, rate-limit-only
access is for local dev only.

- `POST /ingest` — `{"repo_url": "..."}`. Returns `{job_id, status}`
  immediately; actual clone+embed+graph-load runs as a background task.
  Rejects a second concurrent ingest for the same repo (`409`) instead of
  racing two clones against the same workspace directory.
- `GET /ingest/status/{job_id}` — poll for `running` / `done` / `error`.
  Job status is persisted to SQLite (`data/jobs.db`), not held in memory,
  so it survives a process restart as long as `DATA_DIR` is on persistent
  storage. A `running` job older than 30 minutes is flagged `stale: true`
  — a background task lost to a restart has no way to update its own
  status, so this is a signal to re-ingest rather than keep polling.
- `POST /query` — `{"question": "...", "repo_id": "<optional>"}`. Omit
  `repo_id` to query the most recently ingested repo. Returns the answer, a
  groundedness flag, and retrieved sources with file/line citations and
  which channel (vector/bm25/graph) surfaced each one.
- `GET /repos` — lists every repo currently loaded (bounded by
  `MAX_ACTIVE_REPOS`, LRU-evicted) and which one is the default for
  `/query` calls that omit `repo_id`.
- `GET /health` — status, whether any repo is ready to query, and how many
  are currently loaded. No auth required (standard for infra health checks).

Every response carries an `X-Request-ID` header; server-side logs for that
request are tagged with the same ID, so a client-reported error can be
traced back through retrieval/pipeline/storage without guessing.

## Known simplifications (stated, not hidden)

- Ingestion is full delete-then-reinsert, not incremental upsert — correct
  but wasteful for a repo that's barely changed since last index.
- Neo4j's AuraDB free tier is a single shared instance — every node/edge is
  scoped by `repo_id` to prevent cross-repo contamination, but there's no
  per-tenant isolation beyond that property filter.
- `DATA_DIR` and `WORKSPACE_ROOT` default to paths inside the app's own
  filesystem. On most container platforms (Railway included) that's
  ephemeral across redeploys — mount a persistent volume at those paths (or
  point them at one) if ingested repos, the Qdrant vector index, and job
  history need to survive a restart.
- Chunk metadata + BM25 still live in SQLite, mirroring a future
  Postgres+FTS schema — that migration (for the metadata store only, not
  vectors, which are already Qdrant) hasn't happened in this codebase yet.
- Auth is a single shared-secret allowlist (`API_KEYS`), not per-user
  accounts/OAuth/key rotation — sufficient to stop the service being an
  open proxy for compute and LLM spend, not a full identity system.

## Production hardening (added after the initial build)

A structured review of this codebase for production-readiness (not just
correctness) turned up a specific list of gaps, all since addressed:

- **A real vector database** — vector search used to be a brute-force numpy
  scan over every embedding stored as a raw BLOB in SQLite (O(n) per query,
  the whole corpus resident in memory, no index at all). Now backed by
  Qdrant (HNSW-indexed ANN search), embedded/local by default (zero extra
  setup — same operational shape as SQLite) with a one-env-var upgrade path
  (`QDRANT_URL`) to a hosted/self-hosted instance, mirroring exactly how
  Neo4j's optional-upgrade shape already worked.
- **Auth + per-client rate limiting** — `X-API-Key` required once `API_KEYS`
  is set; the token-bucket rate limiter is keyed per caller identity, not
  shared process-wide (previously, one caller exhausting the shared budget
  throttled every other caller too).
- **Multi-repo state** — ingesting a new repo used to silently redirect
  every other in-flight/future query to a different codebase. Pipelines are
  now cached per `repo_id` (LRU-bounded via `MAX_ACTIVE_REPOS`), addressable
  via `/query`'s optional `repo_id` field.
- **Persisted job state** — `GET /ingest/status/{job_id}` used to be backed
  by an in-memory dict, lost on every restart. Now backed by SQLite.
- **LLM failure handling** — `rewrite_query`/`generate_answer` calls are
  now wrapped with a timeout + retry (`tenacity`) and fall back to the
  deterministic `RuleBasedLLM` if the real backend keeps failing, instead
  of the whole request hard-failing on a transient upstream error.
  Malformed JSON from the rewrite call degrades gracefully instead of
  raising an uncaught `JSONDecodeError`.
- **A real (bounded) reflection loop** — previously, `self_check` computed
  a groundedness flag but took no action on a negative result beyond a
  warning string. The LangGraph pipeline now conditionally routes back
  through one stricter regeneration attempt when the first answer looks
  ungrounded, before giving up and returning the warning.
- **No more leaking internals in errors** — `/query`'s exception handler
  used to return `str(e)` straight to the client. It now logs the full
  exception server-side (tagged with the request ID) and returns a fixed,
  generic message.
- **Transactional graph writes** — `Neo4jGraphStore.load_graph`'s
  clear/create-nodes/create-edges steps used to be four separate implicit
  transactions; a crash between any two left the graph partially loaded
  with no way to detect it. Now one explicit transaction, all-or-nothing.
- **Structured logging throughout** — every request gets a request ID
  (returned as `X-Request-ID`, and attached to every log line emitted while
  handling it); Neo4j/graph-fallback failures are logged instead of
  silently swallowed.
- **A real test suite + CI** — `tests/` (pytest, offline/no network —
  `TfidfEmbedder` + `LexicalReranker` + `RuleBasedLLM` stand in for the
  neural/network-dependent backends) covers retrieval RRF math, the graph-
  neighbor-resolution bug class from the fix history below, the chunk-ID
  collision bug class, the LLM-fallback and reflection-loop logic, and the
  API layer's auth/rate-limit/validation/error paths. Runs on every push
  via GitHub Actions (`.github/workflows/ci.yml`).

## Stack

FastAPI · LangGraph · Qdrant (HNSW-indexed vector search, embedded/local by
default, upgrades to hosted/self-hosted via `QDRANT_URL`) · Neo4j AuraDB
(Cypher graph traversal) · SQLite (chunk metadata + BM25 full-text, local
/ Postgres+FTS in prod — not yet migrated, see Known simplifications) ·
sentence-transformers (MiniLM embeddings) · Gemini 2.5 flash-lite via
`google-genai` (query rewrite + generation) · `tenacity` (LLM
retry/backoff) · pytest + GitHub Actions (tests/CI) · vanilla HTML/CSS/JS
(UI, no build step)

## Setup

```bash
git clone https://github.com/rhythmghai/codebase.git
cd codebase
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt   # requirements.txt alone omits pytest/httpx
cp .env.example .env                  # fill in API_KEYS at minimum before any public deployment
git clone https://github.com/fastapi/fastapi.git repo_src   # or point /ingest at any repo instead
python3 ingestion/run_ingestion.py   # regenerates data/embedder.pkl and data/store.db -- not tracked in git, deterministically regeneratable from source
uvicorn api.main:app --reload --port 8000
```

Or via Docker: `docker build -t coderag . && docker run -p 8000:8000 --env-file .env coderag`
(mount a volume at `DATA_DIR`/`WORKSPACE_ROOT` for data to survive a restart).

Run the test suite: `pytest` (offline, no network or GPU required — see
"Production hardening" above for what it covers).

Qdrant (required, zero setup by default): vector search always runs
through Qdrant, but its embedded mode needs no account or server —
`data/qdrant` is created automatically on first ingest. Set `QDRANT_URL`
(+ `QDRANT_API_KEY`) instead to point at a real hosted/self-hosted
instance, with no code changes.

Neo4j (optional but recommended): set `NEO4J_URI`, `NEO4J_USERNAME`,
`NEO4J_PASSWORD` in `.env` (free AuraDB instance). Without it, graph
expansion falls back to the local JSON file automatically.

See `.env.example` for the full list of environment variables (auth, CORS,
rate limits, data/workspace paths, LLM timeout) and what each defaults to.