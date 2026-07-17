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

- **Hybrid retrieval** — dense vector search (sentence-transformers) + BM25
  full-text + graph neighbor expansion (calls / class-containment,
  extracted deterministically from the AST, not LLM-inferred)
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
                                              |                        |
                                        embed (sentence-transformers)  push to Neo4j (Cypher)
                                              |
                                        store (SQLite / Postgres+pgvector in prod)

  query time (POST /query, LangGraph, fixed edges)
    rewrite -> hybrid_retrieve (vector + BM25 + Neo4j graph traversal) -> rerank
             -> assemble_context -> generate -> self_check -> answer
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

- `POST /ingest` — `{"repo_url": "..."}`. Returns `{job_id, status}`
  immediately; actual clone+embed+graph-load runs as a background task.
- `GET /ingest/status/{job_id}` — poll for `running` / `done` / `error`.
- `POST /query` — `{"question": "..."}`. Returns the answer, a groundedness
  flag, and retrieved sources with file/line citations and which channel
  (vector/bm25/graph) surfaced each one.
- `GET /health` — status + currently active repo.

## Known simplifications (stated, not hidden)

- One global pipeline instance backs the API — indexing a new repo replaces
  the previously active one. A production version would key storage by
  `repo_id` to serve multiple repos/users concurrently.
- Neo4j's AuraDB free tier is a single shared instance — every node/edge is
  scoped by `repo_id` to prevent cross-repo contamination, but there's no
  per-tenant isolation beyond that property filter.
- Ingestion is full delete-then-reinsert, not incremental upsert — correct
  but wasteful for a repo that's barely changed since last index.

## Stack

FastAPI · LangGraph · Neo4j AuraDB (Cypher graph traversal) · SQLite (local)
/ Postgres+pgvector (Supabase, prod target) · sentence-transformers (MiniLM
embeddings) · Gemini 2.5 flash-lite via `google-genai` (query rewrite +
generation) · vanilla HTML/CSS/JS (UI, no build step)

## Setup

```bash
git clone https://github.com/rhythmghai/codebase.git
cd codebase
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/fastapi/fastapi.git repo_src   # or point /ingest at any repo instead
python3 ingestion/run_ingestion.py   # regenerates data/embedder.pkl and data/store.db -- not tracked in git, deterministically regeneratable from source
uvicorn api.main:app --reload --port 8000
```

Neo4j (optional but recommended): set `NEO4J_URI`, `NEO4J_USERNAME`,
`NEO4J_PASSWORD` in `.env` (free AuraDB instance). Without it, graph
expansion falls back to the local JSON file automatically.