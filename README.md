# CodeRAG — Hybrid GraphRAG Q&A over Any Codebase

Point it at a GitHub repo URL, wait for it to index, then ask natural-language
questions about that codebase and get grounded answers with file/line
citations. Combines vector search, BM25, and lightweight graph traversal,
reranked, and orchestrated through a **fixed-edge pipeline** rather than an
autonomous agent loop.

Built to demonstrate production-RAG engineering discipline, not just "call
an LLM with retrieved context":

- **Hybrid retrieval, not just vector search** — dense (semantic) + BM25
  (lexical, catches exact identifiers a vector model can miss) + graph
  neighbor expansion (structural context: calls, imports, class membership)
- **Query rewriting before it hits the index** — a single scoped LLM call
  splits the raw question into a semantic variant, a lexical variant, and
  sub-queries for multi-hop questions
- **Reranking as a distinct stage** — broad, cheap retrieval first; a finer
  (more expensive) relevance signal applied only to the top candidates
- **Deterministic where it should be deterministic** — ingestion, AST
  chunking, graph construction, and candidate merging are all plain code.
  Only two things ever touch an LLM: query rewrite and final generation.
  No open-ended agent loop deciding what to do next.
- **Evals before touching weights or prompts** — a hand-labeled 30-query
  eval set measures Recall@8 / Precision@8 / MRR, with an ablation table
  showing the actual effect of each architectural piece (see below).
- **Works on any Python repo, not just one hardcoded corpus** — `/ingest`
  clones a repo, auto-detects its actual source directory (skips
  tests/docs/vendored noise, handles both flat and `src/`-layout packages),
  and indexes it live. Verified against two structurally different real
  repos (FastAPI, `requests`).

## Architecture

```
  ingestion (triggered by POST /ingest, deterministic once triggered)
    git clone -> auto-detect source dir -> AST chunker -> {chunks.jsonl, graph.json}
                                              |
                                        embed (sentence-transformers, pluggable)
                                              |
                                        store (SQLite locally / Postgres+pgvector in prod)

  query time (POST /query, LangGraph, fixed edges)
    rewrite -> hybrid_retrieve (vector + BM25 + graph) -> rerank
             -> assemble_context -> generate -> self_check -> answer
```

A minimal pastel-themed web UI (`ui/index.html`) sits on top of both
endpoints — paste a repo URL, index it, then ask questions, no build step.

**Known simplification, stated rather than hidden:** ingestion is
synchronous (the `/ingest` request blocks until cloning+embedding finish)
and a single global pipeline instance backs the whole API, so indexing a
new repo replaces the previously active one. Fine for a demo; a production
version would run ingestion as a background job with status polling, and
key storage by `repo_id` so multiple repos/users can be served concurrently
without one user's ingest call evicting another's active repo.

## Eval results (30 hand-labeled queries, k=8, indexed repo: FastAPI)

Embeddings: real sentence-transformers (all-MiniLM-L6-v2, 384-dim).
Reranker: lexical field-weighted overlap (see `retrieval/reranker.py` —
a real cross-encoder is implemented and swappable via
`get_reranker("cross-encoder")`, not substituted into this specific run).

| Config                     | Recall@8 | Precision@8 | MRR   |
|-----------------------------|---------:|-------------:|------:|
| vector-only                 | 0.933    | 0.121         | 0.651 |
| vector + BM25 (hybrid)       | 0.933    | 0.121         | 0.683 |
| hybrid + rerank              | 0.967    | 0.125         | 0.748 |
| hybrid + rerank + graph      | 0.967    | 0.125         | 0.753 |

### The debugging story behind these numbers

This system went through five real bugs during development, each caught by
either the eval harness or by testing against a second, different repo —
which is itself the point of building evals and testing on more than one
corpus rather than trusting a single happy-path run.

**1. Hybrid initially underperformed vector-only** (MRR 0.651 → 0.481 in
an earlier run). `merge_candidates` compared raw vector cosine scores
against raw BM25 rank scores directly — two incomparable scales — so
whichever channel produced larger numbers dominated the merge regardless
of actual relevance. Fixed with Reciprocal Rank Fusion (RRF): merge by
each candidate's *rank* within its own list instead of its raw score.

**2. The fix didn't show up in eval numbers on the first rerun.** The eval
harness had its own duplicated retrieval-merge logic that flattened vector
and BM25 results into one pre-ranked list before calling the merge
function, destroying the per-channel rank information RRF depends on.
Removed the duplication so eval and the live API share one retrieval path.

**3. Graph expansion still looked neutral-to-negative after the RRF fix.**
Traced to an unbounded candidate pool reaching the reranker — with no cap,
occasional graph neighbors could out-score genuine hits on lexical overlap
alone, since the lexical reranker doesn't use the retrieval-stage score at
all. Capped the pool to the top 20 candidates by retrieval score before
reranking. Graph then flipped to a small, real, positive contribution
(+0.005 MRR) — modest because this 30-query set is mostly single-hop
docstring lookups, where hybrid+rerank alone already finds the answer;
it should matter more for structural/multi-hop questions ("what breaks if
I change X's signature") that this labeled set under-represents.

**4. Deprecated SDK.** `google-generativeai` reached end-of-support;
migrated `GeminiLLM` and `GeminiEmbedder` to the new `google-genai` client
(`genai.Client(...)` / `client.models.generate_content(...)`), removing a
`FutureWarning` on every startup.

**5. `chunk_id` collisions on a second, different repo.** Ingesting
FastAPI never surfaced this, but running the same pipeline against
`requests` (a different repo, different code patterns) immediately threw
`UNIQUE constraint failed: chunks.chunk_id`. Root cause: `chunk_id` is a
hash of a chunk's qualified name, and the AST chunker had no handling for
legitimate same-name redefinitions — `@property` getter/setter pairs,
`@x.setter`/`@x.deleter`, and conditional `__init__` redefinitions all
produce two different function bodies under the identical qualified name.
Fixed by disambiguating repeated names with their source line number
(`Module.Class.method@L94`) the second time a name is seen within a file.
Verified: 320/320 chunks now produce 320 unique IDs on `requests`, where 20
real same-name redefinitions were correctly caught and disambiguated.

## API

- `POST /ingest` — `{"repo_url": "https://github.com/org/repo"}`. Clones,
  auto-detects the source directory, chunks, embeds, and stores. Returns
  chunk/file counts. Rate-limited separately and more strictly than
  `/query` (cloning+embedding is expensive).
- `POST /query` — `{"question": "..."}`. Runs the full LangGraph pipeline
  against whichever repo was last ingested. Returns the answer, a
  groundedness flag, and the retrieved sources with file/line citations
  and which retrieval channel (vector/bm25/graph) surfaced each one.
- `GET /health` — status + currently active repo.

## Sandbox note

Large portions of this were first built and debugged in a network-restricted
sandbox without access to huggingface.co, where the embedder/reranker
defaulted to offline stand-ins (TF-IDF+SVD, weighted lexical overlap) behind
the same interface as the real neural models. All numbers and screenshots
in this README are from the real local environment (real sentence-transformer
embeddings, real Gemini generation via `google-genai`), not the sandbox
stand-ins — the swap is one line each in `ingestion/embedder.py` /
`retrieval/reranker.py` / `orchestration/llm_client.py`.

## Setup

```bash
git clone https://github.com/rhythmghai/codebase.git
cd codebase
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/fastapi/fastapi.git repo_src   # or point /ingest at any repo instead
python3 ingestion/run_ingestion.py   # regenerates data/embedder.pkl and data/store.db — not tracked in git, since both are deterministically regeneratable from source
uvicorn api.main:app --reload --port 8000
```

## Stack

FastAPI · LangGraph · SQLite (local) / Postgres+pgvector (Supabase, prod
target) · sentence-transformers (MiniLM embeddings) · networkx (call/import
graph) · Gemini 2.5 flash-lite via `google-genai` (query rewrite +
generation) · vanilla HTML/CSS/JS (UI, no build step)