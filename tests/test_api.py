"""
FastAPI TestClient tests for api/main.py: auth, per-client rate limiting,
input validation, error handling, and job status persistence -- the things
a request handler could actually get wrong in production. Retrieval
*quality* is eval/eval_harness.py's job, not this file's.

api.main and common.config read their settings from env vars at import
time, so each test that needs a specific config reloads both modules
fresh against monkeypatched env vars -- this keeps tests isolated from
each other and from the real project's data/ directory.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import TEST_REPO_ID


def _fresh_api_main(monkeypatch, tmp_path, api_keys: str | None = None):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "svc_data"))
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path / "workspaces"))
    if api_keys is None:
        monkeypatch.delenv("API_KEYS", raising=False)
    else:
        monkeypatch.setenv("API_KEYS", api_keys)
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)

    for mod in ("api.main", "common.config"):
        sys.modules.pop(mod, None)
    import api.main as main
    return main


@pytest.fixture
def empty_main(monkeypatch, tmp_path):
    """api.main with no repo ingested/seeded -- for testing the "nothing to
    query yet" and validation paths."""
    return _fresh_api_main(monkeypatch, tmp_path)


@pytest.fixture
def seeded_main(monkeypatch, tmp_path, data_dir):
    """api.main with one repo pre-loaded into pipeline_store directly (not
    via a real /ingest call, which would need network access to clone a
    repo) -- for testing /query."""
    main = _fresh_api_main(monkeypatch, tmp_path)
    from orchestration.pipeline import CodebaseRAGPipeline

    pipeline = CodebaseRAGPipeline(
        db_path=str(data_dir / "store.db"),
        graph_path=str(data_dir / "graph.json"),
        embedder_path=str(data_dir / "embedder.pkl"),
        repo_id=TEST_REPO_ID,
        use_neo4j=False,
    )
    main.pipeline_store.put(TEST_REPO_ID, pipeline, repo_url="https://github.com/test/repo")
    return main


@pytest.fixture
def authed_main(monkeypatch, tmp_path, data_dir):
    """api.main with auth enabled (API_KEYS set) and a repo seeded."""
    main = _fresh_api_main(monkeypatch, tmp_path, api_keys="secret-key-1,secret-key-2")
    from orchestration.pipeline import CodebaseRAGPipeline

    pipeline = CodebaseRAGPipeline(
        db_path=str(data_dir / "store.db"),
        graph_path=str(data_dir / "graph.json"),
        embedder_path=str(data_dir / "embedder.pkl"),
        repo_id=TEST_REPO_ID,
        use_neo4j=False,
    )
    main.pipeline_store.put(TEST_REPO_ID, pipeline, repo_url="https://github.com/test/repo")
    return main


# ---- health / repos ----

def test_health_ok_with_no_repo(empty_main):
    client = TestClient(empty_main.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["ready"] is False


def test_health_ok_with_repo_seeded(seeded_main):
    client = TestClient(seeded_main.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["default_repo_id"] == TEST_REPO_ID


def test_repos_lists_seeded_repo(seeded_main):
    client = TestClient(seeded_main.app)
    resp = client.get("/repos")
    assert resp.status_code == 200
    body = resp.json()
    assert body["default_repo_id"] == TEST_REPO_ID
    assert {r["repo_id"] for r in body["repos"]} == {TEST_REPO_ID}


# ---- /query ----

def test_query_without_any_repo_returns_400(empty_main):
    client = TestClient(empty_main.app)
    resp = client.post("/query", json={"question": "how does foo work?"})
    assert resp.status_code == 400


def test_query_success_returns_answer_and_sources(seeded_main):
    client = TestClient(seeded_main.app)
    resp = client.post("/query", json={"question": "how does foo work?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert body["repo_id"] == TEST_REPO_ID
    assert isinstance(body["sources"], list)


def test_query_explicit_repo_id_success(seeded_main):
    client = TestClient(seeded_main.app)
    resp = client.post("/query", json={"question": "how does foo work?", "repo_id": TEST_REPO_ID})
    assert resp.status_code == 200


def test_query_unknown_repo_id_returns_404(seeded_main):
    client = TestClient(seeded_main.app)
    resp = client.post("/query", json={"question": "how does foo work?", "repo_id": "nonexistent"})
    assert resp.status_code == 404


def test_query_too_short_question_returns_422(seeded_main):
    client = TestClient(seeded_main.app)
    resp = client.post("/query", json={"question": "hi"})
    assert resp.status_code == 422


def test_query_pipeline_exception_returns_generic_500_no_internals_leaked(seeded_main, monkeypatch):
    """
    Regression test: the API used to return f"Pipeline error: {e}" directly
    to the client, leaking exception internals. It must now return a fixed,
    generic message regardless of what the underlying exception says.
    """
    _, pipeline = seeded_main.pipeline_store.get(TEST_REPO_ID)
    monkeypatch.setattr(pipeline, "run", lambda q: (_ for _ in ()).throw(RuntimeError("/etc/passwd secret-path-detail")))

    client = TestClient(seeded_main.app)
    resp = client.post("/query", json={"question": "how does foo work?"})
    assert resp.status_code == 500
    assert "/etc/passwd" not in resp.text
    assert "secret-path-detail" not in resp.text


# ---- /ingest ----

def test_ingest_rejects_non_github_gitlab_url(empty_main):
    client = TestClient(empty_main.app)
    resp = client.post("/ingest", json={"repo_url": "https://evil.example.com/x/y"})
    assert resp.status_code == 400


def test_ingest_rejects_missing_owner_repo_path(empty_main):
    client = TestClient(empty_main.app)
    resp = client.post("/ingest", json={"repo_url": "https://github.com/"})
    assert resp.status_code == 400


def test_ingest_status_unknown_job_returns_404(empty_main):
    client = TestClient(empty_main.app)
    resp = client.get("/ingest/status/does-not-exist")
    assert resp.status_code == 404


def test_ingest_starts_job_and_status_becomes_done(empty_main, monkeypatch, data_dir):
    """Uses a fake ingest_from_url (no network) to verify the whole
    ingest -> background job -> status-poll -> pipeline-registered flow."""

    def fake_ingest_from_url(repo_url, workspace_root=None):
        return {
            "repo_url": repo_url, "slug": "testorg_testrepo", "source_subdir": ".",
            "num_chunks": 5, "num_files": 2,
            "db_path": str(data_dir / "store.db"),
            "graph_path": str(data_dir / "graph.json"),
            "embedder_path": str(data_dir / "embedder.pkl"),
            "neo4j_loaded": False,
        }

    monkeypatch.setattr(empty_main, "ingest_from_url", fake_ingest_from_url)

    client = TestClient(empty_main.app)
    resp = client.post("/ingest", json={"repo_url": "https://github.com/testorg/testrepo"})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    status_resp = client.get(f"/ingest/status/{job_id}")
    assert status_resp.status_code == 200
    body = status_resp.json()
    assert body["status"] == "done"
    assert body["repo_id"] == "testorg_testrepo"
    assert body["num_chunks"] == 5

    # The pipeline must now actually be queryable under its new repo_id.
    query_resp = client.post("/query", json={"question": "how does foo work?", "repo_id": "testorg_testrepo"})
    assert query_resp.status_code == 200


def test_concurrent_ingest_same_repo_returns_409(empty_main):
    """
    Two /ingest calls for the same repo_url must not race on the workspace
    directory (ingest_from_url.clone_repo does shutil.rmtree + reclone of
    the same slug dir) -- the second is rejected while the first is
    in-flight. Marking the slug in-flight directly (rather than actually
    racing two background tasks through TestClient, which runs them
    synchronously and so can't model real concurrency) deterministically
    exercises the same dedup check the endpoint applies to a genuinely
    concurrent second request.
    """
    slug = empty_main.slugify_repo_url("https://github.com/testorg/testrepo")
    with empty_main._in_flight_lock:
        empty_main._in_flight_slugs.add(slug)

    client = TestClient(empty_main.app)
    resp = client.post("/ingest", json={"repo_url": "https://github.com/testorg/testrepo"})
    assert resp.status_code == 409

    with empty_main._in_flight_lock:
        empty_main._in_flight_slugs.discard(slug)


# ---- auth ----

def test_query_without_api_key_when_auth_enabled_returns_401(authed_main):
    client = TestClient(authed_main.app)
    resp = client.post("/query", json={"question": "how does foo work?"})
    assert resp.status_code == 401


def test_query_with_wrong_api_key_returns_401(authed_main):
    client = TestClient(authed_main.app)
    resp = client.post("/query", json={"question": "how does foo work?"}, headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_query_with_valid_api_key_succeeds(authed_main):
    client = TestClient(authed_main.app)
    resp = client.post(
        "/query", json={"question": "how does foo work?"}, headers={"X-API-Key": "secret-key-1"}
    )
    assert resp.status_code == 200


def test_health_does_not_require_api_key(authed_main):
    """/health stays open even when auth is enabled -- standard practice
    for infra health checks, and it reveals nothing sensitive."""
    client = TestClient(authed_main.app)
    resp = client.get("/health")
    assert resp.status_code == 200


# ---- rate limiting ----

def test_query_rate_limit_returns_429_after_budget_exhausted(seeded_main):
    seeded_main.query_limiter = seeded_main.PerClientTokenBucketLimiter(rate_per_minute=2)
    client = TestClient(seeded_main.app)

    r1 = client.post("/query", json={"question": "how does foo work?"})
    r2 = client.post("/query", json={"question": "how does foo work?"})
    r3 = client.post("/query", json={"question": "how does foo work?"})

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429


def test_rate_limit_is_per_client_not_shared(authed_main):
    """The whole point of PerClientTokenBucketLimiter: one client
    exhausting its budget must not affect a different client's budget."""
    authed_main.query_limiter = authed_main.PerClientTokenBucketLimiter(rate_per_minute=1)
    client = TestClient(authed_main.app)

    r1 = client.post("/query", json={"question": "how does foo work?"}, headers={"X-API-Key": "secret-key-1"})
    r2 = client.post("/query", json={"question": "how does foo work?"}, headers={"X-API-Key": "secret-key-1"})
    assert r1.status_code == 200
    assert r2.status_code == 429  # key-1's budget is now exhausted

    r3 = client.post("/query", json={"question": "how does foo work?"}, headers={"X-API-Key": "secret-key-2"})
    assert r3.status_code == 200  # key-2 has its own, untouched budget
