"""
Neo4j-backed graph store, replacing the JSON-file + in-process dict
traversal that graph_expand() previously did by hand.

Why this exists: the AST chunker already extracts real relationships
(calls, class-containment) deterministically from source code -- that part
doesn't change. What changes is *where the graph lives* and *how it's
queried*. Previously: dump edges to a JSON file, load the whole thing into
memory on every query, walk Python dicts by hand. Now: push nodes/edges
into a real graph database once at ingestion time, query it with Cypher at
retrieval time. This is what makes the "graph" in GraphRAG a graph
database, not just an adjacency list with an ambitious name.

Multi-tenancy: AuraDB's free tier is one shared database instance, so every
node/relationship is scoped by `repo_id` (the same slug used elsewhere in
the codebase for per-repo storage paths) to keep multiple ingested repos
from cross-contaminating each other's graphs.

Call resolution: the chunker only records the *unresolved* callee name it
saw in source (e.g. "matches", not "fastapi.routing.APIRoute.matches") --
resolving that to a real qualified_name, when possible, happens once here
at load time rather than on every single query the way the old JSON-based
graph_expand() did it. That's strictly better: the resolution cost is paid
once per ingestion, not once per retrieval call.
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
load_dotenv()


class Neo4jGraphStore:
    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        uri = uri or os.environ.get("NEO4J_URI")
        user = user or os.environ.get("NEO4J_USERNAME", "neo4j")
        password = password or os.environ.get("NEO4J_PASSWORD")
        if not uri or not password:
            raise ValueError("NEO4J_URI and NEO4J_PASSWORD must be set (env vars or passed explicitly)")

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_constraints()

    def close(self):
        self.driver.close()

    def _ensure_constraints(self):
        # Uniqueness per (repo_id, qualified_name) -- lets MERGE act as a
        # real upsert instead of creating duplicate nodes on re-ingestion.
        with self.driver.session() as session:
            session.run(
                """
                CREATE CONSTRAINT entity_repo_qname IF NOT EXISTS
                FOR (n:Entity) REQUIRE (n.repo_id, n.qualified_name) IS UNIQUE
                """
            )

    def clear_repo(self, repo_id: str):
        """Delete all nodes/edges for a repo before re-ingesting it (mirrors
        the SQLite side's delete-then-reinsert full-rebuild approach)."""
        with self.driver.session() as session:
            session.run(
                "MATCH (n:Entity {repo_id: $repo_id}) DETACH DELETE n",
                repo_id=repo_id,
            )

    def load_graph(self, repo_id: str, chunks: list[dict], graph_data: dict):
        """
        chunks: list of chunk dicts (from Chunk.__dict__), used to know which
                qualified_names actually exist as real entities.
        graph_data: the GraphData dict produced by the chunker
                    ({"nodes", "calls", "imports", "contains"}).
        """
        self.clear_repo(repo_id)

        known_qnames = {c["qualified_name"] for c in chunks}
        qname_to_module = {
            c["qualified_name"]: c["qualified_name"].rsplit(".", 1)[0]
            for c in chunks
        }

        # Resolve unresolved callee short-names to real qualified_names,
        # same heuristic the old graph_expand() used (same-module lookup),
        # but done once here instead of on every query.
        resolved_calls = []
        for caller_qname, callee_short_name in graph_data.get("calls", []):
            module = qname_to_module.get(caller_qname, caller_qname.rsplit(".", 1)[0])
            candidate = f"{module}.{callee_short_name}"
            if candidate in known_qnames and candidate != caller_qname:
                resolved_calls.append((caller_qname, candidate))

        contains_edges = [
            (cls_q, method_q) for cls_q, method_q in graph_data.get("contains", [])
            if cls_q in known_qnames and method_q in known_qnames
        ]

        with self.driver.session() as session:
            # Batch-create nodes
            session.run(
                """
                UNWIND $nodes AS node
                MERGE (n:Entity {repo_id: $repo_id, qualified_name: node.qualified_name})
                SET n.kind = node.kind, n.file_path = node.file_path,
                    n.name = node.name, n.start_line = node.start_line, n.end_line = node.end_line
                """,
                repo_id=repo_id,
                nodes=[
                    {
                        "qualified_name": c["qualified_name"], "kind": c["kind"],
                        "file_path": c["file_path"], "name": c["name"],
                        "start_line": c["start_line"], "end_line": c["end_line"],
                    }
                    for c in chunks
                ],
            )

            # Batch-create CALLS edges
            session.run(
                """
                UNWIND $edges AS edge
                MATCH (a:Entity {repo_id: $repo_id, qualified_name: edge.src})
                MATCH (b:Entity {repo_id: $repo_id, qualified_name: edge.dst})
                MERGE (a)-[:CALLS]->(b)
                """,
                repo_id=repo_id,
                edges=[{"src": s, "dst": d} for s, d in resolved_calls],
            )

            # Batch-create CONTAINS edges
            session.run(
                """
                UNWIND $edges AS edge
                MATCH (a:Entity {repo_id: $repo_id, qualified_name: edge.src})
                MATCH (b:Entity {repo_id: $repo_id, qualified_name: edge.dst})
                MERGE (a)-[:CONTAINS]->(b)
                """,
                repo_id=repo_id,
                edges=[{"src": s, "dst": d} for s, d in contains_edges],
            )

        return {"nodes": len(chunks), "calls": len(resolved_calls), "contains": len(contains_edges)}

    def get_neighbors(self, repo_id: str, qualified_names: list[str], max_neighbors_per_seed: int = 3) -> list[str]:
        """
        1-hop neighbors (either direction) via CALLS or CONTAINS, for a set
        of seed qualified_names. This is the direct Cypher replacement for
        the old graph_expand()'s hand-rolled Python dict walk.
        """
        if not qualified_names:
            return []

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (seed:Entity {repo_id: $repo_id})-[:CALLS|CONTAINS]-(neighbor:Entity {repo_id: $repo_id})
                WHERE seed.qualified_name IN $qnames AND NOT neighbor.qualified_name IN $qnames
                RETURN DISTINCT neighbor.qualified_name AS qname
                LIMIT $limit
                """,
                repo_id=repo_id,
                qnames=qualified_names,
                limit=max_neighbors_per_seed * len(qualified_names),
            )
            return [record["qname"] for record in result]


def get_neo4j_store() -> Neo4jGraphStore | None:
    """Returns None (rather than raising) if Neo4j isn't configured, so
    callers can gracefully fall back to the local JSON-based graph."""
    try:
        return Neo4jGraphStore()
    except Exception:
        return None