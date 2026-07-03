"""
End-to-end deterministic ingestion: AST chunk -> embed -> store.
No LLM calls anywhere in this file — that's the point.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.chunker import ingest_repo
from ingestion.embedder import get_embedder
from storage.db import build_store

def main():
    project_root = Path(__file__).parent.parent
    repo_root = str(project_root / "repo_src")
    data_dir = str(project_root / "data")
    db_path = str(project_root / "data" / "store.db")

    print("== Step 1: AST chunking + graph construction ==")
    chunks, graph = ingest_repo(repo_root, data_dir, subdir="fastapi")

    print("\n== Step 2: Embedding ==")
    #embedder = get_embedder("tfidf", dim=256)
    embedder = get_embedder("sentence-transformer")
    # embed on a rich text view: name + docstring + signature + source
    texts = [
        f"{c.qualified_name}\n{c.signature}\n{c.docstring}\n{c.source}"
        for c in chunks
    ]
    embedder.fit(texts)
    vectors = embedder.encode(texts)
    print(f"Embedded {len(vectors)} chunks, dim={vectors.shape[1]}")

    print("\n== Step 3: Storage ==")
    chunk_dicts = [c.__dict__ for c in chunks]
    build_store(db_path, chunk_dicts, vectors)
    print(f"Stored to {db_path}")

    # persist the embedder's fitted vectorizer so query-time encode() is consistent
    import pickle
    with open(Path(data_dir) / "embedder.pkl", "wb") as f:
        pickle.dump(embedder, f)
    print("Saved fitted embedder to data/embedder.pkl")


if __name__ == "__main__":
    main()
