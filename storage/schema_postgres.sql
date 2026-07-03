-- Production schema for Supabase (Postgres + pgvector).
-- Local dev/testing in this build uses an equivalent SQLite schema
-- (storage/db.py) since the sandbox has no Postgres server available.
-- The table shape and query patterns are identical either way.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,          -- function | method | class | module_docstring
    name            TEXT NOT NULL,
    qualified_name  TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    start_line      INTEGER,
    end_line        INTEGER,
    source          TEXT,
    docstring       TEXT,
    signature       TEXT,
    parent_class    TEXT,
    embedding       vector(384),            -- 384 for MiniLM; 768 if using Gemini embeddings
    search_vector   tsvector GENERATED ALWAYS AS (
                        to_tsvector('english', coalesce(name,'') || ' ' ||
                                     coalesce(docstring,'') || ' ' ||
                                     coalesce(signature,'') || ' ' ||
                                     coalesce(source,''))
                    ) STORED
);

-- vector similarity index (approximate nearest neighbor)
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- full-text search index
CREATE INDEX IF NOT EXISTS chunks_search_idx
    ON chunks USING GIN (search_vector);

CREATE INDEX IF NOT EXISTS chunks_file_path_idx ON chunks (file_path);
CREATE INDEX IF NOT EXISTS chunks_qualified_name_idx ON chunks (qualified_name);
