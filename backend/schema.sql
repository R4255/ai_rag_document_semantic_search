-- Nexus RAG storage: Postgres + pgvector.
-- Applied automatically at startup (db.init_db), safe to run again.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per uploaded file (replaces files.json).
CREATE TABLE IF NOT EXISTS documents (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id      text NOT NULL,
  embed_model  text NOT NULL,
  filename     text NOT NULL,
  sha256       text NOT NULL,              -- file fingerprint: the same file is never ingested twice
  chunks       int  NOT NULL DEFAULT 0,
  status       text NOT NULL DEFAULT 'processing',   -- processing | success | error
  error        text,
  created_at   timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, embed_model, sha256)
);

-- One row per chunk: vector and text together (replaces index.faiss + index.pkl).
CREATE TABLE IF NOT EXISTS chunks (
  id           uuid PRIMARY KEY,
  document_id  uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  user_id      text NOT NULL,
  embed_model  text NOT NULL,
  source       text NOT NULL,
  page         int,
  content      text NOT NULL,
  embedding    vector(768) NOT NULL        -- must match providers.EMBED_DIM
);

-- Every query is scoped to one user and one embedding model, so an exact scan over that
-- user's rows is fast. Add an HNSW index on embedding if single users grow very large.
CREATE INDEX IF NOT EXISTS chunks_user_model_idx ON chunks (user_id, embed_model);
CREATE INDEX IF NOT EXISTS documents_user_model_idx ON documents (user_id, embed_model);
