"""
Postgres + pgvector storage for documents and chunks.

DATABASE_URL can point at any Postgres with the pgvector extension available:
a local container, Supabase, Neon or Render Postgres.
"""
import os
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

SCHEMA = Path(__file__).with_name("schema.sql").read_text()

_pool: ConnectionPool | None = None


def init_db(url: str | None = None) -> None:
    """Create tables if needed, then open the connection pool."""
    global _pool
    url = url or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    # The extension must exist before register_vector runs on pooled connections.
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(SCHEMA)
    if _pool is None:
        # prepare_threshold=None: no server-side prepared statements, which transaction-mode
        # poolers (e.g. Neon's "-pooler" endpoint) can't track across connections.
        _pool = ConnectionPool(url, min_size=1, max_size=5, configure=register_vector,
                               kwargs={"prepare_threshold": None}, open=True)


def close_db() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def is_ready() -> bool:
    return _pool is not None


def _conn():
    if _pool is None:
        raise RuntimeError("Database not initialised")
    return _pool.connection()  # commits on success, rolls back on error


# ──────────────────────────── Documents ────────────────────────────

def create_document(user_id: str, embed_model: str, filename: str, sha256: str) -> str | None:
    """Insert a 'processing' record. Returns its id, or None if this user already uploaded this exact file."""
    with _conn() as conn:
        row = conn.execute(
            """INSERT INTO documents (user_id, embed_model, filename, sha256)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (user_id, embed_model, sha256) DO NOTHING
               RETURNING id""",
            (user_id, embed_model, filename, sha256),
        ).fetchone()
    return str(row[0]) if row else None


def document_exists(user_id: str, embed_model: str, sha256: str) -> bool:
    with _conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM documents WHERE user_id = %s AND embed_model = %s AND sha256 = %s",
            (user_id, embed_model, sha256),
        ).fetchone()
    return row is not None


def finish_document(doc_id: str, chunks: int, status: str, error: str | None = None) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE documents SET chunks = %s, status = %s, error = %s WHERE id = %s",
            (chunks, status, error, doc_id),
        )


def delete_document(doc_id: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))  # chunks follow (CASCADE)


def delete_user_document(user_id: str, embed_model: str, doc_id: str) -> str:
    """Delete one of this user's documents: 'deleted', 'processing' (left alone) or 'missing'."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT status FROM documents WHERE id = %s AND user_id = %s AND embed_model = %s FOR UPDATE",
            (doc_id, user_id, embed_model),
        ).fetchone()
        if row is None:
            return "missing"
        if row[0] == "processing":
            return "processing"  # its chunks are still being written
        conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))  # chunks follow (CASCADE)
    return "deleted"


def list_documents(user_id: str, embed_model: str) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            """SELECT id, filename, chunks, extract(epoch FROM created_at), status, error
               FROM documents WHERE user_id = %s AND embed_model = %s
               ORDER BY created_at""",
            (user_id, embed_model),
        ).fetchall()
    return [
        {"id": str(i), "filename": f, "chunks": c, "timestamp": float(ts), "status": s, "error": e}
        for i, f, c, ts, s, e in rows
    ]


# ──────────────────────────── Chunks ────────────────────────────

def insert_chunks(rows: list[tuple]) -> None:
    """rows: (id, document_id, user_id, embed_model, source, page, seq, content, embedding)."""
    with _conn() as conn:
        conn.cursor().executemany(
            """INSERT INTO chunks (id, document_id, user_id, embed_model, source, page, seq, content, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            rows,
        )


def document_chunks(user_id: str, embed_model: str, source: str) -> list[tuple]:
    """(id, page, content) for one document, in reading order."""
    with _conn() as conn:
        return conn.execute(
            """SELECT id, page, content FROM chunks
               WHERE user_id = %s AND embed_model = %s AND source = %s
               ORDER BY page NULLS FIRST, seq NULLS LAST, id""",
            (user_id, embed_model, source),
        ).fetchall()


def load_chunks(user_id: str, embed_model: str) -> list[tuple]:
    """(id, source, page, content) for every chunk this user has, used to build BM25."""
    with _conn() as conn:
        return conn.execute(
            "SELECT id, source, page, content FROM chunks WHERE user_id = %s AND embed_model = %s",
            (user_id, embed_model),
        ).fetchall()


def search_chunks(user_id: str, embed_model: str, query_vec, k: int) -> list[tuple]:
    """(id, source, page, content, cosine_distance) for the k closest chunks."""
    with _conn() as conn:
        return conn.execute(
            """SELECT id, source, page, content, embedding <=> %s AS dist
               FROM chunks WHERE user_id = %s AND embed_model = %s
               ORDER BY dist LIMIT %s""",
            (query_vec, user_id, embed_model, k),
        ).fetchall()
