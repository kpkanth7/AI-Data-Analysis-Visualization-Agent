import os
import re
import json
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

load_dotenv()

_BLOCKED = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE|EXEC)\b",
    re.IGNORECASE,
)


def get_db_url() -> str:
    url = os.getenv("DATABASE_URL", "")
    if not url:
        try:
            import streamlit as st
            url = st.secrets.get("DATABASE_URL", "")
        except Exception:
            url = ""
    return url or "postgresql://pradhyumnakasula@localhost:5432/data_analyst"


@lru_cache(maxsize=1)
def get_engine():
    return create_engine(get_db_url(), poolclass=NullPool)


@contextmanager
def get_connection():
    engine = get_engine()
    with engine.connect() as conn:
        yield conn


def slugify(name: str) -> str:
    name = re.sub(r"\.(csv|xlsx|xls)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    name = name[:63].strip("_").lower()
    if not name:
        name = "dataset"
    if name[0].isdigit():
        name = "t_" + name[:61]
    return name


MAX_QUERY_ROWS = 5000


def execute_sql(sql: str, params: dict | None = None) -> list[dict]:
    if _BLOCKED.search(sql):
        raise ValueError("Only SELECT statements are allowed.")
    if ";" in sql.strip().rstrip(";"):
        raise ValueError("Only SELECT statements are allowed.")
    with get_connection() as conn:
        result = conn.execute(text(sql), params or {})
        rows = [dict(row._mapping) for row in result.fetchmany(MAX_QUERY_ROWS + 1)]
    return rows


def list_datasets_from_db(
    include_owner_only: bool = False,
    session_id: str | None = None,
) -> list[dict]:
    """
    include_owner_only=True  → owner is logged in, show demo + owner-uploaded datasets
    session_id               → guest session id, show demo + that session's uploads
    """
    with get_connection() as conn:
        if include_owner_only:
            # Owner sees demo + all owner-uploaded (not guest-session datasets)
            result = conn.execute(text("""
                SELECT name, slug, row_count, columns_json, is_demo, owner_only, session_id
                FROM datasets
                WHERE is_demo = TRUE OR owner_only = TRUE
                ORDER BY is_demo DESC, uploaded_at DESC
            """))
        elif session_id:
            # Guest sees demo + their session uploads
            result = conn.execute(text("""
                SELECT name, slug, row_count, columns_json, is_demo, owner_only, session_id
                FROM datasets
                WHERE is_demo = TRUE OR session_id = :sid
                ORDER BY is_demo DESC, uploaded_at DESC
            """), {"sid": session_id})
        else:
            # No auth, no session — only demo
            result = conn.execute(text("""
                SELECT name, slug, row_count, columns_json, is_demo, owner_only, session_id
                FROM datasets
                WHERE is_demo = TRUE
                ORDER BY uploaded_at DESC
            """))
        return [dict(r._mapping) for r in result.fetchall()]


def register_dataset(
    name: str,
    slug: str,
    df: pd.DataFrame,
    profile: dict,
    is_demo: bool = False,
    owner_only: bool = False,
    session_id: str | None = None,
) -> None:
    columns_json = json.dumps([
        {"name": col, "dtype": str(df[col].dtype)} for col in df.columns
    ])
    profile_json = json.dumps(profile)
    with get_connection() as conn:
        conn.execute(text("""
            INSERT INTO datasets (name, slug, row_count, columns_json, profile_json, is_demo, owner_only, session_id)
            VALUES (:name, :slug, :row_count, CAST(:columns_json AS jsonb), CAST(:profile_json AS jsonb),
                    :is_demo, :owner_only, :session_id)
            ON CONFLICT (slug) DO UPDATE SET
                row_count = EXCLUDED.row_count,
                columns_json = EXCLUDED.columns_json,
                profile_json = EXCLUDED.profile_json,
                owner_only = EXCLUDED.owner_only,
                session_id = EXCLUDED.session_id
        """), {
            "name": name,
            "slug": slug,
            "row_count": len(df),
            "columns_json": columns_json,
            "profile_json": profile_json,
            "is_demo": is_demo,
            "owner_only": owner_only,
            "session_id": session_id,
        })
        conn.commit()


def delete_dataset(slug: str) -> None:
    """Remove dataset record, drop its PostgreSQL table, and purge Chroma entries."""
    # Fetch columns before deletion for Chroma cleanup
    columns: list[str] = []
    with get_connection() as conn:
        result = conn.execute(
            text("SELECT columns_json FROM datasets WHERE slug = :slug"),
            {"slug": slug},
        )
        row = result.fetchone()
        if row and row[0]:
            cols = row[0]
            if isinstance(cols, str):
                cols = json.loads(cols)
            columns = [c["name"] for c in cols if isinstance(c, dict) and "name" in c]
        conn.execute(text("DELETE FROM datasets WHERE slug = :slug"), {"slug": slug})
        # Safe: slug is validated by slugify() which only allows [a-z0-9_]
        conn.execute(text(f'DROP TABLE IF EXISTS "{slug}"'))
        conn.commit()
    # Purge Chroma — failures here are non-fatal
    if columns:
        try:
            from db.vector_store import remove_dataset_from_chroma
            remove_dataset_from_chroma(slug, columns)
        except Exception:
            pass


def delete_guest_session_datasets(session_id: str) -> None:
    """Clean up all datasets belonging to a guest session."""
    with get_connection() as conn:
        result = conn.execute(
            text("SELECT slug FROM datasets WHERE session_id = :sid"),
            {"sid": session_id},
        )
        slugs = [r[0] for r in result.fetchall()]
    for slug in slugs:
        try:
            delete_dataset(slug)
        except Exception:
            pass


def cleanup_old_guest_datasets(max_age_hours: int = 24) -> None:
    """Remove guest datasets older than max_age_hours. Run at app startup."""
    with get_connection() as conn:
        result = conn.execute(text("""
            SELECT slug FROM datasets
            WHERE session_id IS NOT NULL
              AND is_demo = FALSE
              AND owner_only = FALSE
              AND uploaded_at < NOW() - INTERVAL '1 hour' * :hours
        """), {"hours": max_age_hours})
        slugs = [r[0] for r in result.fetchall()]
    for slug in slugs:
        try:
            delete_dataset(slug)
        except Exception:
            pass


def table_exists(slug: str) -> bool:
    with get_connection() as conn:
        result = conn.execute(text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = :slug)"
        ), {"slug": slug})
        return result.scalar()


# ── Guest usage tracking ──────────────────────────────────────────────────────

def ensure_guest_usage_table() -> None:
    with get_connection() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS guest_usage (
                ip_hash     TEXT NOT NULL,
                usage_date  DATE NOT NULL DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')::DATE,
                queries     INTEGER NOT NULL DEFAULT 0,
                uploads     INTEGER NOT NULL DEFAULT 0,
                upload_bytes BIGINT NOT NULL DEFAULT 0,
                PRIMARY KEY (ip_hash, usage_date)
            )
        """))
        conn.commit()


def get_guest_usage(ip_hash: str) -> dict:
    with get_connection() as conn:
        result = conn.execute(text("""
            SELECT queries, uploads, upload_bytes
            FROM guest_usage
            WHERE ip_hash = :h AND usage_date = (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')::DATE
        """), {"h": ip_hash})
        row = result.fetchone()
        if row:
            return {"queries": row[0], "uploads": row[1], "upload_bytes": row[2]}
        return {"queries": 0, "uploads": 0, "upload_bytes": 0}


def increment_guest_query(ip_hash: str) -> None:
    with get_connection() as conn:
        conn.execute(text("""
            INSERT INTO guest_usage (ip_hash, usage_date, queries)
            VALUES (:h, (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')::DATE, 1)
            ON CONFLICT (ip_hash, usage_date) DO UPDATE
            SET queries = guest_usage.queries + 1
        """), {"h": ip_hash})
        conn.commit()


def increment_guest_upload(ip_hash: str, file_bytes: int) -> None:
    with get_connection() as conn:
        conn.execute(text("""
            INSERT INTO guest_usage (ip_hash, usage_date, uploads, upload_bytes)
            VALUES (:h, (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')::DATE, 1, :b)
            ON CONFLICT (ip_hash, usage_date) DO UPDATE
            SET uploads = guest_usage.uploads + 1,
                upload_bytes = guest_usage.upload_bytes + :b
        """), {"h": ip_hash, "b": file_bytes})
        conn.commit()
