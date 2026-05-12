import os
import re
import json
from contextlib import contextmanager
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
    return os.getenv("DATABASE_URL", "postgresql://pradhyumnakasula@localhost:5432/data_analyst")

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
    name = name.strip("_").lower()
    return name[:63]  # PG identifier limit

def execute_sql(sql: str, params: dict | None = None) -> list[dict]:
    if _BLOCKED.search(sql.strip()):
        raise ValueError("Only SELECT statements are allowed.")
    with get_connection() as conn:
        result = conn.execute(text(sql), params or {})
        rows = [dict(row._mapping) for row in result.fetchmany(1000)]
    return rows

def list_datasets_from_db() -> list[dict]:
    with get_connection() as conn:
        result = conn.execute(text(
            "SELECT name, slug, row_count, columns_json, is_demo FROM datasets ORDER BY uploaded_at DESC"
        ))
        return [dict(r._mapping) for r in result.fetchall()]

def register_dataset(name: str, slug: str, df: pd.DataFrame, profile: dict, is_demo: bool = False) -> None:
    columns_json = json.dumps([
        {"name": col, "dtype": str(df[col].dtype)} for col in df.columns
    ])
    profile_json = json.dumps(profile)
    with get_connection() as conn:
        conn.execute(text("""
            INSERT INTO datasets (name, slug, row_count, columns_json, profile_json, is_demo)
            VALUES (:name, :slug, :row_count, CAST(:columns_json AS jsonb), CAST(:profile_json AS jsonb), :is_demo)
            ON CONFLICT (slug) DO UPDATE SET
                row_count = EXCLUDED.row_count,
                columns_json = EXCLUDED.columns_json,
                profile_json = EXCLUDED.profile_json
        """), {
            "name": name,
            "slug": slug,
            "row_count": len(df),
            "columns_json": columns_json,
            "profile_json": profile_json,
            "is_demo": is_demo,
        })
        conn.commit()

def table_exists(slug: str) -> bool:
    with get_connection() as conn:
        result = conn.execute(text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = :slug)"
        ), {"slug": slug})
        return result.scalar()
