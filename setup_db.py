#!/usr/bin/env python3
"""One-time setup: create catalog tables, load demo datasets.

Works against any Postgres connection string set in DATABASE_URL (local, Supabase, Neon, etc).
Does NOT issue CREATE DATABASE — assumes the database already exists.
"""
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "")
if not DB_URL:
    raise SystemExit(
        "DATABASE_URL is not set. Add it to .env (local) or your Streamlit Cloud secrets."
    )


def create_tables():
    engine = create_engine(DB_URL, poolclass=NullPool)
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS datasets (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                slug VARCHAR(63) NOT NULL UNIQUE,
                row_count INTEGER,
                columns_json JSONB,
                profile_json JSONB,
                uploaded_at TIMESTAMP DEFAULT NOW(),
                is_demo BOOLEAN DEFAULT FALSE,
                owner_only BOOLEAN DEFAULT FALSE,
                session_id TEXT
            )
        """))
        # Backfill columns for older DBs that already have the table
        for col_sql in (
            "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS owner_only BOOLEAN DEFAULT FALSE",
            "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS session_id TEXT",
        ):
            try:
                conn.execute(text(col_sql))
            except Exception:
                pass

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS guest_usage (
                ip_hash      TEXT NOT NULL,
                usage_date   DATE NOT NULL DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')::DATE,
                queries      INTEGER NOT NULL DEFAULT 0,
                uploads      INTEGER NOT NULL DEFAULT 0,
                upload_bytes BIGINT  NOT NULL DEFAULT 0,
                PRIMARY KEY (ip_hash, usage_date)
            )
        """))
        conn.commit()
    print("Tables ready.")
    engine.dispose()


def load_demo_data():
    from db.postgres import slugify, register_dataset
    from core.dataset_profiler import profile_dataframe
    try:
        from db.vector_store import index_dataset_in_chroma
        _chroma_ok = True
    except Exception:
        _chroma_ok = False

    demo_dir = Path("data/demo")
    if not demo_dir.exists():
        print("No data/demo directory — skipping demo load.")
        return

    for csv_file in demo_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except Exception:
                        pass
            slug = slugify(csv_file.name)
            profile = profile_dataframe(df)
            engine = create_engine(DB_URL, poolclass=NullPool)
            df.to_sql(slug, engine, if_exists="replace", index=False)
            engine.dispose()
            register_dataset(csv_file.name, slug, df, profile, is_demo=True)
            if _chroma_ok:
                try:
                    index_dataset_in_chroma(slug, df.columns.tolist(), profile)
                except Exception as e:
                    print(f"  (Chroma index skipped for {slug}: {e})")
            print(f"Loaded demo: {csv_file.name} -> '{slug}' ({len(df)} rows)")
        except Exception as e:
            print(f"Failed to load {csv_file.name}: {e}")


if __name__ == "__main__":
    create_tables()
    load_demo_data()
    print("\nSetup complete. Run: streamlit run app.py")
