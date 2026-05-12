#!/usr/bin/env python3
"""One-time setup: create database, catalog table, load demo datasets."""
import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

load_dotenv()

BASE_DB_URL = os.getenv("DATABASE_URL", "postgresql://pradhyumnakasula@localhost:5432/data_analyst")
PG_ROOT_URL = BASE_DB_URL.rsplit("/", 1)[0] + "/postgres"
DB_NAME = BASE_DB_URL.rsplit("/", 1)[-1]


def create_database():
    engine = create_engine(PG_ROOT_URL, isolation_level="AUTOCOMMIT", poolclass=NullPool)
    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": DB_NAME}
        ).fetchone()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{DB_NAME}"'))
            print(f"Created database: {DB_NAME}")
        else:
            print(f"Database {DB_NAME} already exists.")
    engine.dispose()


def create_tables():
    engine = create_engine(BASE_DB_URL, poolclass=NullPool)
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
                is_demo BOOLEAN DEFAULT FALSE
            )
        """))
        conn.commit()
    print("Tables created.")
    engine.dispose()


def load_demo_data():
    from db.postgres import slugify, register_dataset
    from core.dataset_profiler import profile_dataframe

    demo_dir = Path("data/demo")
    for csv_file in demo_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, parse_dates=True)
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except Exception:
                        pass
            slug = slugify(csv_file.name)
            profile = profile_dataframe(df)
            engine = create_engine(BASE_DB_URL, poolclass=NullPool)
            df.to_sql(slug, engine, if_exists="replace", index=False)
            engine.dispose()
            register_dataset(csv_file.name, slug, df, profile, is_demo=True)
            print(f"Loaded demo: {csv_file.name} → table '{slug}' ({len(df)} rows)")
        except Exception as e:
            print(f"Failed to load {csv_file.name}: {e}")


if __name__ == "__main__":
    create_database()
    create_tables()
    load_demo_data()
    print("\nSetup complete. Run: streamlit run app.py")
