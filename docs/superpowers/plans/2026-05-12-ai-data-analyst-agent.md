# AI Data Analyst Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-quality AI data analyst Streamlit app powered by gpt-4o-mini, PostgreSQL, Chroma, and Plotly — with multi-dataset support, streaming agent steps, interactive charts, anomaly detection, and guest rate-limiting.

**Architecture:** Single Streamlit monolith. LangChain tool-calling agent runs inline. PostgreSQL (existing local PG18, database `data_analyst`) stores all datasets as tables. Chroma (embedded) indexes column metadata for semantic search. Plotly renders all charts with dark theme.

**Tech Stack:** Python 3.11, Streamlit 1.x, LangChain + langchain-openai, gpt-4o-mini, PostgreSQL 18, SQLAlchemy, Chroma, Plotly, pandas, scipy, openpyxl

---

## File Map

| File | Responsibility |
|---|---|
| `setup_db.py` | One-time: create DB, datasets catalog table, load demo CSVs |
| `requirements.txt` | All dependencies pinned |
| `.env.example` | Template for env vars |
| `.streamlit/secrets.toml.example` | Template for Streamlit secrets |
| `agent/schema.py` | Pydantic: AnalysisOutput, SubQueryResult |
| `agent/prompts.py` | System prompt + multi-subquery instructions |
| `agent/tools.py` | 10 LangChain tools |
| `agent/agent.py` | LangChain agent + StreamlitCallbackHandler |
| `db/postgres.py` | SQLAlchemy engine, dataset CRUD, safe SQL exec |
| `db/vector_store.py` | Chroma init, index columns, semantic search |
| `core/dataset_profiler.py` | Upload handler: dtype inference, PG ingestion, profiling |
| `core/anomaly.py` | Z-score + IQR anomaly detection, trend analysis |
| `core/exporter.py` | Excel/CSV export with openpyxl |
| `ui/charts.py` | Plotly figure builders (7 chart types) |
| `ui/sidebar.py` | Dataset manager, upload widget, guest quota display |
| `ui/chat.py` | Chat UI: message bubbles, streaming steps, inline charts |
| `app.py` | Streamlit entry: page config, auth, tab routing |
| `data/demo/` | ecommerce_sales.csv + global_superstore.csv |
| `tests/` | pytest suite |

---

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.streamlit/secrets.toml.example`
- Create: `agent/__init__.py`, `db/__init__.py`, `core/__init__.py`, `ui/__init__.py`
- Create: `tests/__init__.py`
- Create directories: `data/demo/`, `charts/`, `exports/`, `logs/`, `chroma_db/`

- [ ] **Step 1: Create requirements.txt**

```
streamlit>=1.35.0
langchain>=0.2.0
langchain-openai>=0.1.0
langchain-community>=0.2.0
openai>=1.30.0
psycopg2-binary>=2.9.9
sqlalchemy>=2.0.0
pandas>=2.2.0
numpy>=1.26.0
plotly>=5.22.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
openpyxl>=3.1.0
python-dotenv>=1.0.0
pydantic>=2.7.0
scipy>=1.13.0
pytest>=8.0.0
pytest-mock>=3.12.0
```

- [ ] **Step 2: Create .env.example**

```
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://pradhyumnakasula@localhost:5432/data_analyst
CHROMA_PERSIST_DIR=./chroma_db
```

- [ ] **Step 3: Create .streamlit/secrets.toml.example**

```toml
owner_password = "your-secret-password-here"
OPENAI_API_KEY = "sk-..."
DATABASE_URL = "postgresql://pradhyumnakasula@localhost:5432/data_analyst"
```

- [ ] **Step 4: Create package __init__.py files and directories**

```bash
mkdir -p agent db core ui tests data/demo charts exports logs chroma_db .streamlit
touch agent/__init__.py db/__init__.py core/__init__.py ui/__init__.py tests/__init__.py
cp .env.example .env
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

- [ ] **Step 5: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 6: Commit**

```bash
git init
git add requirements.txt .env.example .streamlit/secrets.toml.example agent/__init__.py db/__init__.py core/__init__.py ui/__init__.py tests/__init__.py
git commit -m "scaffold"
```

---

## Task 2: Database Setup

**Files:**
- Create: `setup_db.py`
- Create: `db/postgres.py`
- Create: `tests/test_postgres.py`

- [ ] **Step 1: Write failing test for postgres module**

`tests/test_postgres.py`:
```python
import pytest
from unittest.mock import patch, MagicMock
from db.postgres import get_engine, execute_sql, list_datasets_from_db, register_dataset

def test_execute_sql_blocks_write_statements():
    with pytest.raises(ValueError, match="Only SELECT"):
        execute_sql("INSERT INTO foo VALUES (1)")

def test_execute_sql_blocks_drop():
    with pytest.raises(ValueError, match="Only SELECT"):
        execute_sql("DROP TABLE users")

def test_execute_sql_blocks_update():
    with pytest.raises(ValueError, match="Only SELECT"):
        execute_sql("UPDATE users SET name='x'")

def test_register_dataset_returns_slug():
    import pandas as pd
    from unittest.mock import patch
    df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    with patch("db.postgres.get_engine") as mock_engine:
        mock_conn = MagicMock()
        mock_engine.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.return_value.__exit__ = MagicMock(return_value=False)
        # slug generation is pure function, test it directly
        from db.postgres import slugify
        assert slugify("My Sales Data 2024.csv") == "my_sales_data_2024"
        assert slugify("revenue-report.xlsx") == "revenue_report"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "/Users/pradhyumnakasula/data analysis agent" && python -m pytest tests/test_postgres.py -v 2>&1 | head -20
```

Expected: `ImportError` — module doesn't exist yet.

- [ ] **Step 3: Write db/postgres.py**

```python
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
```

- [ ] **Step 4: Write setup_db.py**

```python
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
            # Try to parse date-looking columns
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
```

- [ ] **Step 5: Run tests**

```bash
cd "/Users/pradhyumnakasula/data analysis agent" && python -m pytest tests/test_postgres.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add db/postgres.py setup_db.py tests/test_postgres.py
git commit -m "db setup"
```

---

## Task 3: Demo Datasets

**Files:**
- Create: `data/demo/ecommerce_sales.csv`
- Create: `data/demo/global_superstore.csv`

- [ ] **Step 1: Generate ecommerce_sales.csv**

```python
# Run this script once to generate demo data
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
n = 2000
dates = pd.date_range("2023-01-01", "2024-12-31", periods=n)
regions = np.random.choice(["North", "South", "East", "West"], n)
categories = np.random.choice(["Electronics", "Clothing", "Books", "Home", "Sports"], n)
products = {
    "Electronics": ["Laptop", "Phone", "Tablet", "Headphones"],
    "Clothing": ["T-Shirt", "Jeans", "Dress", "Jacket"],
    "Books": ["Fiction", "Non-Fiction", "Textbook", "Comic"],
    "Home": ["Lamp", "Chair", "Table", "Rug"],
    "Sports": ["Shoes", "Ball", "Racket", "Gloves"],
}
product_names = [np.random.choice(products[c]) for c in categories]
base_revenue = {"Electronics": 500, "Clothing": 80, "Books": 25, "Home": 150, "Sports": 60}
revenue = [
    round(base_revenue[c] * np.random.uniform(0.5, 2.5) + np.random.normal(0, 20), 2)
    for c in categories
]
revenue = [max(5.0, r) for r in revenue]
units = [max(1, int(r / base_revenue[c] * np.random.randint(1, 10))) for r, c in zip(revenue, categories)]

# inject some anomalies
for i in np.random.choice(n, 30, replace=False):
    revenue[i] *= np.random.choice([0.05, 8.0])

df = pd.DataFrame({
    "order_date": dates,
    "region": regions,
    "category": categories,
    "product": product_names,
    "revenue": revenue,
    "units": units,
    "discount": np.round(np.random.uniform(0, 0.4, n), 2),
    "profit": [round(r * np.random.uniform(0.05, 0.35), 2) for r in revenue],
})
Path("data/demo").mkdir(parents=True, exist_ok=True)
df.to_csv("data/demo/ecommerce_sales.csv", index=False)
print(f"Generated ecommerce_sales.csv: {len(df)} rows")
```

Run: `python scripts/gen_demo_data.py` (save script to `scripts/gen_demo_data.py` first)

- [ ] **Step 2: Download global_superstore.csv**

```bash
# Tableau's public Superstore dataset (widely available, no license issues)
curl -L "https://raw.githubusercontent.com/dsmorgan77/Superstore/master/Sample%20-%20Superstore.csv" \
  -o "data/demo/global_superstore.csv" 2>/dev/null || \
python3 -c "
import pandas as pd, numpy as np
np.random.seed(0)
n = 1500
df = pd.DataFrame({
    'order_date': pd.date_range('2021-01-01', periods=n, freq='8h'),
    'ship_date': pd.date_range('2021-01-05', periods=n, freq='8h'),
    'segment': np.random.choice(['Consumer','Corporate','Home Office'], n),
    'country': np.random.choice(['United States','Canada','Mexico'], n),
    'region': np.random.choice(['East','West','Central','South'], n),
    'category': np.random.choice(['Furniture','Office Supplies','Technology'], n),
    'sub_category': np.random.choice(['Chairs','Phones','Binders','Tables','Storage','Copiers'], n),
    'product_name': ['Product_'+str(i%200) for i in range(n)],
    'sales': np.round(np.random.exponential(250, n), 2),
    'quantity': np.random.randint(1, 15, n),
    'discount': np.round(np.random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5], n), 1),
    'profit': np.round(np.random.normal(50, 120, n), 2),
    'shipping_cost': np.round(np.random.exponential(15, n), 2),
})
df.to_csv('data/demo/global_superstore.csv', index=False)
print('Generated global_superstore.csv:', len(df), 'rows')
"
```

- [ ] **Step 3: Commit demo data**

```bash
git add data/demo/ scripts/
git commit -m "demo data"
```

---

## Task 4: Dataset Profiler

**Files:**
- Create: `core/dataset_profiler.py`
- Create: `tests/test_profiler.py`

- [ ] **Step 1: Write failing test**

`tests/test_profiler.py`:
```python
import pandas as pd
import numpy as np
import pytest
from core.dataset_profiler import profile_dataframe, ingest_dataframe

def test_profile_numeric_column():
    df = pd.DataFrame({"revenue": [100.0, 200.0, np.nan, 400.0]})
    profile = profile_dataframe(df)
    assert "revenue" in profile
    assert profile["revenue"]["null_pct"] == 25.0
    assert profile["revenue"]["mean"] == pytest.approx(233.33, rel=0.01)
    assert profile["revenue"]["dtype"] == "float64"

def test_profile_categorical_column():
    df = pd.DataFrame({"region": ["North", "South", "North", None]})
    profile = profile_dataframe(df)
    assert profile["region"]["unique_count"] == 2
    assert profile["region"]["null_pct"] == 25.0

def test_profile_date_column():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5)})
    profile = profile_dataframe(df)
    assert profile["date"]["is_datetime"] is True

def test_ingest_returns_slug(tmp_path):
    import io
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    csv_bytes = df.to_csv(index=False).encode()
    from unittest.mock import patch, MagicMock
    with patch("core.dataset_profiler.register_dataset"), \
         patch("core.dataset_profiler.get_engine") as mock_eng, \
         patch("core.dataset_profiler.index_dataset_in_chroma"):
        mock_eng.return_value = MagicMock()
        slug = ingest_dataframe(df, "my_data.csv")
    assert slug == "my_data"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_profiler.py -v 2>&1 | head -15
```

Expected: `ImportError`.

- [ ] **Step 3: Write core/dataset_profiler.py**

```python
import re
import pandas as pd
import numpy as np
from sqlalchemy.pool import NullPool

from db.postgres import get_engine, register_dataset, slugify
from db.vector_store import index_dataset_in_chroma


def profile_dataframe(df: pd.DataFrame) -> dict:
    profile = {}
    for col in df.columns:
        series = df[col]
        is_dt = pd.api.types.is_datetime64_any_dtype(series)
        is_num = pd.api.types.is_numeric_dtype(series) and not is_dt
        null_count = int(series.isna().sum())
        entry = {
            "dtype": str(series.dtype),
            "null_count": null_count,
            "null_pct": round(null_count / len(df) * 100, 2) if len(df) else 0.0,
            "unique_count": int(series.nunique()),
            "sample": [str(v) for v in series.dropna().head(3).tolist()],
            "is_datetime": is_dt,
        }
        if is_num:
            entry.update({
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "std": float(series.std()),
            })
        profile[col] = entry
    return profile


def infer_and_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object and ("date" in col.lower() or "time" in col.lower()):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
    return df


def ingest_dataframe(df: pd.DataFrame, filename: str, is_demo: bool = False) -> str:
    df = infer_and_parse_dates(df)
    slug = slugify(filename)
    profile = profile_dataframe(df)
    engine = get_engine()
    df.to_sql(slug, engine, if_exists="replace", index=False)
    engine.dispose()
    register_dataset(filename, slug, df, profile, is_demo=is_demo)
    index_dataset_in_chroma(slug, df.columns.tolist(), profile)
    return slug


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_profiler.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add core/dataset_profiler.py tests/test_profiler.py
git commit -m "profiler"
```

---

## Task 5: Chroma Vector Store

**Files:**
- Create: `db/vector_store.py`
- Create: `tests/test_vector_store.py`

- [ ] **Step 1: Write failing test**

`tests/test_vector_store.py`:
```python
import pytest
from unittest.mock import patch, MagicMock

def test_index_and_search():
    with patch("db.vector_store._get_collection") as mock_col:
        mock_instance = MagicMock()
        mock_col.return_value = mock_instance
        mock_instance.query.return_value = {
            "documents": [["table: sales, column: revenue, type: float64"]],
            "metadatas": [[{"table": "sales", "column": "revenue"}]],
            "distances": [[0.1]],
        }
        from db.vector_store import search_metadata
        results = search_metadata("revenue columns")
        assert isinstance(results, list)

def test_build_document_string():
    from db.vector_store import _build_doc
    profile_entry = {"dtype": "float64", "null_pct": 5.0, "unique_count": 100}
    doc = _build_doc("sales", "revenue", profile_entry)
    assert "sales" in doc
    assert "revenue" in doc
    assert "float64" in doc
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_vector_store.py -v 2>&1 | head -10
```

Expected: `ImportError`.

- [ ] **Step 3: Write db/vector_store.py**

```python
import os
from functools import lru_cache
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "dataset_metadata"
_EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_DIR)


def _get_collection():
    client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_EMBED_FN,
    )


def _build_doc(table: str, column: str, profile_entry: dict) -> str:
    dtype = profile_entry.get("dtype", "unknown")
    null_pct = profile_entry.get("null_pct", 0)
    unique = profile_entry.get("unique_count", 0)
    return (
        f"table: {table}, column: {column}, type: {dtype}, "
        f"null_pct: {null_pct}%, unique_values: {unique}"
    )


def index_dataset_in_chroma(table: str, columns: list[str], profile: dict) -> None:
    collection = _get_collection()
    ids, docs, metas = [], [], []
    for col in columns:
        doc_id = f"{table}::{col}"
        doc = _build_doc(table, col, profile.get(col, {}))
        ids.append(doc_id)
        docs.append(doc)
        metas.append({"table": table, "column": col})
    # upsert so re-indexing is idempotent
    collection.upsert(ids=ids, documents=docs, metadatas=metas)


def search_metadata(query: str, n_results: int = 10) -> list[dict]:
    collection = _get_collection()
    try:
        results = collection.query(query_texts=[query], n_results=min(n_results, collection.count() or 1))
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return [{"doc": d, "meta": m} for d, m in zip(docs, metas)]
    except Exception:
        return []


def remove_dataset_from_chroma(table: str, columns: list[str]) -> None:
    collection = _get_collection()
    ids = [f"{table}::{col}" for col in columns]
    collection.delete(ids=ids)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_vector_store.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add db/vector_store.py tests/test_vector_store.py
git commit -m "vector store"
```

---

## Task 6: Anomaly Detection & Trend Analysis

**Files:**
- Create: `core/anomaly.py`
- Create: `tests/test_anomaly.py`

- [ ] **Step 1: Write failing test**

`tests/test_anomaly.py`:
```python
import pandas as pd
import numpy as np
import pytest
from core.anomaly import detect_anomalies, detect_trends

def test_zscore_flags_outliers():
    normal = [100.0] * 50
    outliers = [5000.0, -5000.0]
    data = pd.DataFrame({"value": normal + outliers})
    result = detect_anomalies(data, "value", method="zscore")
    assert result["is_anomaly"].sum() == 2

def test_iqr_flags_outliers():
    data = pd.DataFrame({"value": list(range(100)) + [9999]})
    result = detect_anomalies(data, "value", method="iqr")
    assert result["is_anomaly"].sum() >= 1

def test_detect_trends_returns_growth_rate():
    dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    revenue = [100, 110, 120, 115, 130, 140, 135, 150, 160, 155, 170, 180]
    df = pd.DataFrame({"date": dates, "revenue": revenue})
    result = detect_trends(df, "date", "revenue")
    assert "rolling_avg" in result.columns
    assert "growth_rate" in result.columns
    assert result["rolling_avg"].notna().any()

def test_anomaly_result_has_severity():
    data = pd.DataFrame({"value": [100.0] * 48 + [50000.0, -50000.0]})
    result = detect_anomalies(data, "value")
    anomalies = result[result["is_anomaly"]]
    assert "severity" in anomalies.columns
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_anomaly.py -v 2>&1 | head -10
```

Expected: `ImportError`.

- [ ] **Step 3: Write core/anomaly.py**

```python
import pandas as pd
import numpy as np
from scipy import stats


def detect_anomalies(
    df: pd.DataFrame,
    column: str,
    method: str = "both",
    zscore_threshold: float = 3.0,
) -> pd.DataFrame:
    result = df.copy()
    series = pd.to_numeric(result[column], errors="coerce")
    
    zscore_mask = pd.Series([False] * len(series), index=series.index)
    iqr_mask = pd.Series([False] * len(series), index=series.index)

    if method in ("zscore", "both"):
        z = np.abs(stats.zscore(series.dropna()))
        z_full = pd.Series(np.nan, index=series.index)
        z_full[series.notna()] = z
        zscore_mask = z_full > zscore_threshold

    if method in ("iqr", "both"):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        iqr_mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)

    is_anomaly = zscore_mask | iqr_mask
    result["is_anomaly"] = is_anomaly.fillna(False)
    
    # Severity: how many std devs from mean
    mean, std = series.mean(), series.std()
    if std > 0:
        result["severity"] = ((series - mean).abs() / std).round(2)
    else:
        result["severity"] = 0.0

    return result


def detect_trends(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    window: int = 7,
) -> pd.DataFrame:
    result = df.copy().sort_values(date_col).reset_index(drop=True)
    result["rolling_avg"] = (
        pd.to_numeric(result[value_col], errors="coerce")
        .rolling(window=window, min_periods=1)
        .mean()
        .round(2)
    )
    result["growth_rate"] = (
        pd.to_numeric(result[value_col], errors="coerce")
        .pct_change()
        .mul(100)
        .round(2)
    )
    return result
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_anomaly.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add core/anomaly.py tests/test_anomaly.py
git commit -m "anomaly detection"
```

---

## Task 7: Exporter

**Files:**
- Create: `core/exporter.py`
- Create: `tests/test_exporter.py`

- [ ] **Step 1: Write failing test**

`tests/test_exporter.py`:
```python
import pandas as pd
import pytest
import os
from core.exporter import export_to_excel, export_to_csv

def test_export_excel_creates_file(tmp_path):
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    path = tmp_path / "test_export.xlsx"
    result = export_to_excel(df, str(path))
    assert os.path.exists(result)
    loaded = pd.read_excel(result)
    assert list(loaded.columns) == ["col1", "col2"]
    assert len(loaded) == 3

def test_export_csv_creates_file(tmp_path):
    df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
    path = tmp_path / "test.csv"
    result = export_to_csv(df, str(path))
    assert os.path.exists(result)

def test_export_excel_styled(tmp_path):
    df = pd.DataFrame({"revenue": [1000.5, 2000.75], "region": ["North", "South"]})
    path = tmp_path / "styled.xlsx"
    result = export_to_excel(df, str(path), title="Revenue Report")
    assert os.path.exists(result)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_exporter.py -v 2>&1 | head -10
```

Expected: `ImportError`.

- [ ] **Step 3: Write core/exporter.py**

```python
import os
from datetime import datetime
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

EXPORTS_DIR = "exports"


def _ensure_exports_dir():
    os.makedirs(EXPORTS_DIR, exist_ok=True)


def export_to_excel(df: pd.DataFrame, path: str | None = None, title: str = "Data Export") -> str:
    _ensure_exports_dir()
    if path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(EXPORTS_DIR, f"export_{ts}.xlsx")

    df.to_excel(path, index=False, engine="openpyxl")

    wb = openpyxl.load_workbook(path)
    ws = wb.active
    ws.title = "Data"

    header_fill = PatternFill(start_color="1E3A5F", end_color="1E3A5F", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True, size=11)
    border = Border(
        bottom=Side(style="thin", color="CCCCCC"),
        right=Side(style="thin", color="EEEEEE"),
    )

    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.border = border
            cell.alignment = Alignment(vertical="center")

    for col_idx, col in enumerate(df.columns, 1):
        max_len = max(len(str(col)), df[col].astype(str).str.len().max() if len(df) > 0 else 0)
        ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 4, 40)

    ws.freeze_panes = "A2"
    wb.save(path)
    return path


def export_to_csv(df: pd.DataFrame, path: str | None = None) -> str:
    _ensure_exports_dir()
    if path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(EXPORTS_DIR, f"export_{ts}.csv")
    df.to_csv(path, index=False)
    return path
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_exporter.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add core/exporter.py tests/test_exporter.py
git commit -m "exporter"
```

---

## Task 8: Pydantic Schema + System Prompt

**Files:**
- Create: `agent/schema.py`
- Create: `agent/prompts.py`
- Create: `tests/test_schema.py`

- [ ] **Step 1: Write failing test**

`tests/test_schema.py`:
```python
from agent.schema import AnalysisOutput, SubQueryResult

def test_analysis_output_defaults():
    out = AnalysisOutput(answer="Revenue is $1M")
    assert out.answer == "Revenue is $1M"
    assert out.sub_results == []
    assert out.export_path is None

def test_sub_query_result():
    sub = SubQueryResult(index=1, question="total revenue", answer="$1M")
    assert sub.index == 1
    assert sub.chart_type is None

def test_analysis_output_with_sub_results():
    subs = [SubQueryResult(index=1, question="q1", answer="a1", chart_type="bar")]
    out = AnalysisOutput(answer="Combined answer", sub_results=subs)
    assert len(out.sub_results) == 1
    assert out.sub_results[0].chart_type == "bar"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_schema.py -v 2>&1 | head -10
```

Expected: `ImportError`.

- [ ] **Step 3: Write agent/schema.py**

```python
from pydantic import BaseModel, Field
from typing import Any, Optional


class SubQueryResult(BaseModel):
    index: int
    question: str
    answer: str
    chart_type: Optional[str] = None
    chart_config: Optional[dict[str, Any]] = None
    sql_used: Optional[str] = None
    export_path: Optional[str] = None
    data_preview: list[dict[str, Any]] = Field(default_factory=list)


class AnalysisOutput(BaseModel):
    answer: str
    sub_results: list[SubQueryResult] = Field(default_factory=list)
    datasets_used: list[str] = Field(default_factory=list)
    export_path: Optional[str] = None
    queries_used: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Write agent/prompts.py**

```python
SYSTEM_PROMPT = """\
You are an expert AI Data Analyst with access to a PostgreSQL database.

## Mandatory First Step
ALWAYS call `list_datasets` at the start of EVERY conversation turn to know what tables exist.
Use `search_metadata` to discover which columns are relevant before writing SQL.

## Tool Selection Guide
- Numeric summary (sum/avg/count) → `compute_stats`
- Filtered rows, joins, groupby → `query_sql`
- Complex pandas expressions → `query_pandas`
- Trend over time → `detect_trends`
- Outliers / anomalies → `detect_anomalies`
- Any visual request OR whenever you have tabular results → `create_visualization`
- User asks to download / export → `export_data`
- End of response → `save_session`

## Multi-Subquery Protocol
If the user's message contains multiple independent requests (signalled by: AND, ALSO, PLUS,
multiple question marks, or clearly distinct topics):
1. State: "I'll answer [N] questions:"
2. Number each sub-task: [1], [2], ...
3. Execute tools for each sub-task fully before moving to the next
4. Label each result block with [1], [2], ...
5. Produce a separate chart per sub-task where meaningful
6. Finish with a brief combined summary

## SQL Rules
- SELECT only — never INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE
- Always include LIMIT 1000 unless the query is a pure aggregation
- Use exact table slugs from `list_datasets`
- PostgreSQL dialect only

## Chart Selection
- Time series data → line
- Category comparison → bar
- Distributions → histogram
- Relationships between two numeric cols → scatter
- Cross-tab or correlation matrix → heatmap
- Part-of-whole percentages → pie
- Anomaly results → anomaly (scatter with red outliers)

## Output Format
Return ONLY valid JSON matching this schema:
{format_instructions}
"""
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_schema.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add agent/schema.py agent/prompts.py tests/test_schema.py
git commit -m "schema and prompts"
```

---

## Task 9: Agent Tools

**Files:**
- Create: `agent/tools.py`
- Create: `tests/test_tools.py`

- [ ] **Step 1: Write failing test**

`tests/test_tools.py`:
```python
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

def test_list_datasets_tool():
    with patch("agent.tools.list_datasets_from_db") as mock_db:
        mock_db.return_value = [{"name": "sales.csv", "slug": "sales", "row_count": 100}]
        from agent.tools import list_datasets
        result = list_datasets.invoke({})
        assert "sales" in result

def test_compute_stats_tool():
    with patch("agent.tools.execute_sql") as mock_sql:
        mock_sql.return_value = [{"revenue": 100}, {"revenue": 200}, {"revenue": 300}]
        from agent.tools import compute_stats
        result = compute_stats.invoke({"table": "sales", "column": "revenue", "metrics": ["sum", "avg"]})
        assert "sum" in result or "600" in result

def test_query_sql_blocks_unsafe():
    from agent.tools import query_sql
    with patch("agent.tools.execute_sql") as mock_sql:
        mock_sql.side_effect = ValueError("Only SELECT")
        result = query_sql.invoke({"sql": "DROP TABLE users"})
        assert "error" in result.lower() or "only select" in result.lower()

def test_search_metadata_tool():
    with patch("agent.tools.search_metadata") as mock_search:
        mock_search.return_value = [{"doc": "table: sales, column: revenue", "meta": {"table": "sales"}}]
        from agent.tools import search_metadata_tool
        result = search_metadata_tool.invoke({"query": "revenue columns"})
        assert "sales" in result
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_tools.py -v 2>&1 | head -10
```

Expected: `ImportError`.

- [ ] **Step 3: Write agent/tools.py**

```python
import json
import traceback
from typing import Any

import pandas as pd
from langchain.tools import tool

from db.postgres import execute_sql, list_datasets_from_db, get_engine
from db.vector_store import search_metadata
from core.anomaly import detect_anomalies, detect_trends
from core.exporter import export_to_excel


@tool
def list_datasets(dummy: str = "") -> str:
    """List all available datasets (tables) with their schemas. Call this first."""
    try:
        datasets = list_datasets_from_db()
        if not datasets:
            return "No datasets loaded yet. Ask the user to upload a CSV file."
        lines = []
        for ds in datasets:
            cols = ds.get("columns_json") or []
            if isinstance(cols, str):
                import json as _json
                cols = _json.loads(cols)
            col_summary = ", ".join(f"{c['name']} ({c['dtype']})" for c in cols[:10])
            demo = " [demo]" if ds.get("is_demo") else ""
            lines.append(f"• {ds['slug']}{demo}: {ds['row_count']} rows | columns: {col_summary}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing datasets: {e}"


@tool
def search_metadata_tool(query: str) -> str:
    """Semantic search over column names and table metadata. Use to find relevant columns before writing SQL."""
    try:
        results = search_metadata(query, n_results=8)
        if not results:
            return "No matching columns found."
        return "\n".join(r["doc"] for r in results)
    except Exception as e:
        return f"Error searching metadata: {e}"


@tool
def query_sql(sql: str) -> str:
    """Run a SELECT SQL query against PostgreSQL. Returns up to 1000 rows as JSON."""
    try:
        rows = execute_sql(sql)
        if not rows:
            return "Query returned no results."
        preview = rows[:5]
        total = len(rows)
        return json.dumps({"total_rows": total, "preview": preview, "all_rows": rows}, default=str)
    except Exception as e:
        return f"SQL error: {e}"


@tool
def query_pandas(table: str, expression: str) -> str:
    """Run a pandas .query() expression on a table. Good for complex string/date filters."""
    try:
        engine = get_engine()
        df = pd.read_sql_table(table, engine)
        engine.dispose()
        result = df.query(expression)
        rows = result.head(100).to_dict(orient="records")
        return json.dumps({"total_rows": len(result), "preview": rows[:5], "all_rows": rows}, default=str)
    except Exception as e:
        return f"Pandas query error: {e}"


@tool
def compute_stats(table: str, column: str, metrics: list[str] = None) -> str:
    """Compute statistics (sum, avg, median, std, min, max, count) on a numeric column."""
    if metrics is None:
        metrics = ["sum", "avg", "min", "max", "count", "std"]
    try:
        engine = get_engine()
        df = pd.read_sql_table(table, engine)
        engine.dispose()
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        result = {}
        metric_map = {
            "sum": lambda s: float(s.sum()),
            "avg": lambda s: float(s.mean()),
            "median": lambda s: float(s.median()),
            "std": lambda s: float(s.std()),
            "min": lambda s: float(s.min()),
            "max": lambda s: float(s.max()),
            "count": lambda s: int(s.count()),
        }
        for m in metrics:
            if m in metric_map:
                result[m] = metric_map[m](series)
        return json.dumps({"column": column, "table": table, "stats": result}, default=str)
    except Exception as e:
        return f"Stats error: {e}"


@tool
def detect_anomalies_tool(table: str, column: str, method: str = "both") -> str:
    """Detect anomalies/outliers in a numeric column using z-score and IQR methods."""
    try:
        engine = get_engine()
        df = pd.read_sql_table(table, engine)
        engine.dispose()
        result = detect_anomalies(df, column, method=method)
        anomalies = result[result["is_anomaly"]].head(50)
        normal_count = (~result["is_anomaly"]).sum()
        anomaly_count = result["is_anomaly"].sum()
        return json.dumps({
            "total_rows": len(result),
            "anomaly_count": int(anomaly_count),
            "normal_count": int(normal_count),
            "anomaly_pct": round(anomaly_count / len(result) * 100, 2),
            "anomalies": anomalies.to_dict(orient="records"),
            "full_data_for_chart": result[["is_anomaly", column, "severity"]].to_dict(orient="records"),
        }, default=str)
    except Exception as e:
        return f"Anomaly detection error: {e}"


@tool
def detect_trends_tool(table: str, date_col: str, value_col: str, window: int = 7) -> str:
    """Detect trends in time-series data. Returns rolling average and growth rates."""
    try:
        engine = get_engine()
        df = pd.read_sql_table(table, engine)
        engine.dispose()
        result = detect_trends(df, date_col, value_col, window=window)
        summary = {
            "total_periods": len(result),
            "avg_growth_rate": float(result["growth_rate"].mean()),
            "max_growth_rate": float(result["growth_rate"].max()),
            "min_growth_rate": float(result["growth_rate"].min()),
            "data": result[[date_col, value_col, "rolling_avg", "growth_rate"]].head(200).to_dict(orient="records"),
        }
        return json.dumps(summary, default=str)
    except Exception as e:
        return f"Trend detection error: {e}"


@tool
def create_visualization(
    chart_type: str,
    data: list[dict],
    x: str,
    y: str,
    title: str = "",
    color: str = None,
    anomaly_col: str = None,
) -> str:
    """
    Create a Plotly chart specification.
    chart_type: line | bar | scatter | histogram | heatmap | pie | anomaly
    Returns JSON config that the UI will render.
    """
    config = {
        "chart_type": chart_type,
        "data": data[:500],  # limit payload
        "x": x,
        "y": y,
        "title": title,
        "color": color,
        "anomaly_col": anomaly_col,
    }
    return json.dumps(config, default=str)


@tool
def export_data(table: str, sql: str = None) -> str:
    """Export a table or SQL query result to a styled Excel file. Returns the file path."""
    try:
        engine = get_engine()
        if sql:
            df = pd.read_sql(sql, engine)
        else:
            df = pd.read_sql_table(table, engine)
        engine.dispose()
        path = export_to_excel(df, title=f"{table} Export")
        return json.dumps({"export_path": path, "rows": len(df), "columns": list(df.columns)})
    except Exception as e:
        return f"Export error: {e}"


@tool
def save_session(conversation: str) -> str:
    """Save the current conversation and results to a JSON log file."""
    import os, datetime
    os.makedirs("logs", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"logs/session_{ts}.txt"
    with open(path, "w") as f:
        f.write(conversation)
    return json.dumps({"log_path": path})


ALL_TOOLS = [
    list_datasets,
    search_metadata_tool,
    query_sql,
    query_pandas,
    compute_stats,
    detect_anomalies_tool,
    detect_trends_tool,
    create_visualization,
    export_data,
    save_session,
]
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_tools.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/tools.py tests/test_tools.py
git commit -m "agent tools"
```

---

## Task 10: LangChain Agent with Streaming

**Files:**
- Create: `agent/agent.py`

- [ ] **Step 1: Write agent/agent.py**

```python
import os
import json
from typing import Generator

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain.agents import create_tool_calling_agent, AgentExecutor

from agent.tools import ALL_TOOLS
from agent.prompts import SYSTEM_PROMPT
from agent.schema import AnalysisOutput

load_dotenv()


class StreamlitStepHandler(BaseCallbackHandler):
    """Streams agent tool calls live into a Streamlit container."""

    def __init__(self, container):
        self.container = container
        self.steps: list[str] = []

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        name = serialized.get("name", "tool")
        self.steps.append(f"⏳ `{name}` ← {str(input_str)[:120]}")
        self.container.markdown("\n\n".join(self.steps))

    def on_tool_end(self, output: str, **kwargs):
        if self.steps:
            last = self.steps[-1]
            # replace spinner with checkmark
            self.steps[-1] = last.replace("⏳", "✅")
            try:
                parsed = json.loads(output)
                if "total_rows" in parsed:
                    self.steps[-1] += f" → {parsed['total_rows']} rows"
                elif "anomaly_count" in parsed:
                    self.steps[-1] += f" → {parsed['anomaly_count']} anomalies"
                elif "export_path" in parsed:
                    self.steps[-1] += f" → saved to `{parsed['export_path']}`"
            except Exception:
                pass
        self.container.markdown("\n\n".join(self.steps))

    def on_agent_finish(self, finish, **kwargs):
        self.steps.append("🏁 Agent finished")
        self.container.markdown("\n\n".join(self.steps))


def build_agent() -> AgentExecutor:
    parser = PydanticOutputParser(pydantic_object=AnalysisOutput)
    format_instructions = parser.get_format_instructions()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", ""),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]).partial(format_instructions=format_instructions)

    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=False,
        return_intermediate_steps=True,
        max_iterations=15,
    )


def run_agent(
    query: str,
    chat_history: list,
    step_container,
) -> AnalysisOutput | None:
    executor = build_agent()
    handler = StreamlitStepHandler(step_container)

    result = executor.invoke(
        {"input": query, "chat_history": chat_history},
        config={"callbacks": [handler]},
    )

    output_text = result.get("output", "")
    parser = PydanticOutputParser(pydantic_object=AnalysisOutput)
    try:
        return parser.parse(output_text)
    except Exception:
        # Fallback: wrap raw output
        return AnalysisOutput(answer=output_text)
```

- [ ] **Step 2: Smoke-test agent builds without error**

```bash
cd "/Users/pradhyumnakasula/data analysis agent" && python -c "
import os; os.environ['OPENAI_API_KEY'] = 'sk-test'
import unittest.mock as mock
with mock.patch('streamlit.secrets', {}):
    from agent.agent import build_agent
    print('Agent built OK')
"
```

Expected: `Agent built OK`

- [ ] **Step 3: Commit**

```bash
git add agent/agent.py
git commit -m "langchain agent"
```

---

## Task 11: Plotly Chart Builders

**Files:**
- Create: `ui/charts.py`
- Create: `tests/test_charts.py`

- [ ] **Step 1: Write failing test**

`tests/test_charts.py`:
```python
import pandas as pd
import pytest
from ui.charts import build_chart

def test_line_chart_returns_figure():
    import plotly.graph_objects as go
    data = [{"date": "2024-01", "revenue": 100}, {"date": "2024-02", "revenue": 200}]
    fig = build_chart({"chart_type": "line", "data": data, "x": "date", "y": "revenue", "title": "Test"})
    assert isinstance(fig, go.Figure)

def test_bar_chart_returns_figure():
    import plotly.graph_objects as go
    data = [{"region": "North", "sales": 500}, {"region": "South", "sales": 300}]
    fig = build_chart({"chart_type": "bar", "data": data, "x": "region", "y": "sales", "title": "By Region"})
    assert isinstance(fig, go.Figure)

def test_unknown_chart_type_raises():
    with pytest.raises(ValueError, match="Unknown chart type"):
        build_chart({"chart_type": "radar", "data": [], "x": "a", "y": "b", "title": ""})

def test_histogram_chart():
    import plotly.graph_objects as go
    data = [{"value": i} for i in range(50)]
    fig = build_chart({"chart_type": "histogram", "data": data, "x": "value", "y": "value", "title": "Dist"})
    assert isinstance(fig, go.Figure)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_charts.py -v 2>&1 | head -10
```

Expected: `ImportError`.

- [ ] **Step 3: Write ui/charts.py**

```python
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TEMPLATE = "plotly_dark"
COLORS = px.colors.qualitative.Plotly


def build_chart(config: dict | str) -> go.Figure:
    if isinstance(config, str):
        config = json.loads(config)

    chart_type = config.get("chart_type", "bar")
    data = config.get("data", [])
    x = config.get("x")
    y = config.get("y")
    title = config.get("title", "")
    color = config.get("color")
    anomaly_col = config.get("anomaly_col")

    df = pd.DataFrame(data) if data else pd.DataFrame()

    if chart_type == "line":
        return _line(df, x, y, title, color)
    elif chart_type == "bar":
        return _bar(df, x, y, title, color)
    elif chart_type == "scatter":
        return _scatter(df, x, y, title, color)
    elif chart_type == "histogram":
        return _histogram(df, x, title)
    elif chart_type == "heatmap":
        return _heatmap(df, title)
    elif chart_type == "pie":
        return _pie(df, x, y, title)
    elif chart_type == "anomaly":
        return _anomaly(df, x, y, anomaly_col, title)
    elif chart_type == "box":
        return _box(df, x, y, title)
    else:
        raise ValueError(f"Unknown chart type: {chart_type}")


def _common_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template=TEMPLATE,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        hoverlabel=dict(bgcolor="#1e1e2e", font_size=12),
    )
    return fig


def _line(df, x, y, title, color=None):
    fig = px.line(df, x=x, y=y, color=color, title=title, template=TEMPLATE, markers=True)
    fig.update_xaxes(rangeslider_visible=len(df) > 30)
    return _common_layout(fig, title)


def _bar(df, x, y, title, color=None):
    barmode = "group" if color else "relative"
    fig = px.bar(df, x=x, y=y, color=color, title=title, template=TEMPLATE,
                 barmode=barmode, text_auto=".2s")
    fig.update_traces(textposition="outside")
    return _common_layout(fig, title)


def _scatter(df, x, y, title, color=None):
    fig = px.scatter(df, x=x, y=y, color=color, title=title, template=TEMPLATE,
                     trendline="ols", hover_data=df.columns.tolist()[:6])
    return _common_layout(fig, title)


def _histogram(df, x, title):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Distribution", "Box Plot"))
    fig.add_trace(go.Histogram(x=df[x], name="Histogram", marker_color=COLORS[0]), row=1, col=1)
    fig.add_trace(go.Box(y=df[x], name="Box", marker_color=COLORS[1]), row=1, col=2)
    fig.update_layout(title=title, template=TEMPLATE, showlegend=False,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def _heatmap(df, title):
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return go.Figure().update_layout(title="No numeric columns for heatmap", template=TEMPLATE)
    corr = numeric.corr().round(2)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        hovertemplate="%{x} × %{y}: %{z}<extra></extra>",
    ))
    return _common_layout(fig, title)


def _pie(df, names, values, title):
    fig = px.pie(df, names=names, values=values, title=title, template=TEMPLATE,
                 hole=0.35, color_discrete_sequence=COLORS)
    fig.update_traces(textinfo="percent+label", pull=[0.03] * len(df))
    return _common_layout(fig, title)


def _anomaly(df, x, y, anomaly_col, title):
    if anomaly_col and anomaly_col in df.columns:
        normal = df[~df[anomaly_col].astype(bool)]
        anomalies = df[df[anomaly_col].astype(bool)]
    else:
        normal, anomalies = df, pd.DataFrame()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal[x] if x in normal.columns else normal.index,
        y=normal[y],
        mode="markers",
        name="Normal",
        marker=dict(color="#636EFA", size=6, opacity=0.7),
    ))
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies[x] if x in anomalies.columns else anomalies.index,
            y=anomalies[y],
            mode="markers",
            name="Anomaly",
            marker=dict(color="#EF553B", size=12, symbol="x", line=dict(width=2)),
        ))
    return _common_layout(fig, title)


def _box(df, x, y, title):
    fig = px.box(df, x=x, y=y, title=title, template=TEMPLATE,
                 color=x, color_discrete_sequence=COLORS)
    return _common_layout(fig, title)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_charts.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ui/charts.py tests/test_charts.py
git commit -m "plotly charts"
```

---

## Task 12: Sidebar UI

**Files:**
- Create: `ui/sidebar.py`

- [ ] **Step 1: Write ui/sidebar.py**

```python
import streamlit as st
import pandas as pd
from db.postgres import list_datasets_from_db
from core.dataset_profiler import read_uploaded_file, ingest_dataframe

GUEST_QUERY_LIMIT = 5


def init_session_state():
    defaults = {
        "is_owner": False,
        "guest_queries": 0,
        "chat_history": [],
        "lc_history": [],
        "active_dataset": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def check_owner_auth() -> bool:
    if st.session_state.get("is_owner"):
        return True
    with st.sidebar.expander("🔐 Owner Login", expanded=False):
        pw = st.text_input("Password", type="password", key="owner_pw_input")
        if st.button("Login", key="owner_login_btn"):
            expected = st.secrets.get("owner_password", "")
            if pw and pw == expected:
                st.session_state["is_owner"] = True
                st.rerun()
            else:
                st.error("Wrong password")
    return False


def can_query() -> bool:
    if st.session_state.get("is_owner"):
        return True
    return st.session_state.get("guest_queries", 0) < GUEST_QUERY_LIMIT


def increment_query_count():
    if not st.session_state.get("is_owner"):
        st.session_state["guest_queries"] = st.session_state.get("guest_queries", 0) + 1


def render_sidebar():
    init_session_state()

    st.sidebar.title("📊 Data Analyst AI")
    st.sidebar.markdown("---")

    is_owner = check_owner_auth()

    # Role badge
    if is_owner:
        st.sidebar.success("👑 Owner mode — unlimited queries")
    else:
        used = st.session_state.get("guest_queries", 0)
        remaining = GUEST_QUERY_LIMIT - used
        if remaining > 0:
            st.sidebar.info(f"👤 Guest — {remaining}/{GUEST_QUERY_LIMIT} queries remaining")
        else:
            st.sidebar.error("⛔ Guest limit reached. Reload for new session.")

    st.sidebar.markdown("---")

    # Dataset upload (owner only)
    if is_owner:
        st.sidebar.subheader("📁 Upload Dataset")
        uploaded = st.sidebar.file_uploader(
            "CSV or Excel", type=["csv", "xlsx", "xls"], key="file_uploader"
        )
        if uploaded:
            with st.sidebar.status(f"Ingesting {uploaded.name}..."):
                try:
                    df = read_uploaded_file(uploaded)
                    slug = ingest_dataframe(df, uploaded.name)
                    st.sidebar.success(f"✅ Loaded `{slug}` ({len(df):,} rows)")
                except Exception as e:
                    st.sidebar.error(f"Upload failed: {e}")
        st.sidebar.markdown("---")

    # Loaded datasets
    st.sidebar.subheader("🗃️ Datasets")
    try:
        datasets = list_datasets_from_db()
    except Exception:
        datasets = []

    if not datasets:
        st.sidebar.caption("No datasets loaded yet.")
    else:
        for ds in datasets:
            label = f"{'★ ' if ds.get('is_demo') else ''}{ds['name']}"
            with st.sidebar.expander(label):
                st.caption(f"Table: `{ds['slug']}` · {ds['row_count']:,} rows")
                cols = ds.get("columns_json") or []
                if isinstance(cols, str):
                    import json
                    cols = json.loads(cols)
                if cols:
                    col_df = pd.DataFrame(cols)
                    st.dataframe(col_df, hide_index=True, use_container_width=True)

    # Demo datasets section for guests
    if not is_owner:
        demo_datasets = [d for d in datasets if d.get("is_demo")]
        if demo_datasets:
            st.sidebar.markdown("---")
            st.sidebar.subheader("💡 Demo Data Available")
            for ds in demo_datasets:
                st.sidebar.caption(f"**{ds['slug']}** — try asking:")
                st.sidebar.markdown(
                    f"- *Total revenue by region?*\n"
                    f"- *Show sales trend over time*\n"
                    f"- *Detect anomalies in revenue AND show top 5 products*"
                )
```

- [ ] **Step 2: Commit**

```bash
git add ui/sidebar.py
git commit -m "sidebar ui"
```

---

## Task 13: Chat UI

**Files:**
- Create: `ui/chat.py`

- [ ] **Step 1: Write ui/chat.py**

```python
import json
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from agent.agent import run_agent
from agent.schema import AnalysisOutput
from ui.charts import build_chart
from ui.sidebar import can_query, increment_query_count


def render_chat_message(role: str, content: str, analysis: AnalysisOutput | None = None):
    with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
        st.markdown(content)
        if analysis:
            _render_analysis_result(analysis)


def _render_analysis_result(analysis: AnalysisOutput):
    # Sub-results (multi-subquery)
    if analysis.sub_results:
        for sub in analysis.sub_results:
            with st.container():
                st.markdown(f"**[{sub.index}] {sub.question}**")
                st.markdown(sub.answer)
                if sub.chart_config:
                    try:
                        fig = build_chart(sub.chart_config)
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_{sub.index}_{id(sub)}")
                    except Exception as e:
                        st.warning(f"Chart render error: {e}")
                if sub.sql_used:
                    with st.expander("SQL used"):
                        st.code(sub.sql_used, language="sql")
                if sub.export_path:
                    with open(sub.export_path, "rb") as f:
                        st.download_button(
                            "📥 Download Excel",
                            data=f.read(),
                            file_name=sub.export_path.split("/")[-1],
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"dl_{sub.index}_{id(sub)}",
                        )

    # Top-level export
    if analysis.export_path and not any(s.export_path for s in analysis.sub_results):
        try:
            with open(analysis.export_path, "rb") as f:
                st.download_button(
                    "📥 Download Excel",
                    data=f.read(),
                    file_name=analysis.export_path.split("/")[-1],
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_main_{id(analysis)}",
                )
        except Exception:
            pass

    # Datasets used
    if analysis.datasets_used:
        st.caption(f"Datasets used: {', '.join(f'`{d}`' for d in analysis.datasets_used)}")


def render_chat_tab():
    st.header("💬 Ask Your Data")

    # Re-render chat history
    for msg in st.session_state.get("chat_history", []):
        role = msg["role"]
        analysis = msg.get("analysis")
        render_chat_message(role, msg["content"], analysis)

    # Input
    query = st.chat_input("Ask anything about your data...", key="chat_input")
    if not query:
        return

    # Guest quota check
    if not can_query():
        st.error("⛔ Guest session limit (5 queries) reached. Reload the page for a new session.")
        return

    # Show user message immediately
    render_chat_message("user", query)
    st.session_state["chat_history"].append({"role": "user", "content": query})

    # Run agent with live streaming
    with st.chat_message("assistant", avatar="🤖"):
        steps_container = st.empty()
        with st.spinner(""):
            analysis = run_agent(
                query=query,
                chat_history=st.session_state.get("lc_history", []),
                step_container=steps_container,
            )

        steps_container.empty()  # clear step display after done

        if analysis:
            st.markdown(analysis.answer)
            _render_analysis_result(analysis)

            # Update LangChain history
            st.session_state["lc_history"].append(HumanMessage(content=query))
            st.session_state["lc_history"].append(AIMessage(content=analysis.answer))

            # Save to chat history for re-render
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": analysis.answer,
                "analysis": analysis,
            })
        else:
            st.error("Agent returned no response. Please try again.")

    increment_query_count()
```

- [ ] **Step 2: Commit**

```bash
git add ui/chat.py
git commit -m "chat ui"
```

---

## Task 14: Data Explorer + History Tabs

**Files:**
- Create: `ui/explorer.py`
- Create: `ui/history.py`

- [ ] **Step 1: Write ui/explorer.py**

```python
import os
import json
import streamlit as st
import pandas as pd
from db.postgres import list_datasets_from_db, get_engine
from core.exporter import export_to_excel


def render_explorer_tab():
    st.header("🔍 Data Explorer")

    try:
        datasets = list_datasets_from_db()
    except Exception as e:
        st.error(f"Could not load datasets: {e}")
        return

    if not datasets:
        st.info("No datasets loaded. Upload a CSV from the sidebar.")
        return

    slugs = [d["slug"] for d in datasets]
    selected = st.selectbox("Select dataset", slugs, key="explorer_table_select")

    if not selected:
        return

    ds_meta = next((d for d in datasets if d["slug"] == selected), {})
    cols_meta = ds_meta.get("columns_json") or []
    if isinstance(cols_meta, str):
        cols_meta = json.loads(cols_meta)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption(f"**{ds_meta.get('name', selected)}** — {ds_meta.get('row_count', '?'):,} rows")
    with col2:
        page_size = st.selectbox("Rows per page", [25, 50, 100, 250], index=1, key="explorer_page_size")

    # Load data
    try:
        engine = get_engine()
        df = pd.read_sql_table(selected, engine)
        engine.dispose()
    except Exception as e:
        st.error(f"Could not load table `{selected}`: {e}")
        return

    # Filter
    search = st.text_input("Filter (pandas query expression)", placeholder='e.g. region == "North" and revenue > 500', key="explorer_filter")
    if search:
        try:
            df = df.query(search)
            st.caption(f"Filtered: {len(df):,} rows")
        except Exception as e:
            st.warning(f"Filter error: {e}")

    # Pagination
    total = len(df)
    max_page = max(1, (total - 1) // page_size + 1)
    page = st.number_input("Page", min_value=1, max_value=max_page, value=1, key="explorer_page")
    start = (page - 1) * page_size
    st.dataframe(df.iloc[start: start + page_size], use_container_width=True, hide_index=True)
    st.caption(f"Showing rows {start + 1}–{min(start + page_size, total)} of {total:,}")

    # Export
    if st.button("📥 Export to Excel", key="explorer_export"):
        path = export_to_excel(df, title=f"{selected} Export")
        with open(path, "rb") as f:
            st.download_button(
                "Download Excel",
                data=f.read(),
                file_name=f"{selected}_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="explorer_dl",
            )
```

- [ ] **Step 2: Write ui/history.py**

```python
import os
import json
import streamlit as st


def render_history_tab():
    st.header("📜 Session History")

    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        st.info("No sessions saved yet.")
        return

    log_files = sorted(
        [f for f in os.listdir(logs_dir) if f.endswith(".txt")],
        reverse=True,
    )

    if not log_files:
        st.info("No sessions saved yet.")
        return

    st.caption(f"{len(log_files)} saved sessions")
    for fname in log_files[:20]:
        path = os.path.join(logs_dir, fname)
        ts = fname.replace("session_", "").replace(".txt", "")
        try:
            formatted = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}:{ts[13:15]}"
        except Exception:
            formatted = fname

        with st.expander(f"Session: {formatted}"):
            with open(path) as f:
                content = f.read()
            st.text(content[:2000] + ("..." if len(content) > 2000 else ""))
            if st.button(f"Delete", key=f"del_{fname}"):
                os.remove(path)
                st.rerun()
```

- [ ] **Step 3: Commit**

```bash
git add ui/explorer.py ui/history.py
git commit -m "explorer and history tabs"
```

---

## Task 15: Main App Entry Point

**Files:**
- Create: `app.py`

- [ ] **Step 1: Write app.py**

```python
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS: tighten padding, style chat bubbles, code blocks
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stChatMessage { border-radius: 12px; margin-bottom: 0.5rem; }
    .stChatMessage[data-testid="chat-message-user"] { background: #1a2535; }
    .stChatMessage[data-testid="chat-message-assistant"] { background: #0e1721; }
    .stExpander { border: 1px solid #2a3a4a; border-radius: 8px; }
    code { background: #1e2a3a !important; }
    .stDownloadButton button { background: #1f6feb; color: white; border: none; }
    .stDownloadButton button:hover { background: #388bfd; }
    h1, h2, h3 { color: #e6edf3; }
</style>
""", unsafe_allow_html=True)

from ui.sidebar import render_sidebar
from ui.chat import render_chat_tab
from ui.explorer import render_explorer_tab
from ui.history import render_history_tab

render_sidebar()

tab_chat, tab_explore, tab_history = st.tabs(["💬 Chat", "🔍 Explorer", "📜 History"])

with tab_chat:
    render_chat_tab()

with tab_explore:
    render_explorer_tab()

with tab_history:
    render_history_tab()
```

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "main app"
```

---

## Task 16: Database Init & End-to-End Smoke Test

**Files:**
- Modify: `setup_db.py` (already written in Task 2, verify it runs)

- [ ] **Step 1: Create scripts/gen_demo_data.py if not done in Task 3**

```python
# scripts/gen_demo_data.py
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
n = 2000
dates = pd.date_range("2023-01-01", "2024-12-31", periods=n)
regions = np.random.choice(["North", "South", "East", "West"], n)
categories = np.random.choice(["Electronics", "Clothing", "Books", "Home", "Sports"], n)
products_map = {
    "Electronics": ["Laptop", "Phone", "Tablet", "Headphones"],
    "Clothing": ["T-Shirt", "Jeans", "Dress", "Jacket"],
    "Books": ["Fiction", "Non-Fiction", "Textbook", "Comic"],
    "Home": ["Lamp", "Chair", "Table", "Rug"],
    "Sports": ["Shoes", "Ball", "Racket", "Gloves"],
}
product_names = [np.random.choice(products_map[c]) for c in categories]
base_rev = {"Electronics": 500, "Clothing": 80, "Books": 25, "Home": 150, "Sports": 60}
revenue = [max(5.0, round(base_rev[c] * np.random.uniform(0.5, 2.5), 2)) for c in categories]
units = [max(1, int(r / base_rev[c] * np.random.randint(1, 10))) for r, c in zip(revenue, categories)]
for i in np.random.choice(n, 30, replace=False):
    revenue[i] = round(revenue[i] * np.random.choice([0.05, 8.0]), 2)

df = pd.DataFrame({
    "order_date": dates,
    "region": regions,
    "category": categories,
    "product": product_names,
    "revenue": revenue,
    "units": units,
    "discount": np.round(np.random.uniform(0, 0.4, n), 2),
    "profit": [round(r * np.random.uniform(0.05, 0.35), 2) for r in revenue],
})
Path("data/demo").mkdir(parents=True, exist_ok=True)
df.to_csv("data/demo/ecommerce_sales.csv", index=False)
print(f"ecommerce_sales.csv: {len(df)} rows")

# global_superstore
n2 = 1500
df2 = pd.DataFrame({
    "order_date": pd.date_range("2021-01-01", periods=n2, freq="8h"),
    "segment": np.random.choice(["Consumer", "Corporate", "Home Office"], n2),
    "country": np.random.choice(["United States", "Canada", "Mexico"], n2),
    "region": np.random.choice(["East", "West", "Central", "South"], n2),
    "category": np.random.choice(["Furniture", "Office Supplies", "Technology"], n2),
    "sub_category": np.random.choice(["Chairs", "Phones", "Binders", "Tables", "Storage", "Copiers"], n2),
    "product_name": [f"Product_{i % 200}" for i in range(n2)],
    "sales": np.round(np.random.exponential(250, n2), 2),
    "quantity": np.random.randint(1, 15, n2),
    "discount": np.round(np.random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5], n2), 1),
    "profit": np.round(np.random.normal(50, 120, n2), 2),
    "shipping_cost": np.round(np.random.exponential(15, n2), 2),
})
df2.to_csv("data/demo/global_superstore.csv", index=False)
print(f"global_superstore.csv: {len(df2)} rows")
```

- [ ] **Step 2: Generate demo data**

```bash
cd "/Users/pradhyumnakasula/data analysis agent" && python scripts/gen_demo_data.py
```

Expected:
```
ecommerce_sales.csv: 2000 rows
global_superstore.csv: 1500 rows
```

- [ ] **Step 3: Run DB setup**

```bash
python setup_db.py
```

Expected:
```
Database data_analyst already exists.  (or Created database: data_analyst)
Tables created.
Loaded demo: ecommerce_sales.csv → table 'ecommerce_sales' (2000 rows)
Loaded demo: global_superstore.csv → table 'global_superstore' (1500 rows)
Setup complete. Run: streamlit run app.py
```

- [ ] **Step 4: Verify DB**

```bash
psql postgres -c "\c data_analyst" -c "\dt" 2>/dev/null || psql data_analyst -c "\dt"
```

Expected: tables `datasets`, `ecommerce_sales`, `global_superstore` listed.

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: all tests PASS (some may require mocking without real DB).

- [ ] **Step 6: Set OPENAI_API_KEY in .env and launch app**

```bash
# Edit .env and set your real key, then:
streamlit run app.py
```

Expected: Streamlit opens in browser at http://localhost:8501. Sidebar shows demo datasets. Chat tab has input box.

- [ ] **Step 7: Smoke test — chat with demo data**

In the running app:
1. Enter owner password (set in `.streamlit/secrets.toml`)
2. Type: `What is the total revenue by region in ecommerce_sales?`
3. Verify: agent steps stream live, bar chart appears, answer contains numbers
4. Type: `Show revenue trend over time AND detect anomalies in revenue`
5. Verify: two separate results labeled [1] and [2], two charts

- [ ] **Step 8: Final commit**

```bash
git add scripts/ data/demo/ .streamlit/ .env.example
git commit -m "e2e setup"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| OpenAI gpt-4o-mini | Task 10 (agent.py) |
| PostgreSQL storage | Tasks 2, 4 |
| Upload via UI | Task 12 (sidebar.py) |
| Multi-dataset support | Tools: list_datasets, postgres.py |
| Multi-subquery decomposition | Task 8 (prompts.py) + Task 13 (chat.py) |
| Streaming agent steps | Task 10 (StreamlitStepHandler) |
| Plotly interactive charts | Task 11 (charts.py) |
| 7 chart types | Task 11 |
| Anomaly detection | Task 6, Tool: detect_anomalies_tool |
| Trend detection | Task 6, Tool: detect_trends_tool |
| Excel export | Task 7, Tool: export_data |
| Vector semantic search | Task 5, Tool: search_metadata_tool |
| Guest rate limiting (5/session) | Task 12 (sidebar.py) |
| Owner mode (password) | Task 12 (sidebar.py) |
| Demo datasets (2) | Task 3, Task 16 |
| Demo schema visible to guests | Task 12 (sidebar) |
| Data Explorer tab | Task 14 (explorer.py) |
| History tab | Task 14 (history.py) |
| Auto-profiling on upload | Task 4 (dataset_profiler.py) |
| Chroma indexing | Task 5 (vector_store.py) |
| Session logging | Tool: save_session |
| Streamlit Cloud ready | .env + st.secrets pattern throughout |

All spec requirements covered. No gaps found.
