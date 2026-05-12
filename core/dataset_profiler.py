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
