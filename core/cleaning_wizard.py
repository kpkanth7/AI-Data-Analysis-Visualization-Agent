"""Pure-data cleaning ops. UI lives in ui/cleaning_wizard_ui.py."""
import pandas as pd
import numpy as np


def profile_for_cleaning(df: pd.DataFrame) -> dict:
    """Compact summary used by the wizard UI."""
    n_rows = len(df)
    cols = []
    for c in df.columns:
        s = df[c]
        is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s)
        is_dt = pd.api.types.is_datetime64_any_dtype(s)
        nulls = int(s.isna().sum())
        cols.append({
            "name": c,
            "dtype": str(s.dtype),
            "is_numeric": bool(is_num),
            "is_datetime": bool(is_dt),
            "null_count": nulls,
            "null_pct": round(nulls / n_rows * 100, 1) if n_rows else 0.0,
            "unique_count": int(s.nunique(dropna=True)),
            "sample": [str(v) for v in s.dropna().head(3).tolist()],
        })
    return {
        "rows": n_rows,
        "cols": cols,
        "dup_count": int(df.duplicated().sum()),
        "n_cols": len(df.columns),
    }


def apply_duplicates(df: pd.DataFrame, choice: str) -> pd.DataFrame:
    if choice == "keep_first":
        return df.drop_duplicates(keep="first")
    if choice == "keep_last":
        return df.drop_duplicates(keep="last")
    if choice == "drop_all":
        return df.drop_duplicates(keep=False)
    return df


def apply_nulls(df: pd.DataFrame, plan: dict[str, dict]) -> pd.DataFrame:
    """plan[col] = {"action": "drop_row|fill_mean|fill_median|fill_mode|fill_const|leave",
                     "const": <value if fill_const>}"""
    df = df.copy()
    drop_cols = [c for c, p in plan.items() if p.get("action") == "drop_row"]
    if drop_cols:
        df = df.dropna(subset=drop_cols)
    for col, p in plan.items():
        action = p.get("action", "leave")
        if action == "fill_mean" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        elif action == "fill_median" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif action == "fill_mode":
            mode = df[col].mode(dropna=True)
            if len(mode):
                df[col] = df[col].fillna(mode.iloc[0])
        elif action == "fill_const":
            df[col] = df[col].fillna(p.get("const", ""))
    return df


def apply_types(df: pd.DataFrame, plan: dict[str, str]) -> pd.DataFrame:
    """plan[col] = "datetime|numeric|string|keep" """
    df = df.copy()
    for col, target in plan.items():
        if target == "datetime":
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
        elif target == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif target == "string":
            df[col] = df[col].astype(str)
    return df


def apply_outliers(df: pd.DataFrame, plan: dict[str, dict], method: str = "iqr",
                   z_thresh: float = 3.0) -> pd.DataFrame:
    """plan[col] = {"action": "clip|drop|leave"} for numeric cols only."""
    df = df.copy()
    drop_mask = pd.Series(False, index=df.index)
    for col, p in plan.items():
        action = p.get("action", "leave")
        if action == "leave" or col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        s = df[col]
        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        else:  # z-score
            mu, sd = s.mean(), s.std()
            lo, hi = mu - z_thresh * sd, mu + z_thresh * sd
        if action == "clip":
            df[col] = s.clip(lower=lo, upper=hi)
        elif action == "drop":
            drop_mask |= (s < lo) | (s > hi)
    if drop_mask.any():
        df = df[~drop_mask]
    return df


def run_cleaning_plan(df: pd.DataFrame, plan: dict) -> pd.DataFrame:
    """Apply a full plan dict in order."""
    out = df.copy()
    if "duplicates" in plan:
        out = apply_duplicates(out, plan["duplicates"])
    if "types" in plan:
        out = apply_types(out, plan["types"])
    if "nulls" in plan:
        out = apply_nulls(out, plan["nulls"])
    if "outliers" in plan:
        out = apply_outliers(
            out,
            plan["outliers"].get("cols", {}),
            method=plan["outliers"].get("method", "iqr"),
            z_thresh=plan["outliers"].get("z_thresh", 3.0),
        )
    return out
