import json
import datetime
import os
import threading
from typing import Optional

import pandas as pd
from langchain.tools import tool

from db.postgres import execute_sql, list_datasets_from_db, get_engine, MAX_QUERY_ROWS
from db.vector_store import search_metadata
from core.anomaly import detect_anomalies, detect_trends
from core.exporter import export_to_excel

# Thread-local session context set by run_agent before each invocation
_ctx = threading.local()


def set_session_context(is_owner: bool = False, session_id: str | None = None) -> None:
    _ctx.is_owner = is_owner
    _ctx.session_id = session_id


def _get_ctx() -> dict:
    return {
        "is_owner": getattr(_ctx, "is_owner", False),
        "session_id": getattr(_ctx, "session_id", None),
    }


def _accessible_slugs() -> set[str]:
    """Return the set of table slugs the current session can query."""
    ctx = _get_ctx()
    datasets = list_datasets_from_db(
        include_owner_only=ctx["is_owner"],
        session_id=ctx["session_id"],
    )
    return {d["slug"] for d in datasets}


def _guard_table(table: str) -> None:
    """Raise if the table is not accessible in the current session."""
    allowed = _accessible_slugs()
    if table not in allowed:
        raise PermissionError(
            f"Table '{table}' is not accessible in this session. "
            f"Available: {sorted(allowed)}"
        )


@tool
def list_datasets(dummy: str = "") -> str:
    """List all datasets available in this session with their schemas. Always call this first."""
    try:
        ctx = _get_ctx()
        datasets = list_datasets_from_db(
            include_owner_only=ctx["is_owner"],
            session_id=ctx["session_id"],
        )
        if not datasets:
            return "No datasets loaded yet. Upload a CSV or Excel file to get started."
        lines = []
        for ds in datasets:
            cols = ds.get("columns_json") or []
            if isinstance(cols, str):
                cols = json.loads(cols)
            col_summary = ", ".join(f"{c['name']} ({c['dtype']})" for c in cols[:10])
            tag = " [demo]" if ds.get("is_demo") else (" [owner]" if ds.get("owner_only") else " [session]")
            lines.append(f"• {ds['slug']}{tag}: {ds['row_count']} rows | columns: {col_summary}")
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
    """Run a SELECT SQL query against PostgreSQL. Returns up to MAX_QUERY_ROWS rows as JSON.
    If `truncated` is true, the underlying result exceeded the cap — add LIMIT or aggregate."""
    try:
        rows = execute_sql(sql)
        truncated = len(rows) > MAX_QUERY_ROWS
        if truncated:
            rows = rows[:MAX_QUERY_ROWS]
        if not rows:
            return json.dumps({"total_rows": 0, "preview": [], "all_rows": [], "truncated": False})
        return json.dumps({
            "total_rows": len(rows),
            "preview": rows[:10],
            "all_rows": rows,
            "truncated": truncated,
            "row_cap": MAX_QUERY_ROWS,
        }, default=str)
    except Exception as e:
        return f"SQL error: {e}"


@tool
def query_pandas(table: str, expression: str) -> str:
    """Run a pandas .query() expression on a table. Good for complex string/date filters."""
    try:
        _guard_table(table)
        engine = get_engine()
        df = pd.read_sql_table(table, engine)
        result = df.query(expression)
        rows = result.head(100).to_dict(orient="records")
        return json.dumps({"total_rows": len(result), "preview": rows[:10], "all_rows": rows}, default=str)
    except PermissionError as e:
        return f"Access denied: {e}"
    except Exception as e:
        return f"Pandas query error: {e}"


@tool
def compute_stats(table: str, column: str, metrics: Optional[list[str]] = None) -> str:
    """Compute statistics (sum, avg, median, std, min, max, count) on a numeric column."""
    if metrics is None:
        metrics = ["sum", "avg", "min", "max", "count", "std"]
    try:
        _guard_table(table)
        engine = get_engine()
        df = pd.read_sql_table(table, engine)
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        metric_map = {
            "sum": lambda s: float(s.sum()),
            "avg": lambda s: float(s.mean()),
            "median": lambda s: float(s.median()),
            "std": lambda s: float(s.std()),
            "min": lambda s: float(s.min()),
            "max": lambda s: float(s.max()),
            "count": lambda s: int(s.count()),
        }
        result = {m: metric_map[m](series) for m in metrics if m in metric_map}
        return json.dumps({"column": column, "table": table, "stats": result}, default=str)
    except PermissionError as e:
        return f"Access denied: {e}"
    except Exception as e:
        return f"Stats error: {e}"


@tool
def detect_anomalies_tool(table: str, column: str, method: str = "both") -> str:
    """Detect anomalies/outliers in a numeric column using z-score and IQR methods."""
    try:
        _guard_table(table)
        engine = get_engine()
        df = pd.read_sql_table(table, engine)
        result = detect_anomalies(df, column, method=method)
        anomalies = result[result["is_anomaly"]].head(50)
        anomaly_count = int(result["is_anomaly"].sum())
        return json.dumps({
            "total_rows": len(result),
            "anomaly_count": anomaly_count,
            "normal_count": len(result) - anomaly_count,
            "anomaly_pct": round(anomaly_count / len(result) * 100, 2),
            "anomalies": anomalies.to_dict(orient="records"),
            "full_data_for_chart": result[["is_anomaly", column, "severity"]].to_dict(orient="records"),
        }, default=str)
    except PermissionError as e:
        return f"Access denied: {e}"
    except Exception as e:
        return f"Anomaly detection error: {e}"


@tool
def detect_trends_tool(table: str, date_col: str, value_col: str, window: int = 7) -> str:
    """Detect trends in time-series data. Returns rolling average and growth rates."""
    try:
        _guard_table(table)
        engine = get_engine()
        df = pd.read_sql_table(table, engine)
        result = detect_trends(df, date_col, value_col, window=window)
        return json.dumps({
            "total_periods": len(result),
            "avg_growth_rate": float(result["growth_rate"].mean()),
            "max_growth_rate": float(result["growth_rate"].max()),
            "min_growth_rate": float(result["growth_rate"].min()),
            "data": result[[date_col, value_col, "rolling_avg", "growth_rate"]].head(200).to_dict(orient="records"),
        }, default=str)
    except PermissionError as e:
        return f"Access denied: {e}"
    except Exception as e:
        return f"Trend detection error: {e}"


@tool
def create_visualization(
    chart_type: str,
    data: list,
    title: str = "",
    x: Optional[str] = None,
    y: Optional[str] = None,
    color: Optional[str] = None,
    anomaly_col: Optional[str] = None,
) -> str:
    """
    Create a Plotly chart specification.
    chart_type: line | bar | stacked_bar | scatter | histogram | heatmap | pie | anomaly | box
    x: column for x-axis (or pie names column). Optional for histogram/heatmap.
    y: column for y-axis (or pie values column). Optional for histogram/heatmap.
    color: grouping column for bar/stacked_bar/line (creates grouped or stacked series).
    stacked_bar: use when showing how parts compose a whole across categories or time (requires color=).
    bar with color=: use for side-by-side grouped comparison across two dimensions.
    IMPORTANT: The returned JSON string must be included as chart_config in your final output.
    Only call this when a chart genuinely helps — NOT for single-number answers or simple text lookups.
    """
    config = {
        "chart_type": chart_type,
        "data": data[:500],
        "x": x,
        "y": y,
        "title": title,
        "color": color,
        "anomaly_col": anomaly_col,
    }
    return json.dumps(config, default=str)


@tool
def export_data(table: str, sql: Optional[str] = None) -> str:
    """Export a table or SQL query result to a styled Excel file. Returns the file path."""
    try:
        _guard_table(table)
        engine = get_engine()
        if sql:
            df = pd.read_sql(sql, engine)
        else:
            df = pd.read_sql_table(table, engine)
        path = export_to_excel(df, title=f"{table} Export")
        return json.dumps({"export_path": path, "rows": len(df), "columns": list(df.columns)})
    except PermissionError as e:
        return f"Access denied: {e}"
    except Exception as e:
        return f"Export error: {e}"


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
]
