import json
import datetime
import os

import pandas as pd
from langchain.tools import tool

from db.postgres import execute_sql, list_datasets_from_db, get_engine
from db.vector_store import search_metadata
from core.anomaly import detect_anomalies, detect_trends
from core.exporter import export_to_excel


@tool
def list_datasets(dummy: str = "") -> str:
    """List all available datasets (tables) with their schemas. Always call this first."""
    try:
        datasets = list_datasets_from_db()
        if not datasets:
            return "No datasets loaded yet. Ask the user to upload a CSV file."
        lines = []
        for ds in datasets:
            cols = ds.get("columns_json") or []
            if isinstance(cols, str):
                cols = json.loads(cols)
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
            return json.dumps({"total_rows": 0, "preview": [], "all_rows": []})
        return json.dumps({"total_rows": len(rows), "preview": rows[:5], "all_rows": rows}, default=str)
    except Exception as e:
        return f"SQL error: {e}"


@tool
def query_pandas(table: str, expression: str) -> str:
    """Run a pandas .query() expression on a table. Good for complex string/date filters."""
    try:
        engine = get_engine()
        df = pd.read_sql_table(table, engine)
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
    except Exception as e:
        return f"Stats error: {e}"


@tool
def detect_anomalies_tool(table: str, column: str, method: str = "both") -> str:
    """Detect anomalies/outliers in a numeric column using z-score and IQR methods."""
    try:
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
    except Exception as e:
        return f"Anomaly detection error: {e}"


@tool
def detect_trends_tool(table: str, date_col: str, value_col: str, window: int = 7) -> str:
    """Detect trends in time-series data. Returns rolling average and growth rates."""
    try:
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
    except Exception as e:
        return f"Trend detection error: {e}"


@tool
def create_visualization(
    chart_type: str,
    data: list,
    x: str,
    y: str,
    title: str = "",
    color: str = None,
    anomaly_col: str = None,
) -> str:
    """
    Create a Plotly chart specification.
    chart_type: line | bar | scatter | histogram | heatmap | pie | anomaly | box
    Returns JSON config that the UI will render as an interactive Plotly chart.
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
def export_data(table: str, sql: str = None) -> str:
    """Export a table or SQL query result to a styled Excel file. Returns the file path."""
    try:
        engine = get_engine()
        if sql:
            df = pd.read_sql(sql, engine)
        else:
            df = pd.read_sql_table(table, engine)
        path = export_to_excel(df, title=f"{table} Export")
        return json.dumps({"export_path": path, "rows": len(df), "columns": list(df.columns)})
    except Exception as e:
        return f"Export error: {e}"


@tool
def save_session(conversation: str) -> str:
    """Save the current conversation to a log file."""
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
