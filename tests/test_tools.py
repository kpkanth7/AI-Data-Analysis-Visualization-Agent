import pytest
import json
from unittest.mock import patch, MagicMock
import pandas as pd


def test_list_datasets_tool_returns_table_info():
    with patch("agent.tools.list_datasets_from_db") as mock_db:
        mock_db.return_value = [{
            "name": "sales.csv",
            "slug": "sales",
            "row_count": 100,
            "columns_json": [{"name": "revenue", "dtype": "float64"}],
            "is_demo": False,
        }]
        from agent.tools import list_datasets
        result = list_datasets.invoke({"dummy": ""})
        assert "sales" in result
        assert "revenue" in result


def test_list_datasets_empty():
    with patch("agent.tools.list_datasets_from_db") as mock_db:
        mock_db.return_value = []
        from agent.tools import list_datasets
        result = list_datasets.invoke({"dummy": ""})
        assert "No datasets" in result


def test_compute_stats_returns_json():
    mock_df = pd.DataFrame({"revenue": [100.0, 200.0, 300.0]})
    with patch("agent.tools.get_engine") as mock_eng, \
         patch("agent.tools.pd.read_sql_table", return_value=mock_df), \
         patch("agent.tools._accessible_slugs", return_value={"sales"}):
        mock_eng.return_value = MagicMock()
        from agent.tools import compute_stats
        result = compute_stats.invoke({"table": "sales", "column": "revenue", "metrics": ["sum", "avg"]})
        parsed = json.loads(result)
        assert parsed["stats"]["sum"] == pytest.approx(600.0)
        assert parsed["stats"]["avg"] == pytest.approx(200.0)


def test_query_sql_blocks_unsafe():
    with patch("agent.tools.execute_sql") as mock_sql:
        mock_sql.side_effect = ValueError("Only SELECT statements are allowed.")
        from agent.tools import query_sql
        result = query_sql.invoke({"sql": "DROP TABLE users"})
        assert "SQL error" in result or "Only SELECT" in result


def test_create_visualization_returns_config():
    from agent.tools import create_visualization
    data = [{"date": "2024-01", "revenue": 100}]
    result = create_visualization.invoke({
        "chart_type": "line",
        "data": data,
        "x": "date",
        "y": "revenue",
        "title": "Test Chart",
    })
    parsed = json.loads(result)
    assert parsed["chart_type"] == "line"
    assert parsed["x"] == "date"


def test_save_session_creates_file(tmp_path):
    """save_session moved to core/session_manager. Verify it saves a JSON file."""
    from core.session_manager import save_session
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    import os, pathlib
    # Redirect SESSIONS_DIR to tmp_path
    import core.session_manager as sm
    orig_dir = sm.SESSIONS_DIR
    sm.SESSIONS_DIR = pathlib.Path(tmp_path)
    try:
        path = save_session(history, label="test")
        assert os.path.exists(path)
        data = json.loads(open(path).read())
        assert len(data["messages"]) == 2
    finally:
        sm.SESSIONS_DIR = orig_dir
