import pandas as pd
import pytest
import plotly.graph_objects as go
from ui.charts import build_chart


def test_line_chart_returns_figure():
    data = [{"date": "2024-01", "revenue": 100}, {"date": "2024-02", "revenue": 200}]
    fig = build_chart({"chart_type": "line", "data": data, "x": "date", "y": "revenue", "title": "Test"})
    assert isinstance(fig, go.Figure)


def test_bar_chart_returns_figure():
    data = [{"region": "North", "sales": 500}, {"region": "South", "sales": 300}]
    fig = build_chart({"chart_type": "bar", "data": data, "x": "region", "y": "sales", "title": "By Region"})
    assert isinstance(fig, go.Figure)


def test_histogram_chart():
    data = [{"value": i} for i in range(50)]
    fig = build_chart({"chart_type": "histogram", "data": data, "x": "value", "y": "value", "title": "Dist"})
    assert isinstance(fig, go.Figure)


def test_heatmap_chart():
    data = [{"a": i, "b": i * 2, "c": i * 3} for i in range(10)]
    fig = build_chart({"chart_type": "heatmap", "data": data, "x": "a", "y": "b", "title": "Corr"})
    assert isinstance(fig, go.Figure)


def test_pie_chart():
    data = [{"category": "A", "amount": 30}, {"category": "B", "amount": 70}]
    fig = build_chart({"chart_type": "pie", "data": data, "x": "category", "y": "amount", "title": "Share"})
    assert isinstance(fig, go.Figure)


def test_anomaly_chart():
    data = [{"idx": i, "value": i * 10, "is_anomaly": i == 5} for i in range(10)]
    fig = build_chart({"chart_type": "anomaly", "data": data, "x": "idx", "y": "value",
                       "anomaly_col": "is_anomaly", "title": "Anomalies"})
    assert isinstance(fig, go.Figure)
    # should have 2 traces: normal + anomaly
    assert len(fig.data) == 2


def test_unknown_chart_type_falls_back_to_bar():
    # Unknown types fall back to bar rather than crashing the UI
    fig = build_chart({"chart_type": "radar", "data": [{"a": 1, "b": 2}], "x": "a", "y": "b", "title": ""})
    assert isinstance(fig, go.Figure)


def test_build_chart_accepts_json_string():
    import json
    config = json.dumps({"chart_type": "bar", "data": [{"x": 1, "y": 2}], "x": "x", "y": "y", "title": ""})
    fig = build_chart(config)
    assert isinstance(fig, go.Figure)
