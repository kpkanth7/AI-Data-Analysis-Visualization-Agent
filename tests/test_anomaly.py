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


def test_detect_trends_returns_rolling_avg():
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
    assert (anomalies["severity"] > 0).all()


def test_both_method_combines_flags():
    data = pd.DataFrame({"value": list(range(100)) + [99999]})
    result_both = detect_anomalies(data, "value", method="both")
    result_iqr = detect_anomalies(data, "value", method="iqr")
    # both should flag at least as many as iqr alone
    assert result_both["is_anomaly"].sum() >= result_iqr["is_anomaly"].sum()


def test_detect_trends_sorted_by_date():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({"date": dates[::-1], "val": [50, 40, 30, 20, 10]})
    result = detect_trends(df, "date", "val")
    # after sort, values should be ascending
    assert result["val"].iloc[0] == 10
