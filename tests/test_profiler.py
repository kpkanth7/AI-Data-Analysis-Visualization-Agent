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
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    from unittest.mock import patch, MagicMock
    with patch("core.dataset_profiler.register_dataset"), \
         patch("core.dataset_profiler.get_engine") as mock_eng, \
         patch("core.dataset_profiler.index_dataset_in_chroma"):
        mock_engine_instance = MagicMock()
        mock_eng.return_value = mock_engine_instance
        # patch df.to_sql so it doesn't hit real DB
        with patch.object(df.__class__, "to_sql"):
            slug = ingest_dataframe(df, "my_data.csv")
    assert slug == "my_data"

def test_profile_empty_dataframe():
    df = pd.DataFrame({"x": pd.Series([], dtype=float)})
    profile = profile_dataframe(df)
    assert profile["x"]["null_pct"] == 0.0
    assert profile["x"]["unique_count"] == 0
