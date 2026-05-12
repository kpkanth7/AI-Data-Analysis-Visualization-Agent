import pandas as pd
import pytest
import os
from core.exporter import export_to_excel, export_to_csv


def test_export_excel_creates_file(tmp_path):
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    path = str(tmp_path / "test_export.xlsx")
    result = export_to_excel(df, path)
    assert os.path.exists(result)
    loaded = pd.read_excel(result)
    assert list(loaded.columns) == ["col1", "col2"]
    assert len(loaded) == 3


def test_export_csv_creates_file(tmp_path):
    df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
    path = str(tmp_path / "test.csv")
    result = export_to_csv(df, path)
    assert os.path.exists(result)
    loaded = pd.read_csv(result)
    assert list(loaded.columns) == ["x", "y"]


def test_export_excel_styled(tmp_path):
    df = pd.DataFrame({"revenue": [1000.5, 2000.75], "region": ["North", "South"]})
    path = str(tmp_path / "styled.xlsx")
    result = export_to_excel(df, path, title="Revenue Report")
    assert os.path.exists(result)
    import openpyxl
    wb = openpyxl.load_workbook(result)
    ws = wb.active
    # header row should have dark blue fill
    assert ws["A1"].fill.fgColor.rgb == "001E3A5F"


def test_export_excel_default_path():
    df = pd.DataFrame({"a": [1]})
    result = export_to_excel(df)
    assert result.startswith("exports/")
    assert result.endswith(".xlsx")
    assert os.path.exists(result)
    os.remove(result)


def test_export_csv_default_path():
    df = pd.DataFrame({"a": [1]})
    result = export_to_csv(df)
    assert result.startswith("exports/")
    assert result.endswith(".csv")
    assert os.path.exists(result)
    os.remove(result)
